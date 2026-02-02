# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor

@dataclass
class MambaMOELayerSubmodules:
    """
    Configuration class for specifying the submodules of a Mamba layer.

    This class defines the structure and default implementations for various
    components of a Mamba layer, allowing for flexible customization of the
    layer's architecture.

    Args:
        norm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        mixer (Union[ModuleSpec, type]): Specification for the along-sequence mixing mechanism.
        mamba_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the mixer.
    """

    norm: Union[ModuleSpec, type] = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityOp
    # mlp
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class MambaMOELayer(MegatronModule):
    """
    A single Mamba layer.

    Mamba layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaMOELayerSubmodules,
        mamba_ssm_ngroups=8,
        layer_number: int = 1,
        residual_in_fp32=False,
    ):
        """Initialize Mamba Layer."""
        super().__init__(config)
        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = config.hidden_dropout
        self.mixer = build_module(
            submodules.mixer,
            self.config,
            d_model=self.config.hidden_size,
            ngroups=mamba_ssm_ngroups,
            layer_number=layer_number,
        )
        self.norm = build_module(submodules.norm, self.config, self.config.hidden_size)
        self.mamba_bda = build_module(submodules.mamba_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

        # [Module 0: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # [Module 1: MLP block]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 2: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):
        """
        Perform a forward pass through the Mamba layer.

        This method implements the core computation of a Mamba layer, including
        the convolution and the selective SSM/SSD.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention. Not used by this layer.
            inference_params (object, optional): Parameters for inference-time optimizations.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        residual = hidden_states
        # from megatron.training import print_rank_0
        # print_rank_0("### MambaMOELayer forward rank: {}, layer_num: {}, input hidden_states: {}, shape: {}\n".format(torch.distributed.get_rank(), self.layer_number, hidden_states, hidden_states.shape))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        hidden_states = self.norm(hidden_states)
        # print_rank_0("### MambaMOELayer forward rank: {}, layer_num: {}, after norm hidden_states: {}\n".format(torch.distributed.get_rank(), self.layer_number, hidden_states))

        mixer_out_with_bias = self.mixer(hidden_states, inference_params=inference_params)
        # print_rank_0("### MambaMOELayer forward rank: {}, layer_num: {}, mixer_out_with_bias: {}\n".format(torch.distributed.get_rank(), self.layer_number, mixer_out_with_bias))

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mamba_bda(self.training, self.config.bias_dropout_fusion)(
                mixer_out_with_bias, residual, self.hidden_dropout
            )
        # print_rank_0("### MambaMOELayer forward rank: {}, layer_num: {}, hidden_states after bda: {}\n".format(torch.distributed.get_rank(), self.layer_number, hidden_states))

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        # print_rank_0("### MambaMOELayer forward rank: {}, layer_num: {}, pre_mlp_layernorm_output after mlp norm: {}\n".format(torch.distributed.get_rank(), self.layer_number, pre_mlp_layernorm_output))

        # Knowledge Layer before moe/mlp
        # if self.knowledge_attention is not None and kv_compressed is not None:
        #     knowledge_output_with_bias = self.knowledge_attention(kv_compressed)
        #     with self.bias_dropout_add_exec_handler():
        #         pre_mlp_layernorm_output = self.knowledge_attn_bda(self.training, self.config.bias_dropout_fusion)(
        #             knowledge_output_with_bias, pre_mlp_layernorm_output, self.hidden_dropout
        #         )

        # MLP.
        # print_rank_0(self.mlp)
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        # print_rank_0("### MambaMOELayer forward rank: {}, layer_num: {}, mlp_output_with_bias after mlp: {}\n".format(torch.distributed.get_rank(), self.layer_number, mlp_output_with_bias))
        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        # print_rank_0("### MambaMOELayer forward rank: {}, layer_num: {}, hidden_states after mlp_bda: {}\n".format(torch.distributed.get_rank(), self.layer_number, hidden_states))

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # CUDA graph requires returned values to be Tensors
        # if self.config.external_cuda_graph and self.training:
        #     return output
        return output, context
        # return hidden_states, context

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """Allocate the inference cache."""
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the mamba layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the mamba layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
