# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union
from copy import deepcopy

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_with_cos_sin,
)
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import divide

from .enums import AttnMaskType
from .transformer_config import MLATransformerConfig

try:
    from flash_attn import flash_attn_with_kvcache
except:
    flash_attn_with_kvcache = None


try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


@dataclass
class KnoledgeAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a self-attention.
    """

    kn_layernorm: Union[ModuleSpec, type] = None
    kn_up_proj: Union[ModuleSpec, type] = None
    kn_att_out_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None

class KnowledgeAttention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: KnoledgeAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.no_mask,
        attention_type="cross",
        cp_comm_type: str = None,
    ):
       

        self.config = deepcopy(config)
        # print("### kn att before self.config.num_attention_heads:{}\nself.config.kv_channels :{}\n".format(self.config.num_attention_heads, self.config.kv_channels ))
        
        self.config.num_attention_heads = self.config.knowledge_block_heads_num
        self.config.kv_channels = self.config.knowledge_block_heads_dim
        self.config.num_query_groups = self.config.knowledge_block_heads_num
        # self.config.num_query_groups = None
        # print("### kn att after self.config.num_attention_heads:{}\nself.config.kv_channels :{}\n".format(self.config.num_attention_heads, self.config.kv_channels ))
        super().__init__(config=self.config)
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        assert self.config.knowledge_block_freq is not None and self.config.knowledge_block_freq > 0
        assert self.config.knowledge_block_heads_dim is not None and self.config.knowledge_block_heads_dim > 0
        assert self.config.knowledge_block_heads_num is not None and self.config.knowledge_block_heads_num > 0

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.knowledge_block_heads_dim * self.config.knowledge_block_heads_num
        # self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups
        self.input_dim = self.config.kv_lora_rank
        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tensor_model_parallel_size = world_size
        self.rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.knowledge_block_heads_num
        )
        self.num_attention_heads_per_partition = divide(self.config.knowledge_block_heads_num, world_size)
        # self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        
        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'
        # Layer norm
        self.kn_layernorm = build_module(
            submodules.kn_layernorm,
            hidden_size=self.input_dim,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        # Input.
        self.kn_up_proj = build_module(
            submodules.kn_up_proj,
            self.input_dim,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        #Core att
        # print("### kn att Core att self.config:{}\n self.config.num_attention_heads:{}\nself.config.kv_channels :{}\n".format(self.config, self.config.num_attention_heads, self.config.kv_channels ))
        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=cp_comm_type,
        )


        # Output.
        self.kn_att_out_proj = build_module(
            submodules.kn_att_out_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

        # Key matrix
        self.key_matrix = torch.nn.Parameter(
            torch.empty((self.config.knowledge_block_fields_num, self.config.knowledge_block_heads_num, self.config.knowledge_block_heads_dim), dtype=torch.float32)
        )
        if self.config.perform_initialization:
            self.config.init_method(self.key_matrix)
        self.key_matrix.data = self.key_matrix.data.to(dtype=self.config.params_dtype)

        # Value matrix
        self.value_matrix = torch.nn.Parameter(
            torch.empty((self.config.knowledge_block_fields_num, self.config.knowledge_block_heads_num, self.config.knowledge_block_heads_dim), dtype=torch.float32)
        )
        if self.config.perform_initialization:
            self.config.init_method(self.value_matrix)
        self.value_matrix.data = self.value_matrix.data.to(dtype=self.config.params_dtype)

    # def _checkpointed_attention_forward(
    #     self,
    #     query,
    #     key,
    #     value,
    #     attention_mask,
    #     rotary_pos_emb=None,
    #     attn_mask_type=None,
    #     attention_bias=None,
    #     packed_seq_params=None,
    # ):
    #     """Forward method with selective activation checkpointing."""

    #     def custom_forward(*inputs):
    #         query = inputs[0]
    #         key = inputs[1]
    #         value = inputs[2]
    #         attention_mask = inputs[3]
    #         attn_mask_type = inputs[5]
    #         attn_mask_type = AttnMaskType(attn_mask_type.item())
    #         output_ = self.core_attention(
    #             query,
    #             key,
    #             value,
    #             attention_mask,
    #             attn_mask_type=attn_mask_type,
    #             attention_bias=attention_bias,
    #             packed_seq_params=packed_seq_params,
    #         )
    #         return output_

    #     if attn_mask_type is None:
    #         attn_mask_type = self.attn_mask_type
    #     attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
    #     hidden_states = tensor_parallel.checkpoint(
    #         custom_forward, False, query, key, value, attention_mask, rotary_pos_emb, attn_mask_type
    #     )

    #     return hidden_states

    # def _allocate_memory(self, inference_max_sequence_length, batch_size, dim, dtype):
    #     """Allocate memory to store kv cache during inference."""

    #     return torch.empty(
    #         inference_max_sequence_length,
    #         batch_size,
    #         self.num_query_groups_per_partition,
    #         dim,
    #         dtype=dtype,
    #         device=torch.cuda.current_device(),
    #     )

    # def _adjust_key_value_for_inference(
    #     self,
    #     inference_params: InferenceParams,
    #     query: Tensor,
    #     key: Tensor,
    #     value: Tensor,
    #     rotary_pos_emb: Tensor,
    #     rotary_pos_cos: Tensor = None,
    #     rotary_pos_sin: Tensor = None,
    # ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    #     """
    #     Saves the generated key and value tensors to the end of the buffers in inference_params.
    #     Returns the full size keys and values from the provided inference_params, as well as
    #     adjusted rotary_pos_emb.

    #     Returns a tuple: (key, value, rotary_pos_emb)

    #     """
    #     attn_mask_type = self.attn_mask_type
    #     if inference_params is None:
    #         return query, key, value, rotary_pos_emb, attn_mask_type

    #     # =================================================
    #     # Pre-allocate memory for key-values for inference.
    #     # =================================================
    #     if self.layer_number not in inference_params.key_value_memory_dict:
    #         inf_max_seq_length = inference_params.max_sequence_length
    #         inf_max_batch_size = inference_params.max_batch_size
    #         inference_key_memory = self._allocate_memory(
    #             inf_max_seq_length, inf_max_batch_size, key.shape[-1], key.dtype
    #         )
    #         inference_value_memory = self._allocate_memory(
    #             inf_max_seq_length, inf_max_batch_size, value.shape[-1], value.dtype
    #         )
    #         inference_params.key_value_memory_dict[self.layer_number] = (
    #             inference_key_memory,
    #             inference_value_memory,
    #         )
    #     else:
    #         # Get the pre-allocated buffers for this layer
    #         inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[
    #             self.layer_number
    #         ]

    #     if inference_params.sequence_len_offset > 0:
    #         # This should mean that we are past the prompt forward_step
    #         # and so we need to turn off masking
    #         attn_mask_type = AttnMaskType.no_mask

    #     batch_start = inference_params.batch_size_offset
    #     batch_end = batch_start + key.size(1)
    #     assert batch_end <= inference_key_memory.size(1)
    #     sequence_start = inference_params.sequence_len_offset
    #     sequence_end = sequence_start + key.size(0)
    #     assert sequence_end <= inference_key_memory.size(0)

    #     if self.config.flash_decode:
    #         assert (
    #             rotary_pos_cos is not None and rotary_pos_sin is not None
    #         ), "Flash decoding requires precomputed cos and sin tensors"
    #         if inference_params.sequence_len_offset > 0:  # Decode phase, not prefill
    #             rotary_pos_cos_q = rotary_pos_cos[sequence_end - 1 : sequence_end]
    #             rotary_pos_sin_q = rotary_pos_sin[sequence_end - 1 : sequence_end]
    #             rotary_pos_cos_k = rotary_pos_cos[sequence_end - 1 : sequence_end]
    #             rotary_pos_sin_k = rotary_pos_sin[sequence_end - 1 : sequence_end]
    #         else:
    #             rotary_pos_cos_q = rotary_pos_cos[:sequence_end]
    #             rotary_pos_sin_q = rotary_pos_sin[:sequence_end]
    #             rotary_pos_cos_k = rotary_pos_cos[:sequence_end]
    #             rotary_pos_sin_k = rotary_pos_sin[:sequence_end]

    #         # Flash Decoding assumes that the keys stored in the KV Cache already have RoPE applied.
    #         # Apply RoPE before we store the keys to make it compatible with flash decoding kernel.
    #         key = apply_rotary_pos_emb_with_cos_sin(key, rotary_pos_cos_k, rotary_pos_sin_k)
    #         query = apply_rotary_pos_emb_with_cos_sin(query, rotary_pos_cos_q, rotary_pos_sin_q)
    #     else:
    #         rotary_pos_cos_q = None
    #         rotary_pos_sin_q = None

    #     # Copy key and values.
    #     inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key
    #     inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value
    #     key = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
    #     value = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

    #     # adjust the key rotary positional embedding
    #     if rotary_pos_emb is None:
    #         return query, key, value, rotary_pos_emb, attn_mask_type

    #     q_pos_emb, k_pos_emb = rotary_pos_emb
    #     q_pos_emb = q_pos_emb[sequence_start:sequence_end, :, :, :]
    #     k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
    #     rotary_pos_emb = (q_pos_emb, k_pos_emb)

    #     return query, key, value, rotary_pos_emb, attn_mask_type

    def get_query_key_value_tensors(self, query_input):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        bs = query_input.size(1)
        
        # print("### KnowledgeAttention get_query_key_value_tensors before kn_up_proj query_input: {}\n query_input shape: {},\n".format(query_input, query_input.shape))

        query, _ = self.kn_up_proj(self.kn_layernorm(query_input))

        # print("### KnowledgeAttention get_query_key_value_tensors after kn_up_proj query: {}\n query shape: {},\n".format(query, query.shape))

        new_tensor_shape_query = query.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
    
        query = query.view(*new_tensor_shape_query).contiguous()
        # print("### KnowledgeAttention get_query_key_value_tensors after reshape query: {}\n query shape: {},\n".format(query, query.shape))

        new_tensor_shape_kv = (
            self.config.knowledge_block_fields_num,
            bs,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        
        # print("### KnowledgeAttention self.key_matrix: {}\n self.key_matrix: {},\n".format(self.key_matrix, self.key_matrix.shape))
        if self.tensor_model_parallel_size > 1:
            key = self.key_matrix[:,self.rank * self.num_attention_heads_per_partition : self.rank * self.num_attention_heads_per_partition + self.num_attention_heads_per_partition,:].unsqueeze(1).expand(-1, bs, -1, -1).view(*new_tensor_shape_kv).contiguous()

            # print("### KnowledgeAttention get_query_key_value_tensors key: {}\n key shape: {},\n".format(key, key.shape))

            value = self.value_matrix[:,self.rank * self.num_attention_heads_per_partition : self.rank * self.num_attention_heads_per_partition + self.num_attention_heads_per_partition,:].unsqueeze(1).expand(-1, bs, -1, -1).view(*new_tensor_shape_kv).contiguous()

        else:
            if bs > 1:
                key = self.key_matrix.unsqueeze(1).expand(-1, bs, -1, -1).view(*new_tensor_shape_kv).contiguous()

                # print("### KnowledgeAttention get_query_key_value_tensors key: {}\n key shape: {},\n".format(key, key.shape))

                value = self.value_matrix.unsqueeze(1).expand(-1, bs, -1, -1).view(*new_tensor_shape_kv).contiguous()
            else:
                key = self.key_matrix.unsqueeze(1).expand(-1, bs, -1, -1).view(*new_tensor_shape_kv).contiguous().clone()

                # print("### KnowledgeAttention get_query_key_value_tensors key: {}\n key shape: {},\n".format(key, key.shape))

                value = self.value_matrix.unsqueeze(1).expand(-1, bs, -1, -1).view(*new_tensor_shape_kv).contiguous().clone()


        return query, key, value

    # def flash_decoding(
    #     self,
    #     sequence_len_offset: Tensor,
    #     query_layer: Tensor,
    #     key_layer: Tensor,
    #     value_layer: Tensor,
    #     inference_key_memory: Tensor,
    #     inference_value_memory: Tensor,
    #     rotary_cos: Tensor,
    #     rotary_sin: Tensor,
    # ) -> (Tensor, Tensor):
    #     """
    #     The flash decoding kernel will do the following in a single execution:
    #     1. Compute RoPE embedding with precomputed cos & sin tensors
    #     2. Update the KV Cache
    #     3. Performs the flash attention operation
    #     """
    #     assert flash_attn_with_kvcache is not None, (
    #         "Flash Decoding requires the flash_attn_with_kvcache kernel, "
    #         "available in the flash-attn package."
    #     )
    #     cache_seqlens = sequence_len_offset - 1
    #     q = query_layer.permute(1, 0, 2, 3)
    #     k = key_layer.permute(1, 0, 2, 3)
    #     v = value_layer.permute(1, 0, 2, 3)
    #     k_cache = inference_key_memory.permute(1, 0, 2, 3)
    #     v_cache = inference_value_memory.permute(1, 0, 2, 3)

    #     if rotary_cos is not None:
    #         rotary_cos = rotary_cos.to(query_layer.dtype)
    #     if rotary_sin is not None:
    #         rotary_sin = rotary_sin.to(query_layer.dtype)

    #     out = flash_attn_with_kvcache(
    #         q=q,
    #         k_cache=k_cache,
    #         v_cache=v_cache,
    #         k=k,
    #         v=v,
    #         rotary_cos=rotary_cos,
    #         rotary_sin=rotary_sin,
    #         cache_seqlens=cache_seqlens,
    #         rotary_interleaved=False,
    #     )
    #     return out

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        """
        Perform a forward pass through the attention module.
        """
        # 1. x=kn_layer_norm(kv_conpressed)
        # 2. q=kn_up_proj(x)
        # 3. att=core_att(q, k, v)
        #    TODO: core_att TE DotProductAttention(TransformerEngineBaseModule) implement:
        #       a. topk_indices, topk_scores = topk(torch.bmm(q, k.transpose(,))
        #       b. topk_v = torch.gather(v, dim=, index=topk_indices)
        #       c. att=torch.bmm(torch.softmax(topk_scores * scale_factor), topk_v)
        # 4. x_new=attout(att)

        query, key, value = self.get_query_key_value_tensors(hidden_states)
        # print("### forward q: {}, k: {}, v:{}\n".format(query.shape, key.shape, value.shape))
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=self.attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.kn_att_out_proj(core_attn_out)

        return output, bias