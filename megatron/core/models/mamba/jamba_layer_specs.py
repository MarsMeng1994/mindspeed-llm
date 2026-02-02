# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import Optional
import warnings
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.ssm.mamba_block import (
    # MambaStack, 
    # MambaMoEStack, 
    # MambaStackSubmodules, 
    # MambaMoEStackSubmodules,
    # MambaMoEStackV2, 
    # MambaMoEStackSubmodulesV2,
    MambaMoEStackV3, 
    MambaMoEStackSubmodulesV3,
    
)
# from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.ssm.mamba_moe_layer import MambaMOELayer, MambaMOELayerSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
# from megatron.core.transformer.moe.moe_layer_v2 import MoELayerV2, MoESubmodulesV2
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention


# from megatron.core.transformer.custom_layers.transformer_engine import (
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
    TERowParallelGroupedLinear,
    TEColumnParallelGroupedLinear,

)
try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TELinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False
# try:
#     import apex  # pylint: disable=unused-import

#     from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

#     HAVE_APEX = True
#     LNImpl = FusedLayerNorm
# except ImportError:
from megatron.core.transformer.torch_norm import WrappedTorchNorm

warnings.warn('Apex is not installed. Falling back to Torch Norm')
LNImpl = WrappedTorchNorm

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer, 
    TransformerLayerSubmodules,
    )
from megatron.core.transformer.transformer_layer_v2 import (
    TransformerLayerV2, 
    TransformerLayerSubmodulesV2,
    )
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
# from megatron.core.transformer.multi_latent_attention import (
#     MLASelfAttention,
#     MLASelfAttentionSubmodules,
# )
from megatron.core.transformer.multi_latent_attention_v2 import (
    MLASelfAttentionV2,
    MLASelfAttentionSubmodulesV2,
)
from megatron.core.transformer.multi_latent_attention_local import (
    MLASelfAttentionV3,
    MLASelfAttentionSubmodulesV3,
)
from megatron.core.transformer.knowledge_attention import (
    KnowledgeAttention,
    KnoledgeAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.models.gpt.gpt_layer_specs import (
    # get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    # get_gpt_layer_with_transformer_engine_spec_mamba,
    # get_gpt_layer_local_spec_mamba,
)

def get_jamba_stack_spec(config: TransformerConfig, use_transformer_engine: bool = True):
    """Get the mamba moe stack spec based on config and transformer engine flag."""
    
    # if use_transformer_engine:
    #     base_spec_func = get_gpt_layer_with_transformer_engine_spec_mamba
    # else:
    #     base_spec_func = get_gpt_layer_local_spec_mamba

    # Define the mamba layer spec
    if use_transformer_engine:
        mlp = get_mlp_module_spec(
            use_te=True, num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm, fp8=config.fp8
        )
        mamba_layer_spec = ModuleSpec(
            module=MambaMOELayer,
            submodules=MambaMOELayerSubmodules(
                norm=TENorm,
                # norm=LNImpl,
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=TEColumnParallelLinear if use_transformer_engine else ColumnParallelLinear,
                        out_proj=TERowParallelLinear if use_transformer_engine else RowParallelLinear,
                        # in_proj=ColumnParallelLinear,
                        # out_proj=RowParallelLinear,
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm,
                # pre_mlp_layernorm=LNImpl,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )

        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                            config.num_moe_experts, 
                            config.moe_grouped_gemm,
                            config.qk_layernorm, 
                            config.multi_latent_attention, 
                            config.fp8, 
                            config.moe_use_legacy_grouped_gemm,
                            )
        
    else:
        mlp = get_mlp_module_spec(
            use_te=False, num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm, fp8=config.fp8
        )
        mamba_layer_spec = ModuleSpec(
            module=MambaMOELayer,
            submodules=MambaMOELayerSubmodules(
                norm=LNImpl,
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj= ColumnParallelLinear,
                        out_proj= RowParallelLinear,
                        # in_proj=TEColumnParallelLinear if use_transformer_engine else ColumnParallelLinear,
                        # out_proj=TERowParallelLinear if use_transformer_engine else RowParallelLinear,
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
                pre_mlp_layernorm=LNImpl,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )

        transformer_layer_spec = get_gpt_layer_with_local_spec(
                            config.num_moe_experts, 
                            config.moe_grouped_gemm,
                            config.qk_layernorm, 
                            config.multi_latent_attention, 
                            config.fp8, 
                            config.moe_use_legacy_grouped_gemm,
                            )

        
    return ModuleSpec(
            module=MambaMoEStackV3,
            submodules=MambaMoEStackSubmodulesV3(
                mamba_moe_layer=mamba_layer_spec,
                transformer_layer=transformer_layer_spec,
            ),
        )


def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,
) -> ModuleSpec:
    """Helper function to get module spec for MLP"""
    print("### get_mlp_module_spec num_experts: {}\n".format(num_experts))
    if num_experts is not None:
        # if use_te and moe_grouped_gemm:
        #     linear_fc1 = TEColumnParallelGroupedLinear
        #     linear_fc2 = TERowParallelGroupedLinear
        # elif use_te and fp8:
        #     linear_fc1 = TEColumnParallelLinear
        #     linear_fc2 = TERowParallelLinear
        # else:
        #     linear_fc1 = ColumnParallelLinear
        #     linear_fc2 = RowParallelLinear

        # use_te_grouped_gemm = use_te and TEColumnParallelGroupedLinear is not None

        # return  ModuleSpec(
        #     module=MoELayer, 
        #     submodules=MoESubmodules(
        #         experts=ModuleSpec(
        #             module=TEGroupedMLP,
        #             submodules=(
        #                 MLPSubmodules(
        #                     linear_fc1=linear_fc1, 
        #                     linear_fc2=linear_fc2,
        #                 ),
        #             ),
        #         ),
        #             # MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
        #             # if not moe_grouped_gemm or use_te_grouped_gemm
        #             # else None
        #             # ),
        #         shared_experts=ModuleSpec(
        #             module=SharedExpertMLP,
        #             params={"gate": False},
        #             submodules=MLPSubmodules(
        #                 linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
        #                 linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        #                 ),
        #             ),
        #         ),
        #     )
        return get_moe_module_spec(
            use_te=use_te,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            # moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )


    else:

        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                # linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    mlp = get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    if multi_latent_attention:
        # return ModuleSpec(
        #     module=TransformerLayer,
        #     submodules=TransformerLayerSubmodules(
        #         input_layernorm=TENorm,
        #         self_attention=ModuleSpec(
        #             module=MLASelfAttention,
        #             params={"attn_mask_type": AttnMaskType.causal},
        #             submodules=MLASelfAttentionSubmodules(
        #                 linear_q_proj=TEColumnParallelLinear,
        #                 linear_q_down_proj=TELinear,
        #                 linear_q_up_proj=(
        #                     TELayerNormColumnParallelLinear
        #                     if qk_layernorm
        #                     else TEColumnParallelLinear
        #                 ),
        #                 linear_kv_down_proj=TELinear,
        #                 linear_kv_up_proj=(
        #                     TELayerNormColumnParallelLinear
        #                     if qk_layernorm
        #                     else TEColumnParallelLinear
        #                 ),
        #                 core_attention=TEDotProductAttention,
        #                 linear_proj=TERowParallelLinear,
        #                 q_layernorm=IdentityOp,
        #                 kv_layernorm=IdentityOp,
        #             ),
        #         ),
        #         self_attn_bda=get_bias_dropout_add,
        #         pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
        #         mlp=mlp,
        #         mlp_bda=get_bias_dropout_add,
        #     ),
        # )
        return ModuleSpec(
                module=TransformerLayerV2,
                submodules=TransformerLayerSubmodulesV2(
                    input_layernorm=TENorm,
                    # input_layernorm=LNImpl,
                    self_attention=ModuleSpec(
                        module=MLASelfAttentionV2,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=MLASelfAttentionSubmodulesV2(
                            linear_q_proj=TEColumnParallelLinear,
                            linear_q_down_proj=TELinear,
                            linear_q_up_proj=TEColumnParallelLinear,
                            # (
                            #     TELayerNormColumnParallelLinear
                            #     if qk_layernorm
                            #     else TEColumnParallelLinear
                            # ),
                            linear_kv_down_proj=TELinear,
                            linear_kv_up_proj=TEColumnParallelLinear,
                            # (
                            #     TELayerNormColumnParallelLinear
                            #     if qk_layernorm
                            #     else TEColumnParallelLinear
                            # ),
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                            q_layernorm=TENorm if qk_layernorm else IdentityOp,
                            kv_layernorm=TENorm if qk_layernorm else IdentityOp,
                    ),
                        # (
                        #     linear_q_proj=TEColumnParallelLinear,
                        #     linear_q_down_proj=TEColumnParallelLinear,
                        #     linear_q_up_proj=TEColumnParallelLinear,
                        #     linear_kv_down_proj=TEColumnParallelLinear,
                        #     linear_kv_up_proj=TEColumnParallelLinear,
                        #     core_attention=TEDotProductAttention,
                        #     linear_proj=TERowParallelLinear,
                        #     q_layernorm=TENorm if qk_layernorm else IdentityOp,
                        #     kv_layernorm=TENorm if qk_layernorm else IdentityOp,
                        # ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=TENorm,
                    # pre_mlp_layernorm=LNImpl,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    knowledge_attention=ModuleSpec(
                        module=KnowledgeAttention,
                        submodules=KnoledgeAttentionSubmodules(
                            kn_layernorm=TENorm,
                            # kn_layernorm=LNImpl,
                            kn_up_proj=TEColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            kn_att_out_proj=TERowParallelLinear,
                        ),
                    ),
                    knowledge_attn_bda=get_bias_dropout_add,
                    # knowledge_attention=IdentityOp,
                    # knowledge_attn_bda=IdentityOp,
                ),
            )
    else:

        # TENorm significantly harms convergence when used
        # for QKLayerNorm if TE Version < 1.9;
        # we instead use the Apex implementation.
        qk_norm = TENorm

        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                        k_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    
def get_gpt_layer_with_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    mlp = get_mlp_module_spec(
        use_te=False,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    if multi_latent_attention:
        return ModuleSpec(
                module=TransformerLayerV2,
                submodules=TransformerLayerSubmodulesV2(
                    input_layernorm=LNImpl,
                    self_attention=ModuleSpec(
                        module=MLASelfAttentionV3,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=MLASelfAttentionSubmodulesV3(
                            linear_q_proj=ColumnParallelLinear,
                            linear_q_down_proj=ColumnParallelLinear,
                            linear_q_up_proj=ColumnParallelLinear,
                            linear_kv_down_proj=ColumnParallelLinear,
                            linear_kv_up_proj=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                            kv_layernorm=LNImpl if qk_layernorm else IdentityOp,
                        ),
                        # (
                        #     linear_q_proj=TEColumnParallelLinear,
                        #     linear_q_down_proj=TEColumnParallelLinear,
                        #     linear_q_up_proj=TEColumnParallelLinear,
                        #     linear_kv_down_proj=TEColumnParallelLinear,
                        #     linear_kv_up_proj=TEColumnParallelLinear,
                        #     core_attention=TEDotProductAttention,
                        #     linear_proj=TERowParallelLinear,
                        #     q_layernorm=TENorm if qk_layernorm else IdentityOp,
                        #     kv_layernorm=TENorm if qk_layernorm else IdentityOp,
                        # ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=LNImpl,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    knowledge_attention=ModuleSpec(
                        module=KnowledgeAttention,
                        submodules=KnoledgeAttentionSubmodules(
                            kn_layernorm=LNImpl,
                            kn_up_proj=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            kn_att_out_proj=RowParallelLinear,
                        ),
                    ),
                    knowledge_attn_bda=get_bias_dropout_add,
                    # knowledge_attention=IdentityOp,
                    # knowledge_attn_bda=IdentityOp,
                ),
            )
        # return ModuleSpec(
        #     module=TransformerLayer,
        #     submodules=TransformerLayerSubmodules(
        #         input_layernorm=LNImpl,
        #         self_attention=ModuleSpec(
        #             module=MLASelfAttentionV3,
        #             params={"attn_mask_type": AttnMaskType.causal},
        #             submodules=MLASelfAttentionSubmodulesV3(
        #                 linear_q_proj=ColumnParallelLinear,
        #                 linear_q_down_proj=ColumnParallelLinear,
        #                 linear_q_up_proj=ColumnParallelLinear,
        #                 linear_kv_down_proj=ColumnParallelLinear,
        #                 linear_kv_up_proj=ColumnParallelLinear,
        #                 core_attention=DotProductAttention,
        #                 linear_proj=RowParallelLinear,
        #                 q_layernorm=LNImpl if qk_layernorm else IdentityOp,
        #                 kv_layernorm=LNImpl if qk_layernorm else IdentityOp,
        #             ),
        #         ),
        #         self_attn_bda=get_bias_dropout_add,
        #         pre_mlp_layernorm=LNImpl,
        #         mlp=mlp,
        #         mlp_bda=get_bias_dropout_add,
        #     ),
        # )
    else:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=LNImpl,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=ColumnParallelLinear,
                        core_attention=DotProductAttention,
                        linear_proj=RowParallelLinear,
                        q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                        k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=LNImpl,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map={
                    'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                    'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                },
            ),
        )
    

def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    mlp = MLPSubmodules(
        linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
        linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
    )

    # experts spec
    if moe_grouped_gemm:
        ## use GroupedMLP
        if use_te and TEColumnParallelGroupedLinear is not None and not moe_use_legacy_grouped_gemm:
            ## use TEGroupedLinear
            expert_module = TEGroupedMLP
            expert_submodule = MLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
            )
        else:
            ## use legacy GroupedMLP
            expert_module = GroupedMLP
            expert_submodule = None
            warnings.warn(
                'The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. '
                'Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP.'
            )
    else:
        ## use SequentialMLP
        expert_module = SequentialMLP
        if use_te and not is_te_min_version("1.7.0.dev0"):
            warnings.warn(
                "Only transformer-engine>=1.7.0 supports MoE experts, "
                f"but your version is {get_te_version()}. Use local linear implementation instead."
            )
            expert_submodule = MLPSubmodules(
                linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
            )
        else:
            expert_submodule = mlp

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": False}, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
    )
    # moe_module_spec = ModuleSpec(
    #     module=MoELayerV2, submodules=MoESubmodulesV2(experts=experts, shared_experts=shared_experts)
    # )
    return moe_module_spec