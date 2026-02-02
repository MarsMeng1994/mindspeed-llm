# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain Mamba."""
# from patchs import patch
# patch(
#     # dpsk_fp8=True, 
#     # zb2p=True, 
#       cross_entropy_loss=True, 
#     #   bf16mv_adam=False, 
#       swiglu=True,
#     #   fa3=False
#     )

import os
import torch
from functools import partial
from typing import List, Optional, Tuple, Union
from contextlib import nullcontext
import inspect

import torch.distributed
from mindspeed_llm import megatron_adaptor

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.models.mamba import MambaMoEModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
# from megatron.core.models.mamba.mamba_layer_moe_specs import get_mamba_moe_stack_spec
from megatron.core.models.mamba.jamba_layer_specs import get_jamba_stack_spec
# from megatron.training.tokenizer import build_tokenizer
# import transformers

stimer = StragglerDetector()

# args = get_args()
# local_tokenizer = transformers.AutoTokenizer.from_pretrained(
#             pretrained_model_name_or_path="/sharedata/zyf/models/Qwen2.5-1.5B-Instruct", 
#             trust_remote_code=True,
#         )
# print("### local_tokenizer : {}\n".format(local_tokenizer))
def count_parameters_in_layer(model, layer_name):
    num_params = 0
    for name, param in model.named_parameters():
        if layer_name in name:
            num_params += param.numel()
            # print_rank_0(f" - {name}: {param.numel()}")
    return num_params

# test
def print_total_parameters(model):
    """打印模型的总参数量"""
    print_rank_0("############## test model parameters count #####")
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            # print_rank_0(f"p: {p}")
            total_params += p.numel()
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank_0(f"Total parameters v2 test: {total_params}")
def print_total_parameters_all(model):
    """打印模型的总参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    print_rank_0(f"Total parameters v3 test: {total_params}")
def print_total_parameters_another(model):
    """打印模型的总参数量"""
    params = (np for np in model.named_parameters())
    params = sum(p.numel() for _, p in params)
    print_rank_0(f"Total parameters v4 test: {params}")


def model_provider(pre_process=True, post_process=True) -> MambaMoEModel:
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        MambaModel: The returned model
    """
    args = get_args()

    print_rank_0('building Mamba model ...')
    config = core_transformer_config_from_args(get_args())

    assert args.use_legacy_models == False, "Mamba only supported in Mcore!"
    print_rank_0(f"args.spec: {args.spec}")

    # if args.spec is not None:
    #     mamba_moe_stack_spec = import_module(args.spec)
    # else:
    #     raise("You must provide a valid Mamba layer spec!")
    # print_rank_0(f"@@@@@@@@@@@@@@@@ config before moe setting: {config}")
    # # set moe related params here  -- method2
    # config.num_moe_experts = 8
    # # config.num_moe_experts = 16
    # config.moe_router_topk = 2
    # config.moe_router_load_balancing_type = "aux_loss"
    # config.moe_grouped_gemm = True
    # config.moe_aux_loss_coeff = 1e-2
    # config.moe_token_dispatcher_type = "alltoall"
    # config.moe_per_layer_logging = True
    # print_rank_0(f"@@@@@@@@@@@@@@@@ config after moe setting: {config}")
    mamba_moe_stack_spec = get_jamba_stack_spec(
        config=config,
        use_transformer_engine=(args.transformer_impl == 'transformer_engine')
    )
    print_rank_0(f" == all args: {args}")
    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:
            raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")
    with build_model_context(**build_model_context_args):
        model = MambaMoEModel(
            config=config,
            mamba_moe_stack_spec=mamba_moe_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            hybrid_attention_ratio=args.hybrid_attention_ratio,
            hybrid_mlp_ratio=args.hybrid_mlp_ratio,
            hybrid_override_pattern=args.hybrid_override_pattern,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base
        )
    total_params = 0
    print_rank_0(f" == model arch: {model}")
    print_rank_0(f" == model.decoder.num_layers_per_pipeline_rank: {model.decoder.num_layers_per_pipeline_rank}")
    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        total_params += layer_params
        print_rank_0(f" == params layer {l}: {layer_params}")
    # convert to billion
    total_params_in_billions = total_params / 1000000000
    print_rank_0(f" == Total decoder parameters: {total_params_in_billions:.2f}B") # 0.38b
    print_total_parameters(model)
    print_total_parameters_all(model)
    print_total_parameters_another(model)
    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if ((not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage())) or data_iterator==None:
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    # print("### get batch batch: {}\n".format(batch))

    # print("### get batch tokens: {}\n labels: {}\n".format(local_tokenizer.batch_decode(batch["tokens"]), local_tokenizer.batch_decode(batch["labels"])))
    # print("### get batch tokens: {}\n labels: {}\n".format(batch["tokens"], batch["labels"]))
    # print("### get batch tokens len : {}\n labels len: {}\n".format(len(batch["tokens"][0]), len(batch["labels"][0])))

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])
    # print("### rank: {}, loss_func total_tokens: {}, losses: {}, loss_mask: {}, loss: {}\n".format(torch.distributed.get_rank(), total_tokens, losses, loss_mask, loss))
    if args.context_parallel_size > 1:
        # torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        handle = torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group(), async_op=True)
        handle.wait()  # 确保 all_reduce 操作完成

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    # torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    handle = torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group(), async_op=True)
    handle.wait()

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: MambaMoEModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (MambaModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def forward_step_chat(input_tokens=None, position_ids=None, model: MambaMoEModel = None):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (MambaModel): The GPT Model
    """
    args = get_args()
    # timers = get_timers()

    # Get the batch.
    # timers('batch-generator', log_level=2).start()
    global stimer
    # with stimer(bdata=True):
    #     tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
    #         data_iterator)
    # timers('batch-generator').stop()
    # from megatron.training import print_rank_0
    # print_rank_0("### rank: {}, input_tokens: {}, shape: {}, position_ids: {}\n".format(torch.distributed.get_rank(), input_tokens, input_tokens.shape, position_ids))

    with stimer:
        if not mpu.is_pipeline_first_stage():
            output_tensor = model(input_ids=None, position_ids=position_ids, decoder_input=input_tokens)
        else:
            output_tensor = model(input_tokens, position_ids)
    # print_rank_0("### rank: {}, output_tensor: {}, shape: {}, position_ids: {}\n".format(torch.distributed.get_rank(), output_tensor, output_tensor.shape, position_ids))
    return output_tensor


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)
    # print_rank_0(f" == core_gpt_dataset_config_from_args blend: {blend}")
    # print_rank_0(f" == core_gpt_dataset_config_from_args blend_per_split: {blend_per_split}")
    # print("### core_gpt_dataset_config_from_args blend: {}\n".format(blend))
    # print("### core_gpt_dataset_config_from_args blend_per_split: {}\n".format(blend_per_split))
    
    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
        is_sft_dataset=args.is_sft_dataset,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset
    
    # print("### train_valid_test_datasets_provider config: {}\n".format(config))

    print_rank_0("> building train, validation, and test datasets for GPT ...")
    print_rank_0(f"> train_val_test_num_samples: {train_val_test_num_samples}") # (50292968, 851968, 65536)

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()
    # print("### train_valid_test_datasets_provider train_ds: {}\n".format(train_ds))
    # print("### train_valid_test_datasets_provider valid_ds: {}\n".format(valid_ds))
    # print("### train_valid_test_datasets_provider test_ds: {}\n".format(test_ds))

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step_chat,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
