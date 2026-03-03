#!/bin/bash
# MAX_RETRIES=100  # 最大重试次数
# RETRY_INTERVAL=10  # 每次重试的间隔时间（秒）

# # 循环执行 nslookup，直到成功解析地址
# retry_count=0
# while true; do
#     # 执行 nslookup 命令
#     nslookup_output=$(nslookup "$MASTER_ADDR" 2>&1)
#     nslookup_status=$?

#     # 检查 nslookup 是否成功
#     if [ $nslookup_status -eq 0 ]; then
#         echo "Successfully resolved $MASTER_ADDR"
#         break  # 成功解析后退出循环
#     else
#         echo "Failed to resolve $MASTER_ADDR (Attempt $retry_count/$MAX_RETRIES):"
#         echo "$nslookup_output"
#         retry_count=$((retry_count + 1))

#         # 如果达到最大重试次数，退出脚本
#         if [ $retry_count -ge $MAX_RETRIES ]; then
#             echo "Failed to resolve $MASTER_ADDR after $MAX_RETRIES attempts. Exiting."
#             exit 1
#         fi

#         # 等待一段时间后重试
#         echo "Retrying in $RETRY_INTERVAL seconds..."
#         sleep $RETRY_INTERVAL
#     fi
# done
# sleep $RETRY_INTERVAL
# nslookup $MASTER_ADDR
# export MASTER_IP=$(nslookup $MASTER_ADDR | grep "Address:" | tail -n 1 | awk '{print $2}')

# export CUDA_PATH=/usr/local/cuda
# export CUDA_HOME=/usr/local/cuda
# export CUDNN_PATH=/usr
# export CXX=/usr/bin/g++
# export PATH=$CUDA_PATH/bin:$PATH
sysctl -w fs.file-max=102400
ulimit -n 65536
sysctl -w vm.max_map_count=262144
MODEL_SCALE="60b"
case "${MODEL_SCALE}" in
    "356M")
        TENSOR_MODEL_PARALLEL_SIZE=1
        EXPERT_MODEL_PARALLEL_SIZE=8
        PIPELINE_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=16
        HIDDEN_SIZE=1024
        FFN_HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=8
        GLOBAL_BATCH_SIZE=1024
        ;;
    "1.2b") 
        TENSOR_MODEL_PARALLEL_SIZE=1
        EXPERT_MODEL_PARALLEL_SIZE=8
        PIPELINE_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=48
        HIDDEN_SIZE=1024
        FFN_HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=16
        GLOBAL_BATCH_SIZE=2048
        ;;
    "2b") 
        TENSOR_MODEL_PARALLEL_SIZE=1
        EXPERT_MODEL_PARALLEL_SIZE=2
        PIPELINE_MODEL_PARALLEL_SIZE=4
        NUM_LAYERS=4
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=8192
        NUM_ATTENTION_HEADS=32
        GLOBAL_BATCH_SIZE=256
        ;;
    "7b") 
        TENSOR_MODEL_PARALLEL_SIZE=8
        EXPERT_MODEL_PARALLEL_SIZE=8
        PIPELINE_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=48
        HIDDEN_SIZE=2048
        FFN_HIDDEN_SIZE=8192
        NUM_ATTENTION_HEADS=16
        GLOBAL_BATCH_SIZE=512
        ;;
    "20b") 
        TENSOR_MODEL_PARALLEL_SIZE=1
        EXPERT_MODEL_PARALLEL_SIZE=8
        PIPELINE_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=18
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=8192
        NUM_ATTENTION_HEADS=32
        GLOBAL_BATCH_SIZE=256
        ;;
    "60b")
        TENSOR_MODEL_PARALLEL_SIZE=1
        CONTEXT_PARALLEL_SIZE=1
        EXPERT_MODEL_PARALLEL_SIZE=4
        PIPELINE_MODEL_PARALLEL_SIZE=8
        NUM_LAYERS=30
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=8192
        NUM_ATTENTION_HEADS=32
        GLOBAL_BATCH_SIZE=32
        ;;
    *)
        echo "Invalid version specified"
        exit 1
        ;;
esac


# DATA_PATH=`cat ./data.lst`
DATA_PATH=/sharedata/msm/data/Qwen2.5-1.5B-Instruct-tokenizer/txt_360_text_document
TOKENIZER_PATH=/sharedata/msm/models/Qwen2.5-1.5B-Instruct-tokenizer/
# ========== HCCL（昇腾通信库）核心配置 ==========
# 开启HCCL调试日志（对应NCCL_DEBUG=INFO）
export HCCL_DEBUG=INFO
# 设置通信超时时间（对应NCCL_TIMEOUT=600）
export HCCL_TIMEOUT=600
# 启用IB/RDMA网卡（对应NCCL_IB_DISABLE=0）
export HCCL_IB_DISABLE=0
# 设置Socket通信线程数（对应NCCL_SOCKET_NTHREADS=8）
export HCCL_SOCKET_NTHREADS=8
# 若使用RDMA，启用设备级RDMA（对应FI_EFA_USE_DEVICE_RDMA=1）
export HCCL_RDMA_ENABLE=1

# ========== 网络提供者配置（若用RDMA） ==========
# 替换FI_PROVIDER=efa为NPU兼容的RDMA
export FI_PROVIDER=rdma
# 设置RDMA相关参数
export FI_RDMA_CONNECT_TIMEOUT=6000

# export CUDA_DEVICE_MAX_CONNECTIONS=1
# # export NCCL_IB_TIMEOUT=19
# # export NCCL_IB_QPS_PER_CONNECTION=8
# export NCCL_SOCKET_NTHREADS=8
# # export NCCL_IB_SL=1
# export NCCL_IB_DISABLE=0
# export NCCL_TIMEOUT=600 

# # export DS_BUILD_FUSED_ADAM=1
# # export NCCL_PROTO=simple
# export NCCL_DEBUG=INFO
# # export HCCL_OVER_OFI=1
# # 使用efa
# export FI_PROVIDER=efa
# # export NCCL_IGNORE_DISABLED_P2P=1
# ## 如果拉起的集群为A100/H100（P4d/P5机型）节点，可以设置如下的NCCL参数启用RDMA，以获得多机多卡训练时节点GPU更大的网络吞吐量
# export FI_EFA_USE_DEVICE_RDMA=1

# 从环境变量中读取值
GPUS_PER_NODE=${GPUS_PER_NODE:-16}
# MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_IP=${MASTER_IP:-"192.168.62.7"}
MASTER_PORT=${MASTER_PORT:-6000}
NUM_NODES=${NUM_NODES:-2}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# 打印环境变量的值
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "MASTER_IP: $MASTER_IP"
echo "MASTER_PORT: $MASTER_PORT"
echo "NUM_NODES: $NUM_NODES"
echo "NODE_RANK: $NODE_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"

export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=6000
export NODE_RANK=$NODE_RANK
export NNODES=$NUM_NODES
export WORLD_SIZE=$WORLD_SIZE


PROJECT_DIR="ascend"
EXP_NAME="mamba_moe_60b_continue_pretrain_ckpt75000"
LOG_DIR="./logs/${EXP_NAME}_${NODE_RANK}"
mkdir -p ${LOG_DIR}

SAVE_CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
# LOAD_CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
LOAD_CHECKPOINT_DIR="./jamba_60b_aws_oh_pp8_ep4_efa_512k_sft_v1_16node_ckpt75000/"

DATACACHE_DIR="./data-cache/${EXP_NAME}"
TENSORBOARD_DIR="./tensorboard/${EXP_NAME}"
WANDB_DIR="./wandb/${EXP_NAME}"

mkdir -p ${CHECKPOINT_SAVE_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${WANDB_DIR}

SEQ_LEN=4096
TRAIN_SAMPLES=19200000
LR_WARMUP_SAMPLES=128000
LR_DECAY_SAMPLES=19072000  # TRAIN_SAMPLES - LR_WARMUP_SAMPLES

GPT_MODEL_ARGS=(
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --swiglu
    --num-layers $NUM_LAYERS
    # --old-num-layers 30
    # --old-encoder-num-layers 30
    # --jamba-use-upcycling
    # --old-decoder-last-pipeline-num-layers 2
    # --old-pp-num 8
    # --old-ep-num 4
    # --local2te
    --num-attention-heads $NUM_ATTENTION_HEADS
    # --group-query-attention
    # --num-query-groups 8
    --normalization RMSNorm
    --position-embedding-type none
    # --no-position-embedding
    --seq-length $SEQ_LEN
    --max-position-embeddings $SEQ_LEN
    --original-max-position-embeddings 131072
    # --rotary-base 10000
    --attention-backend fused # Can use (flash/fused/unfused/local)
    # --use-flash-attn 
    # --vocab-size 151936
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_PATH
    --attention-dropout 0.0
    --hidden-dropout 0.0


    # --use-dist-ckpt
    # --auto-detect-ckpt-format
    --ckpt-format torch
    --use-distributed-optimizer
    # --no-gradient-accumulation-fusion


    # --use-precision-aware-optimizer
    # --exp-avg-dtype fp16
    # --exp-avg-sq-dtype fp16
    # --use-bf16mv-optimizer
    # --main-grads-dtype bf16

    # --reset-position-ids
    # --reset-attention-mask

    # --add-qkv-bias
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    # --cross-entropy-loss-fusion

    # mla
    --multi-latent-attention
    --q-lora-rank 512
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-base 5000000

    # knowledge block
    --use-knowledge-block
    --knowledge-block-freq 4
    # # --knowledge-block-start-layer-idx 5
    --knowledge-block-fields-num 64
    --knowledge-block-heads-num 32
    --knowledge-block-heads-dim 128
    
    --seed 3147
    # --no-rope-fusion

    --use-mcore-models
    --no-create-attention-mask-in-dataloader

    --decoder-last-pipeline-num-layers 2
    
    --old-pp-num 8
    --old-ep-num 1
)

MOE_ARGS=(
    --num-experts 16
    --moe-router-topk 2
    # --moe-per-layer-logging

    # --moe-router-gating-score-type sigmoid
    --moe-router-score-function sigmoid
    --moe-ffn-hidden-size $FFN_HIDDEN_SIZE
    --moe-shared-expert-intermediate-size $FFN_HIDDEN_SIZE
    --moe-shared-expert-overlap
    --moe-router-load-balancing-type none
    # --moe-router-load-balancing-type aux_loss
    # --moe-aux-loss-coeff 1e-8
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall

    --input-conv-freq 4
    # --input-conv-start-layer-idx 5
    --conv-attention-kernel-size 4
    --moe-router-dtype fp32

    # --moe-layer-recompute

    # --moe-shared-expert-intermediate-size 8192
    # --moe-shared-expert-overlap

    # --moe-router-gating-score-type softmax
    # --moe-output-connection-type stack
    # --moe-expert-capacity-factor 1.0
    # --moe-z-loss-coeff 1e-3
    # --moe-token-drop-policy position
    # --moe-pad-expert-input-to-capacity
    
)

MAMBA_ARGS=(
    --hybrid-attention-ratio 0.25
    --hybrid-mlp-ratio 0.5
    --mamba-expand 2
    --mamba-chunk-size 128
)
# GPT_MODEL_ARGS=(
#     --num-layers 32 
#     --hidden-size 4096 
#     --num-attention-heads 96 
#     --seq-length 2048 
#     --max-position-embeddings 2048 
# )
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

TRAINING_ARGS=(
    --calculate-per-token-loss
    --micro-batch-size 1
    --global-batch-size $GLOBAL_BATCH_SIZE 
    # --global-batch-size 32
    # --global-batch-size 256
    # --rampup-batch-size 16 16 5859375 
    # --train-iters 100 
    --train-samples $TRAIN_SAMPLES
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --lr-decay-samples $LR_DECAY_SAMPLES
    --weight-decay 0.1 
    # --optimizer muon
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    # --fp8-format e4m3
    # --fp8-format hybrid
    # --lr 1.72e-4
    # --lr-decay-style linear 
    # --min-lr 0.72e-4
    --lr 5e-6
    --lr-decay-style constant
    # --min-lr 1e-5
    --distributed-backend nccl
    # --recompute-activations

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    # --distribute-saved-activations
   
    # --lr-warmup-fraction .001 
    # --lr-decay-iters 430000 

    # --defer-embedding-wgrad-compute
    # --wgrad-deferral-limit 8
    # --transformer-impl local
    # --no-persist-layer-norm

    # --account-for-embedding-in-pipeline-split
    # --account-for-loss-in-pipeline-split
    # --skip-train
    --no-load-optim
    --no-load-rng
    # --override-opt_param-scheduler
)

MODEL_PARALLEL_ARGS=(
	# --tensor-model-parallel-size 1
	# --pipeline-model-parallel-size 1
    # # --num-layers-per-virtual-pipeline-stage 2
    # --expert-model-parallel-size 8
    
    # --sequence-parallel
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
    --expert-model-parallel-size $EXPERT_MODEL_PARALLEL_SIZE
    # --sequence-parallel
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
    --context-parallel-size $CONTEXT_PARALLEL_SIZE
    # --overlap-param-gather
    # --overlap-grad-reduce
    # --use-ring-exchange-p2p
    # --pipeline-model-parallel-size 1 

)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --data-cache-path $DATACACHE_DIR
    # --vocab-file $VOCAB_FILE 
    # --merge-file $MERGE_FILE 
    --split 99,1,0
    # --split 0,1,0
    --dataloader-type cyclic
    # --is-sft-dataset
    # --s3
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 500
    --eval-interval 2000
    --save $SAVE_CHECKPOINT_DIR 
    --load $LOAD_CHECKPOINT_DIR 
    --eval-iters 32
    # --eval-and-save-only
    ### tensorboard & wandb
    --tensorboard-dir $TENSORBOARD_DIR 
    --log-timers-to-tensorboard
    # --timing-log-level 2
    --log-throughput
    --log-params-norm
    --use-wandb
    --wandb-project $PROJECT_DIR
    --wandb-exp-name $EXP_NAME
)

LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"  

torchrun ${DISTRIBUTED_ARGS[@]} ../../../pretrain_mamba_moe_exp.py \
    ${GPT_MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MAMBA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} 2>&1 | tee ${LOG_FILE}