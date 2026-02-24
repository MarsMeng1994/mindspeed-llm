TOKENIZER_PATH=/sharedata/msm/models/Qwen2.5-1.5B-Instruct-tokenizer/

python ../../../preprocess_sft_data.py \
    --input /sharedata/msm/data/sft/20251118/mmlu-pro/ --workers 1 --output-prefix /sharedata/msm/data/sft-idx-bin/20251118/cot-qwen-72b-gen \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_PATH 