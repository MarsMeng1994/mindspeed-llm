TOKENIZER_PATH=/sharedata/msm/models/Qwen2.5-1.5B-Instruct-tokenizer

python ../../../preprocess_data.py \
    --input /sharedata/data/raw_corpus/TxT360/data/arxiv/1-1/1_171.jsonl \
    --output-prefix /sharedata/msm/data/Qwen2.5-1.5B-Instruct-tokenizer/txt_360 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --workers 1