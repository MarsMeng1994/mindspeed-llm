from megatron.core.datasets import indexed_dataset
from transformers import AutoTokenizer
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer', type=str, default='/workspace/zimoliu/models/Qwen2.5-1.5B-Instruct-tokenizer') 
parser.add_argument('--data-prefix', type=str, default='/sharedata/zimoliu/data/sft_data_sample/v1_2025_1105_megatron_format/unit_simple_15k_question_document')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
mmap_file = indexed_dataset._MMapBinReader(args.data_prefix + '.bin')
idx_file = indexed_dataset._IndexReader(args.data_prefix + '.idx', False)

counts = idx_file.sequence_count
print('*'*50 + f' 数据总数量: {counts}')
print("### idx_file document_indices: {}, sequence_lengths: {}\n".format(idx_file.document_indices, idx_file.sequence_lengths))
for sample_idx in range(counts):
    total_ids = mmap_file.read(idx_file.dtype, np.int32(idx_file.sequence_lengths[sample_idx]), idx_file.sequence_pointers[sample_idx])
    token_ids = mmap_file.read(idx_file.dtype, np.int32(idx_file.sequence_lengths[sample_idx]/2), idx_file.sequence_pointers[sample_idx])
    print('*'*50 + f' sample_idx: {sample_idx}' + ' / ' + f'{counts}')
    # print('*'*50 + f' token_ids:')
    print(total_ids)
    # print("len: {}\n".format(len(token_ids)))
    # half_len = int(len(token_ids)/2)
    # origin_ids = token_ids[:half_len]
    # label_ids = token_ids[half_len:]
    # # 复制数组
    # label_ids_copy = label_ids.copy()

    # # 修改副本
    # label_ids_copy[label_ids_copy == -100] = 0
    # print("first 100 token: {}\n".format(token_ids[:100]))
    # print("last 10 token: {}\n".format(token_ids[-10:]))
    # print('*'*50 + f' decoding contents:')

    # print("origin ids decoding: {}\n".format(tokenizer.decode(origin_ids, skip_special_tokens=False)))
    # print("label ids decoding: {}\n".format(tokenizer.decode(label_ids_copy, skip_special_tokens=False)))
    # print('*'*50 + f' decoding contents:')
    text=tokenizer.decode(token_ids, skip_special_tokens=False)
    print(text)
    if "<|endoftext|>" not in text:
        exit(1)
    # sample_idx = sample_idx + 1
    # inp = input('input “q” will be quited, other will continue: ')

'''
python check_data.py \
    --tokenizer /sharedata/msm/models/Qwen2.5-1.5B-Instruct-tokenizer/ \
    --data-prefix /sharedata/msm/data/sft-idx-bin/20251118/cot-qwen-72b-gen_1_question_document
'''