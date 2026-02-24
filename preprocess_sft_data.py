# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import glob
import torch
import numpy as np
import multiprocessing
try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars
    nltk_available = True
except ImportError:
    PunktLanguageVars = object  # Fallback to the built-in object class
    nltk_available = False

from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset


class CustomLanguageVars(PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,
                    lang_vars=CustomLanguageVars())
            else:
                Encoder.splitter = splitter
        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        full_ids = []
        label_ids = []
        sentence_lens = []
        for key in self.args.json_keys:
            
            if (
                "question" not in data 
                or "response" not in data
                # or "explanation" not in data
            #     # or "director_names" not in data
            #     # or "scriptwriter_names" not in data
            #     # or "actor_names" not in data
            #     # or "text_info" not in data
            #     # or "类型" not in data
            #     # or "制片国家/地区" not in data
            #     # or "语言" not in data
            #     # or "上映日期" not in data
            #     # or "片长" not in data
            #     # or "又名"  not in data
            #     # or "comment" not in data
                ):
            #     # or "input" not in data
                print(data)
                return {}, {}, 0
            # sentence = "<|im_start|>" + "\n".join(["{}: {}".format(k, data[k]) for k in ["type_name", 
            #                                                              "movie_name",
            #                                                              "director_names",
            #                                                              "scriptwriter_names",
            #                                                              "actor_names",
            #                                                              "text_info",
            #                                                              "类型",
            #                                                              "制片国家/地区",
            #                                                              "语言",
            #                                                              "上映日期",
            #                                                              "片长",
            #                                                              "又名" ,
            #                                                              "comment"]]) + "<|im_end|>"
            data["system"] = "You are a helpful assistant. Your name is AIGCode, created by 蔻町智能." 
            sentence = "<|im_start|>system\n{}<|im_end|>\n".format(data["system"])
            sentence_ids = Encoder.tokenizer.tokenize(sentence, truncation=False)
            full_ids.extend(sentence_ids)
            label_ids.extend([-100] * len(sentence_ids))
            sentence_lens.append(len(sentence_ids))
            
            role = "user"
            sentence = "<|im_start|>{}\n{}<|im_end|>\n".format(role, data["question"])
            sentence_ids = Encoder.tokenizer.tokenize(sentence, truncation=False)
            full_ids.extend(sentence_ids)
            label_ids.extend([-100] * len(sentence_ids))
            sentence_lens[-1] = sentence_lens[-1] + len(sentence_ids)

            role = "assistant"
            sentence = "<|im_start|>{}\n".format(role)
            sentence_ids = Encoder.tokenizer.tokenize(sentence, truncation=False)
            full_ids.extend(sentence_ids)
            label_ids.extend([-100] * len(sentence_ids))
            sentence_lens[-1] = sentence_lens[-1] + len(sentence_ids)

            sentence = "{}<|im_end|>\n".format(data["response"])
            sentence_ids = Encoder.tokenizer.tokenize(sentence, truncation=False)
            full_ids.extend(sentence_ids)
            label_ids.extend(sentence_ids)
            sentence_lens[-1] = sentence_lens[-1] + len(sentence_ids)
                        
            sentence = "<|endoftext|>"
            sentence_ids = Encoder.tokenizer.tokenize(sentence, truncation=False)
            full_ids.extend(sentence_ids)
            label_ids.extend([-100] * len(sentence_ids))
            sentence_lens[-1] = sentence_lens[-1] + len(sentence_ids)
            
            ids[key] = full_ids + label_ids
            sentence_lens[-1] = sentence_lens[-1] + sentence_lens[-1]
            lens[key] = sentence_lens
            # sentence = sentence + "<|im_start|>user\n{}<|im_end|>\n".format(data["question"])
            #                                                                                             # data["A"],
            #                                                                                             # data["B"],
            #                                                                                             # data["C"],
            #                                                                                             # data["D"])
            # #                                                                 # + "\n" + data["input"])
            # sentence = sentence + "<|im_start|>assistant\n{}<|im_end|>\n".format(data["response"])
            # sentence = sentence + "<|endoftext|>"
            # # if isinstance(session, list):
            # #     if "source" in data and "identity" in data["source"]:
            # #         return {}, {}, 0
            # #     if "system" not in data:
            # #         data["system"] = "You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
            # #     #     print("data: {}\n".format(data))
            # #     #     return {}, {}, 0
            # #     if isinstance(session[0], dict) and "from" in session[0] and "value" in session[0]:
            # #         sentence = "<|im_start|>system\n{}<|im_end|>\n".format(data["system"])
            # #         # sentence = ""
            # #         for turn in session:
            # #             sentence = sentence + "<|im_start|>{}\n{}<|im_end|>\n".format(turn["from"], turn["value"])
            # #             if turn["from"] == "assistant":
            # #                 if "qwen3_235b" not in turn["info"]["source"]:
            # #                     return {}, {}, 0
            # #             # sentence = sentence + "<|im_start|>{}\n{}<|im_end|>\n".format(turn["response"])
            # #         sentence = sentence + "<|endoftext|>"
            # #         sentences = [sentence]
            # #     else:
            # #         sentences = session
            # # else:
            # #     sentences = [session]
            # # print(data)
            # # print(sentence)
            # sentences = [sentence]
            # doc_ids = []
            # sentence_lens = []
            # for sentence in sentences:
            #     sentence_ids = Encoder.tokenizer.tokenize(sentence, truncation=False)
            #     if len(sentence_ids) > 0:
            #         doc_ids.extend(sentence_ids)
            #         sentence_lens.append(len(sentence_ids))
            # if len(doc_ids) > 0 and self.args.append_eod:
            #     doc_ids.append(Encoder.tokenizer.eod)
            #     sentence_lens[-1] += 1
            # ids[key] = doc_ids
            # lens[key] = sentence_lens
        return ids, lens, len(json_line)


class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {count} documents",
                  f"({count / elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()

    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')

        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        for key in builders:
            builders[key].finalize(output_idx_files[key])


def get_args():
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input directory containing JSON or JSONL files')
    group.add_argument('--json-keys', nargs='+', default=['question'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group = parser.add_argument_group(title='tokenization process')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                       help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    args.model_max_length = 256000

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def main():
    args = get_args()

    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    # Get all .json and .jsonl files in the input directory
    # input_files = glob.glob(os.path.join(args.input, "*.json")) + glob.glob(os.path.join(args.input, "*.jsonl"))
    # Get all .json and .jsonl files in the input directory and its subdirectories
    #  glob.glob(os.path.join(args.input, "**/*.json"), recursive=True) +
    input_files = glob.glob(os.path.join(args.input, "**/*.json"), recursive=True) + glob.glob(os.path.join(args.input, "**/*.jsonl"), recursive=True)
    print("input files : {}\n".format(input_files))
    in_ss_out_names = []
    for idx, input_file in enumerate(input_files):
        file_name, extension = os.path.splitext(input_file)
        sentence_split_file = file_name + "_ss" + extension
        output_prefix = args.output_prefix + "_" + str(idx)
        file_names = {
            'partition': input_file,
            'sentence_split': sentence_split_file,
            'output_prefix': output_prefix}
        in_ss_out_names.append(file_names)

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers // args.partitions)

    # Split sentences in partition files
    if args.split_sentences:
        processes = []
        for name in in_ss_out_names:
            p = multiprocessing.Process(target=partition.split_sentences,
                                        args=((name['partition'], name['sentence_split']),))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # Encode partition files in parallel
    processes = []
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    for name in in_ss_out_names:
        p = multiprocessing.Process(target=partition.process_json_file,
                                    args=((name[input_key], name['output_prefix']),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge bin/idx partitions
    level = "document"
    if args.split_sentences:
        level = "sentence"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer = build_tokenizer(args, model_max_length=256000, max_seq_length=4096000)

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in in_ss_out_names:
            partition_output_prefix = name['output_prefix']
            full_partition_output_prefix = "{}_{}_{}".format(partition_output_prefix,
                                                             key, level)
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    main()
