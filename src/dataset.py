'''
@Author: dzy
@Date: 2021-01-01 11:00:51
@LastEditTime: 2020-09-10 17:36:24
@LastEditors: dzy
@Description: Define the format of data used in the model.
@FilePath: /JDProductSummaryGeneration/model/model.py
'''

import sys
import os
import pathlib
from collections import Counter
from typing import Callable
import torch
from torch.utils.data import Dataset
from vocab import Vocab
import config
from utils import simple_tokenizer, count_words, sort_batch_by_len, source2ids, abstract2ids

abs_path = pathlib.Path(__file__).parent.absolute()

class PairDataset(object):
    def __init__(self, 
                filename: str,                          # 数据集文件名
                tokenize: Callable = simple_tokenizer,  # 分词器
                max_src_len: int = None,                # 最长特征文本长度
                max_tgt_len: int = None,                # 最长标签文本长度
                truncate_src: bool = False,             # 是否截断特征文本
                truncate_tgt: bool = False):            # 是否截断标签文本
        
        self.filename = filename
        self.pairs = []

        with open(filename, 'rt', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                pair = line.strip().split('<sep>')
                if len(pair) != 2:
                    continue

                # 对特征文本进行分词
                src = tokenize(pair[0])

                # 比最长特征文本长度长的截断
                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[0:max_src_len]
                    else:
                        continue

                # 对标签文本进行分词
                tgt = tokenize(pair[1])

                # 比最长标签文本长度长的截断
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = src[tgt:max_tgt_len]
                    else:
                        continue
                
                # 添加到特征文本和标签文本对中
                self.pairs.append((src, tgt))      

    def build_vocab(self, embed_file: str = None) -> Vocab:
        word_counts = Counter()

        # 用Counter()统计每个词出现的次数
        count_words(word_counts, [src + tgt for src, tgt in self.pairs])       

        vocab = Vocab()

        # 只取最大词典数量个词
        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.add_words([word])
        
        # 从预训练中加载词向量
        if embed_file != None:
            vocab.load_embeddings(embed_file)
        
        return vocab

class SampleDataset(Dataset):
    """The class represents a sample set for training.

    """
    def __init__(self, data_pair, vocab):
        # 特征文本
        self.src_sents = [x[0] for x in data_pair]

        # 标签文本
        self.trg_sents = [x[1] for x in data_pair]

        self.vocab = vocab
        self._len = len(data_pair)

    def __getitem__(self, index):
        # 获取当前特征文本的index表示，以及oov列表
        x, oov = source2ids(self.src_sents[index], self.vocab)

        # 获取当前标签文本的index表示
        y = abstract2ids(self.trg_sents[index], self.vocab, oov)

        return {
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'OOV': oov,
            'len_OOV': len(oov),
            'y': [self.vocab.SOS] + y + [self.vocab.EOS],
            'x_len': len(self.src_sents[index]),
            'y_len': len(self.trg_sents[index])
        }

    def __len__(self):
        return self._len   

def collate_fn(batch):
    """Split data set into batches and do padding for each batch.

    Args:
        x_padded (Tensor): Padded source sequences.
        y_padded (Tensor): Padded reference sequences.
        x_len (int): Sequence length of the sources.
        y_len (int): Sequence length of the references.
        OOV (dict): Out-of-vocabulary tokens.
        len_OOV (int): Number of OOV tokens.
    """
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item))
                      for item in indice]
        return torch.tensor(pad_indice)

    data_batch = sort_batch_by_len(batch)

    x = data_batch["x"]
    x_max_length = max([len(t) for t in x])
    y = data_batch["y"]
    y_max_length = max([len(t) for t in y])

    OOV = data_batch["OOV"]
    len_OOV = torch.tensor(data_batch["len_OOV"])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch["x_len"])
    y_len = torch.tensor(data_batch["y_len"])
    return x_padded, y_padded, x_len, y_len, OOV, len_OOV