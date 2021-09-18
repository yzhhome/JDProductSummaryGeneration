'''
@Author: dzy
@Date: 2021-09-01 14:18:13
@LastEditTime: 2021-09-10 20:07:26
@LastEditors: dzy
@Description: Define the vocabulary object
@FilePath: /JDProductSummaryGeneration/model/vocab.py
'''

from collections import Counter
from typing import List
import numpy as np

class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.reserved[:]
        self.embeddings = None

    def add_words(self, words: List):
        """
        添加单词到词典中
        """
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)

        # 更新单词计数
        self.word2count.update(words)

    def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
        """
        从文件中加载词向量，返回加载的词向量个数
        """
        num_embeddings = 0
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for line in f:
                # 分隔出词和词向量
                line = line.split()
                
                # 取出词
                word = line[0].decode('utf-8')

                # 取词在词典中的索引
                idx = self.word2index.get(word)

                if idx is not None:
                    # 取出词向量
                    vec = np.array(line[1:], dtype=dtype)

                    if self.embeddings is None:
                        # 词向量的维度
                        n_dims = len(vec)

                        # embeddings初始化全为0的正太分布
                        self.embeddings = np.random.normal(
                            np.zeros((vocab_size, n_dims))).astype(dtype)
                        
                        # PAD为embeddings中第0个词
                        self.embeddings[self.PAD] = np.zeros(n_dims)

                    # 添加到embeddings
                    self.embeddings[idx] = vec
                    num_embeddings += 1
            
        return num_embeddings

    def __getitem__(self, item):
        """
        获取词典中的值
        如果item为整数，则返回索引对应的词
        如果item为字符串，则返回词对应的索引
        """
        if type(item) is int:
            return self.index2word[item]
        elif type(item) is str:
            return self.word2index.get(item, self.UNK)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.index2word)

    def size(self):
        return len(self.index2word)  
                        