'''
@Author: dzy
@Date: 2021-09-13 11:07:48
@LastEditTime: 2021-09-26 20:25:17
@LastEditors: dzy
@Description: Replace low tfidf-scored tokens with another token 
              that is similar in the embedding space.
@FilePath: /JDProductSummaryGeneration/data/embed_replace.py
'''

from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
from numpy.lib.function_base import select
from data_utils import read_samples, isChinese, write_samples
import os
from gensim import matutils
from itertools import islice
import numpy as np
from src import config

class EmbedReplace(object):
    def __init__(self, sample_path, wv_path):
        self.samples = read_samples(sample_path)

        # 所有的reference
        self.refs = [sample.split('<sep>')[1].split() for sample in self.samples]

        # 加载word2vec词向量
        self.wv = KeyedVectors.load_word2vec_format(wv_path, binary=False)

        tfidf_path = config.root_path + '/data/saved/tfidf'

        # 如果这前保存过就加载原来的模型
        # 没有就构建词典语料库并保存模型
        if os.path.exists(tfidf_path + '.model'):
            self.tfidf_model = TfidfModel.load(tfidf_path + '.model')
            self.dct = Dictionary.load(tfidf_path + '.dict')
            # 语料库
            self.corpus = [self.dct.doc2bow(doc) for doc in self.refs]
        else:
            self.dct = Dictionary(self.refs)
            self.corpus = [self.dct.doc2bow(doc) for doc in self.refs]
            self.tfidf_model = TfidfModel(self.refs)
            self.dct.save(tfidf_path + '.dcit')
            self.tfidf_model.save(tfidf_path + '.model')
            self.vocab_size = len(self.dct.token2id)

    def vectorize(self, docs, vocab_size):
        """
        把语料库转换成稠密矩阵表示
        """
        return matutils.corpus2dense(docs, vocab_size)

    def extract_keywords(self, dct, tfidf, threshold=0.2, topk=5):
        """find high TFIDF socore keywords
        Args:
            dct (Dictionary): gensim.corpora Dictionary  a reference Dictionary
            tfidf (list of tfidf):  model[doc]  [(int, number)]
            threshold (float) : high TFIDF socore must be greater than the threshold
            topk(int): num of highest TFIDF socore 
        Returns:
            (list): A list of keywords
        """
        # 先对tfidf score按降序排序
        tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)

        # 再返回topk个词
        return list(slice([dct[w] for w, score in tfidf if score > threshold], topk))

    
    def replace(self, token_list, doc):
        """replace token by another token which is similar in wordvector 
        Args:
            token_list (list): reference token list
            doc (list): A reference represented by a word bag model
        Returns:
            (str):  new reference str
        """
        keywords = self.extract_keywords(self.dct, self.tfidf_model[doc])
        num = int(len(token_list) * 0.3)
        new_tokens = token_list.copy()

        while num == int(len(token_list) * 0.3):
            # 随机选取num个token的index
            indexes = np.random.choice(len(token_list), num)

            for index in indexes:
                token = token_list[index]

                # 从词向量中选取最相似的词向量
                if isChinese(token) and token not in keywords and token in self.wv:
                    new_tokens[index] = self.wv.most_similar(
                        positive=token, negative=None, topn=1)[0][0]
            num -= 1

        return ' '.join(new_tokens)

    def generate_samples(self, write_path):
        """generate new samples file
        Args:
            write_path (str):  new samples file path
        """
        replaced = []
        count = 0
        for sample, token_list, doc in zip(self.samples, self.refs, self.corpus):
            count += 1

            # 每生成100个新sample保存一次
            if count % 100 == 0:
                print(count)
                write_samples(replaced, write_path, 'a')
                replaced = []

            # 新的sample只替换reference部分
            replaced.append(sample.split('<sep>')[0] + ' <sep> ' + 
                self.replace(token_list, doc))

if __name__ == '__main__':
    dir_path_list = [config.root_path + '/data/output',
                    config.root_path + '/data/word_vectors',
                    config.root_path + '/data/saved']
    
    for dir_path in dir_path_list:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    sample_path = config.root_path + '/files/train.txt'
    wv_path = dir_path_list[1] + '/merge_sgns_bigram_char300.txt'
    replacer = EmbedReplace(sample_path, wv_path)
    replacer.generate_samples(dir_path_list[0] + '/replaced.txt')