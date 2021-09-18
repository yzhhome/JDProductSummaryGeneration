'''
@Author: dzy
@Date: 2021-09-01 20:16:37
@LastEditTime: 2021-09-10 17:28:41
@LastEditors: dzt
@Description: Process a raw dataset into a sample file.
@FilePath: /JDProductSummaryGeneration/data/process.py
'''

import os
import pathlib
from src import config
from typing import List

# 当前文件所在的目录
abs_path = pathlib.Path(__file__).parent.absolute()

def read_samples(file_name: str) -> List:
    """
    读取数据文件中所有的数据，返回列表
    """
    samples = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            samples.append(line.strip())
    return samples

def write_samples(samples: List, file_path: str, opt='w'):    
    """
    写入samples列表中的每行到数据文件中
    """
    with open(file_path, opt, encoding='utf-8') as f:
        for line in samples:
            f.write(line)
            f.write('\n')

def partition(samples: List):
    """
    划分训练集，验证集，测试集
    """
    train, dev, test = [], [], []
    count = 0
    
    for sample in samples:
        count += 1

        # 测试集为1000条
        if count <= 1000:
            test.append(sample)
        # 验证集为5000条
        elif count <= 6000:
            dev.append(sample)
        # 其他的为训练集
        else:
            train.append(sample)

    write_samples(train, os.path.join(abs_path, '../files/train.txt'))
    write_samples(dev, os.path.join(abs_path, '../files/dev.txt'))
    write_samples(test, os.path.join(abs_path, '../files/test.txt'))

def isChinese(word) -> bool:
    """
    判断一个字是否为中文
    """
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def merge_samples():
    """
    将进行数据增强后的数据和原始训练数据集合并到一起
    """
    train_sample = config.root_path + '/data/files/train.txt'  
    replaced_sample = config.root_path + '/data/output/replaced.txt'    

    back_translated_sample = config.root_path + '/data/output/back_translated.txt'
    semi_beam_sample = config.root_path + '/data/output/semi_beam.txt'

    big_sample = config.root_path + '/data/files/big_sample.txt'

    sample_list = [train_sample, 
                    replaced_sample, 
                    back_translated_sample, 
                    semi_beam_sample]  

    for sample_path in sample_list:        
        if  not os.path.exists(sample_path):
            continue

        with open(sample_path, 'r', encoding='utf-8') as sf:
            if not os.path.exists(big_sample):
                with open(big_sample, 'w', encoding='utf-8') as tf:
                    tf.write(sf.read().strip())
            else:
                with open(big_sample, 'a', encoding='utf-8') as tf:
                    tf.write('\n')
                    tf.write(sf.read().strip())