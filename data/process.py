'''
@Author: dzy
@Date: 2021-09-01 20:16:37
@LastEditTime: 2021-09-10 17:28:41
@LastEditors: dzy
@Description: Process a raw dataset into a sample file.
@FilePath: /JDProductSummaryGeneration/data/process.py
'''

import os
import pathlib
import json
import jieba
from data_utils import write_samples, partition

# 当前文件所在的目录
abs_path = pathlib.Path(__file__).parent.absolute()

samples = set()

json_path = os.path.join(abs_path, '../files/服饰_50k.json')

with open(json_path, 'r', encoding='utf-8') as file:
    jsf = json.load(file)

for jsobj in jsf.values():
    # 获取tile
    title = jsobj['title'] + ' '

    # 获取kb属性下所有的值并转换为字典
    kb = dict(jsobj['kb'].items())

    # 所有的属性名属性值合并在一起
    kb_merged = ''
    for key, value in kb.items():
        kb_merged += key + ' ' + value + ' '

    # 获取ocr识别的文本
    ocr = jsobj['ocr']
    ocr = jieba.cut(ocr)
    ocr = ' '.join(list(ocr))

    texts = []
    texts.append(title + ocr + kb_merged)

    reference = jsobj['reference']
    reference = jieba.cut(reference)
    reference = ' '.join(list(reference))

    # 把text和reference用<sep>分隔开
    for text in texts:
        sample = text + '<sep>' + reference
        samples.add(sample)

write_path = os.path.join(abs_path, '../file/samples.txt')

# 写入数据文件中
write_samples(samples, write_path)

# 划分训练集，数据集，测试集
partition(samples)