'''
@Author: dzy
@Date: 2021-09-13 11:07:48
@LastEditTime: 2021-09-26 20:25:17
@LastEditors: dzy
@Description: Use semi-supervised learning to generate new source from existing reference.
@FilePath: /JDProductSummaryGeneration/data/semi-supervise.py
'''

from src import config
import os
from src.predict import Predict
from data_utils import write_samples

def semi_supervised(samples_path, write_path, beam_search):
    """use reference to predict source
    Args:
        samples_path (str): The path of reference
        write_path (str): The path of new samples
    """
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))

    count = 0
    semi = []

    with open(samples_path, 'r') as f:
        for line in f:
            count += 1
            source, ref = line.strip().split('<sep>')

            # 用reference预测source
            prediction = pred.predict(ref.split(), beam_search=beam_search)

            # 组成新的sample
            semi.append(prediction + ' <sep> ' + ref)

            # 每处理100个就保存semi sample
            if count % 100 == 0:
                print('semi sample processed:' , count)
                write_samples(semi, write_path, 'a')
                semi = []

if __name__ == '__main__':
    dir_path = config.root_path + '/data/output'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    samples_path = config.root_path + '/files/train.txt'
    write_path_greedy = dir_path + '/semi_greedy.txt'
    write_path_beam = dir_path + '/semi_beam.txt'

    beam_search = False
    if beam_search:
        write_path = write_path_beam
    else:
        write_path = write_path_greedy

    semi_supervised(samples_path, write_path, beam_search)