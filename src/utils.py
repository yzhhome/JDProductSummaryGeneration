'''
@Author: dzy
@Date: 2021-09-13 11:07:48
@LastEditTime: 2021-09-26 20:25:17
@LastEditors: dzy
@Description: Helper functions or classes used for the model.
@FilePath: /JDProductSummaryGeneration/src/utils.py
'''

import numpy as np
import time
import heapq
import random
import sys
import pathlib
import torch
import config

abs_path = pathlib.Path(__file__).parent.absolute()

def timer(module):
    """Decorator function for a timer.

    Args:
        module (str): Description of the function being timed.
    """
    def wrapper(func):
        """Wrapper of the timer function.

        Args:
            func (function): The function to be timed.
        """
        def cal_time(*args, **kwargs):
            """The timer function.

            Returns:
                res (any): The returned value of the function being timed.
            """
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for ', module)
            return res
        return cal_time
    return wrapper

def simple_tokenizer(text):
    return text.split()

def count_words(counter, text):
    """
    统计每个词出现的次数
    """
    for sentence in text:
        for word in sentence:
            counter[word] += 1

def sort_batch_by_len(data_batch):
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    # Sort indices of data in batch by lengths.
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()

    data_batch = {
        name: [_tensor[i] for i in sorted_indices]
        for name, _tensor in res.items()
    }
    return data_batch

def outputids2words(id_list, source_oovs, vocab):
    """
        Maps output ids to words, including mapping in-source OOVs from
        their temporary ids to the original OOV string (applicable in
        pointer-generator mode).
        Args:
            id_list: list of ids (integers)
            vocab: Vocabulary object
            source_oovs:
                list of OOV words (strings) in the order corresponding to
                their temporary source OOV ids (that have been assigned in
                pointer-generator mode), or None (in baseline mode)
        Returns:
            words: list of words (strings)
    """
    words = []
    for i in id_list:
        try:
            w = vocab.index2word[i]  # might be [UNK]
        except IndexError:  # w is OOV
            assert_msg = "Error: cannot find the ID the in the vocabulary."
            assert source_oovs is not None, assert_msg
            source_oov_idx = i - vocab.size()
            try:
                w = source_oovs[source_oov_idx]
            except ValueError:  # i doesn't correspond to an source oov
                raise ValueError(
                    'Error: model produced word ID %i corresponding to source OOV %i \
                     but this example only has %i source OOVs'
                    % (i, source_oov_idx, len(source_oovs)))
        words.append(w)
    return ' '.join(words)

def source2ids(source_words, vocab):
    """
    把词映射成id表示，返回所有词的索引和oov中的词索引
    """
    ids = []
    oovs = []
    unk_id = vocab.UNK
    for w in source_words:
        i = vocab[w]

        # oov的词
        if i == unk_id: 
            # 添加到oov中
            if w not in oovs:  
                oovs.append(w)
           
            # 词在oov中的索引
            oov_num = oovs.index(w)

            # 在所有词的索引为词典大小+在oov中的索引
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, source_oovs):
    """Map tokens in the abstract (reference) to ids.
       OOV tokens in the source will be remained.
    Args:
        abstract_words (list): Tokens in the reference.
        vocab (vocab.Vocab): The vocabulary.
        source_oovs (list): OOV tokens in the source.
    Returns:
        list: The reference with tokens mapped into ids.
    """
    ids = []
    unk_id = vocab.UNK

    for w in abstract_words:
        i = vocab[w]

        # w是一个oov词
        if i == unk_id:  
            # w在source oov中出现过
            if w in source_oovs:
                # 直接映射成source oov中的索引
                vocab_idx = vocab.size() + source_oovs.index(w)
                ids.append(vocab_idx)
            # w不在source oov中
            else: 
                # 映射成UNK的索引
                ids.append(unk_id)
        else:
            # 不是oov词直接添加词的索引
            ids.append(i)
    return ids


class Beam(object):
    def __init__(self,
                 tokens,
                 log_probs,
                 decoder_states,
                 coverage_vector):

        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.coverage_vector = coverage_vector

    def extend(self,
               token,
               log_prob,
               decoder_states,
               coverage_vector):

        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    coverage_vector=coverage_vector)

    def seq_score(self):
        """
        This function calculate the score of the current sequence.
        The scores are calculated according to the definitions in
        https://opennmt.net/OpenNMT/translation/beam_search/.
        1. Lenth normalization is used to normalize the cumulative score
        of a whole sequence.
        2. Coverage normalization is used to favor the sequences that fully
        cover the information in the source. (In this case, it serves different
        purpose from the coverage mechanism defined in PGN.)
        3. Alpha and beta are hyperparameters that used to control the
        strengths of ln and cn.
        """
        len_Y = len(self.tokens)

        # Lenth normalization
        ln = (5+len_Y)**config.alpha / (5+1)**config.alpha

        # Coverage normalization
        cn = config.beta * torch.sum( 
            torch.log(config.eps +
                torch.where(self.coverage_vector < 1.0,
                    self.coverage_vector,
                    torch.ones((1, self.coverage_vector.shape[1])).to \
                        (torch.device(config.DEVICE)))))

        score = sum(self.log_probs) / ln + cn
        return score

    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    def __le__(self, other):
        return self.seq_score() <= other.seq_score()


def add2heap(heap, item, k):
    """Maintain a heap with k nodes and the smallest one as root.

    Args:
        heap (list): The list to heapify.
        item (tuple):
            The tuple as item to store.
            Comparsion will be made according to values in the first position.
            If there is a tie, values in the second position will be compared,
            and so on.
        k (int): The capacity of the heap.
    """
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)


def replace_oovs(in_tensor, vocab):
    """Replace oov tokens in a tensor with the <UNK> token.

    Args:
        in_tensor (Tensor): The tensor before replacement.
        vocab (vocab.Vocab): The vocabulary.

    Returns:
        Tensor: The tensor after replacement.
    """    
    # 填充一个和输入tensor一样shape的全是UNK的矩阵
    oov_token = torch.full(in_tensor.shape, vocab.UNK).long().to(config.DEVICE)

    # 不在词典中的用oov_token, 在词典中的用原来的值
    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)
    
    return out_tensor


class ScheduledSampler():
    def __init__(self, phases):
        self.phases = phases
        self.scheduled_probs = [i / (self.phases - 1) for i in range(self.phases)]

    def teacher_forcing(self, phase):
        """According to a certain probability to choose whether to execute teacher_forcing
        Args:
            phase (int): probability level  if phase = 0, 100% teacher_forcing ,
            phase = self.phases - 1, 0% teacher_forcing 

        Returns:
            bool: teacher_forcing or not 
        """
        sampling_prob = random.random()
        if sampling_prob >= self.scheduled_probs[phase]:
            return True
        else:
            return False


def config_info(config):
    """get some config information
    Args:
        config (model): define in  model/config.py
    Returns:
        string: config information
    """
    info = 'model_name = {}, pointer = {}, coverage = {}, fine_tune = {}, ' + \
            'scheduled_sampling = {}, weight_tying = {}, source = {}  '

    return (info.format(config.model_name, config.pointer, config.coverage, 
            config.fine_tune, config.scheduled_sampling, config.weight_tying, 
            config.source))