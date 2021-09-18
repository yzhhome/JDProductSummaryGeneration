'''
@Author: dzy
@Date: 2021-01-01 11:00:51
@LastEditTime: 2020-09-10 17:36:24
@LastEditors: dzy
@Description: Define configuration parameters.
@FilePath: /JDProductSummaryGeneration/model/config.py
'''

import pathlib
import torch
from typing import Optional

# 通用参数
root_path = str(pathlib.Path(__file__).parent.parent.absolute())
hidden_size: int = 512                              # 隐藏层神经元数量
dec_hidden_size: Optional[int] = 512                # decoder隐藏层神经元数量
embed_size: int = 300                               # 词向量弦度
pointer = True

# 数据参数
max_vocab_size = 50000                              # 词典最大的大小
embed_file: Optional[str] = None                    # 使用预训练的词向量
source = 'big_samples'                              # 两种取值:train 或 big_samples 
data_path: str = root_path + '/files/{}.txt'.format(source)   # 训练数据集文件 
val_data_path: Optional[str] = root_path + '/files/dev.txt'   # 验证数据集文件
test_data_path: Optional[str] = root_path + '/files/test.txt' # 测试数据集文件 
stop_word_file = root_path + '/files/HIT_stop_words.txt'      # 停用词文件 
max_src_len: int = 300  
max_tgt_len: int = 100  
truncate_src: bool = True
truncate_tgt: bool = True
min_dec_steps: int = 30
max_dec_steps: int = 100
enc_rnn_dropout: float = 0.5
enc_attn: bool = True
dec_attn: bool = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0

# 训练参数
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 8
batch_size = 32
coverage = False
fine_tune = False
scheduled_sampling = False
weight_tying = False
max_grad_norm = 2.0
is_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if is_cuda else "cpu")
LAMBDA = 1
update_loss_batch = 1               # 每训练多少个batch更新损失显示

if pointer:                         # 使用pointer generator network
    if coverage:
        if fine_tune:
            model_name = 'ft_pgn'   # 使用fine tune pointer generator network
        else:
            model_name = 'cov_pgn'  # 使用coverage pointer generator network
    elif scheduled_sampling:
        model_name = 'ss_pgn'
    elif weight_tying:
        model_name = 'wt_pgn'
    else:  
        if source == 'big_samples':
            model_name = 'pgn_big_samples'
        else:
            model_name = 'pgn'          # 使用普通pointer generator network
else:
    model_name = 'baseline'         # 使用seq2seq attention base line model,

encoder_save_name = root_path + '/saved_model/' + model_name + '/encoder.pt'

decoder_save_name = root_path + '/saved_model/' + model_name + '/decoder.pt'

attention_save_name = root_path + '/saved_model/' + model_name + '/attention.pt'

reduce_state_save_name = root_path + '/saved_model/' + model_name + '/reduce_state.pt'

losses_path = root_path + '/saved_model/' + model_name + '/val_losses.pkl'

log_path = root_path + '/runs/' + model_name

# Beam search参数
beam_size: int = 3
alpha = 0.2
beta = 0.2
gamma = 0.6