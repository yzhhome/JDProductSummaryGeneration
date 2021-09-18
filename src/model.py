'''
@Author: dzy
@Date: 2021-01-01 11:00:51
@LastEditTime: 2020-09-10 17:36:24
@LastEditors: dzy
@Description: Define the models base on this paper:
              https://arxiv.org/abs/1704.04368
@FilePath: /JDProductSummaryGeneration/src/model.py
'''

import os
import sys
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from utils import timer, replace_oovs

abs_path = pathlib.Path(__file__).parent.absolute()

class Encoder(nn.Module):
    def __init__(self, 
                vocab_size,            # 词典大小
                embed_size,            # 词向量维度                      
                hidden_size,           # 隐藏层大小
                rnn_drop: float = 0):  # dropout的比率

        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, 
                            hidden_size, 
                            bidirectional=True, 
                            dropout=rnn_drop,
                            batch_first=True)

    def forward(self, x, decoder_embedding):
        """Define forward propagation for the endoer.
        Args:
            x (Tensor): The input samples as shape (batch_size, seq_len).
        Returns:
            output (Tensor):
                The output of lstm with shape
                (batch_size, seq_len, 2 * hidden_units).
            hidden (tuple):
                The hidden states of lstm (h_n, c_n).
                Each with shape (2, batch_size, hidden_units)
        """
        if config.weight_tying:
            embedded = decoder_embedding(x)
        else:
            # 先进行embedding操作
            embedded = self.embedding(x)

        # 双向lstm输出
        output, hidden = self.lstm(embedded)

        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()

        # 定义feed-forward层
        # 这里把lstm的输出h_dec和c_dec拼接在一起
        # 所以维度是2*hidden_units
        self.Wh = nn.Linear(2*hidden_units, 2*hidden_units, bias=False)
        self.Ws = nn.Linear(2*hidden_units, 2*hidden_units)

        # wc是用作coverage机制的线性层
        self.wc = nn.Linear(1, 2*hidden_units, bias=False)

        # v是attention的标的向量
        self.v = nn.Linear(2*hidden_units, 1, bias=False)

    def forward(self,
                decoder_states,
                encoder_output,
                x_padding_masks,
                coverage_vector):
        """Define forward propagation for the attention network.
        Args:
            decoder_states (tuple):
                The hidden states from lstm (h_n, c_n) in the decoder,
                each with shape (1, batch_size, hidden_units)
            encoder_output (Tensor):
                The output from the lstm in the decoder with
                shape (batch_size, seq_len, hidden_units).
            x_padding_masks (Tensor):
                The padding masks for the input sequences
                with shape (batch_size, seq_len).
            coverage_vector (Tensor):
                The coverage vector from last time step.
                with shape (batch_size, seq_len).
        Returns:
            context_vector (Tensor):
                Dot products of attention weights and encoder hidden states.
                The shape is (batch_size, 2*hidden_units).
            attention_weights (Tensor): The shape is (batch_size, seq_length).
            coverage_vector (Tensor): The shape is (batch_size, seq_length).
        """
        # 拼接h_dec和c_dec为s_t，并扩展s_t的维度
        h_dec, c_dec = decoder_states

        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)

        # (batch_size, 1, 2*hidden_units)
        s_t = s_t.transpose(0, 1)

        # 扩展和encoder_output一样的维度
        # (batch_size, seq_length, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()

        # 根据公式(1)计算attention score

        # Wh * h_* (batch_size, seq_length, 2*hidden_units)
        encoder_features = self.Wh(encoder_output.contiguous())

        # Ws * s_t (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)

        # (batch_size, seq_length, 2*hidden_units)
        att_inputs = encoder_features + decoder_features

        # 根据论文公式(11)实现coverage机制
        if config.coverage:
            coverage_features = self.wc(coverage_vector.unsqueeze(2))
            att_inputs = att_inputs + coverage_features

        # attention score 
        # (batch_size, seq_length, 1)
        score = self.v(torch.tanh(att_inputs))

        # (batch_size, seq_length)
        attention_weights = F.softmax(score, dim=1).squeeze(2)

        # 过滤掉进行过padding的输入
        attention_weights = attention_weights * x_padding_masks

        # Normalize attention weights after excluding padded positions.
        normalization_factor = attention_weights.sum(1, keepdim=True)
        attention_weights = attention_weights / normalization_factor
        
        # attetton weights和encoder_output进行向量相乘得到输入的context_vector
        # (batch_size, 1, 2*hidden_units)
        context_vector = torch.bmm(attention_weights.unsqueeze(1),  encoder_output)

        # (batch_size, 2*hidden_units)
        context_vector = context_vector.squeeze(1)

        # 更新coverage vector
        # 因为coverage vector 是attenion weights的加和
        if config.coverage:
            coverage_vector = coverage_vector + attention_weights

        return context_vector, attention_weights, coverage_vector        

class Decoder(nn.Module):
    def __init__(self,                 
                vocab_size,
                embed_size,
                hidden_size,
                enc_hidden_size=None,
                is_cuda=True):
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.DEVICE = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Decoder是单向的，所以用单向lstm
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # 最终Decoder部分结合Encoder的输入状态，Context向量，Decoder的历史输入

        # w1输入的维度为双向lstm的2*hidden_size + attention的hidden_size
        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)

        # 第2个全连接层
        self.W2 = nn.Linear(self.hidden_size, vocab_size)

        # 使用pointer generation network
        if config.pointer:
            # context vector的维度: 2*hidden_size
            # decoder states的维度: 2*hidden_size
            # decoder embedding的维度: embed_size
            # 所以wgen输入的维度为: hidden_size * 4
            # wgen输出wgen维度因为要做sigmoid，所以为1
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)

    def forward(self, x_t, decoder_states, context_vector):
        """Define forward propagation for the decoder.

        Args:
            x_t (Tensor):
                The input of the decoder x_t of shape (batch_size, 1).
            decoder_states (tuple):
                The hidden states(h_n, c_n) of the decoder from last time step.
                The shapes are (1, batch_size, hidden_units) for each.
            context_vector (Tensor):
                The context vector from the attention network
                of shape (batch_size,2*hidden_units).

        Returns:
            p_vocab (Tensor):
                The vocabulary distribution of shape (batch_size, vocab_size).
            docoder_states (tuple):
                The lstm states in the decoder.
                The shapes are (1, batch_size, hidden_units) for each.
            p_gen (Tensor):
                The generation probabilities of shape (batch_size, 1).
        """        
        decoder_emb = self.embedding(x_t)

        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # 拼接context vector和decoder state

        # (batch_size, 3*hidden_units)
        decoder_output = decoder_output.view(-1, config.hidden_size)
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1)

         # 根据论文中公式(4)计算vocabulary distribution

        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)

         # (batch_size, vocab_size)
        if config.weight_tying:
            FF2_out = torch.mm(FF1_out, torch.t(self.embedding.weight))
        else:
            FF2_out = self.W2(FF1_out)        

        # vocabulary distribution
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)

        # 以下是根据论文中公式(8)计算p_gen

        # 拼接h_n和c_n得到s_t并扩展s_t的维度
        h_dec, c_dec = decoder_states

        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim = 2)

        p_gen = None
        if config.pointer:       
            # 拼接 context_vector, decoder_state, decoder_emb
            # s_t为当前time step的hidden states的h_n, c_n的拼接
            # decoder_emb为当前的输入进行embedding后的向量
            x_gen = torch.cat([
                context_vector,          # 输入经过attention后的向量
                s_t.squeeze(0),          # s_t第0维为1，要降维
                decoder_emb.squeeze(1)   # decoder_emb第1维也要降维，保持维度统一
            ],  dim=-1)
             
            p_gen = torch.sigmoid(self.w_gen(x_gen))

        return p_vocab, decoder_states, p_gen

class ReduceState(nn.Module):
    """
    对使用双向LSTM的encoder hidden states进行降维
    降维到和使用单向LSTM的decoder的输入一样的维度
    """
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        """The forward propagation of reduce state module.
        Args:
            hidden (tuple):
                Hidden states of encoder,
                each with shape (2, batch_size, hidden_units).
        Returns:
            tuple:
                Reduced hidden states,
                each with shape (1, batch_size, hidden_units).
        """
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)

class PGN(nn.Module):
    def __init__( self, vocab):
        super(PGN, self).__init__()
        self.vocab = vocab
        self.DEVICE = config.DEVICE
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(len(vocab),
                                config.embed_size,
                                config.hidden_size,
        )
        self.decoder = Decoder(len(vocab),
                               config.embed_size,
                               config.hidden_size,
                               )
        self.reduce_state = ReduceState()

    def load_model(self):
        if (os.path.exists(config.encoder_save_name)):
            print('Loading model: ', config.encoder_save_name)
            self.encoder = torch.load(config.encoder_save_name)
            self.decoder = torch.load(config.decoder_save_name)
            self.attention = torch.load(config.attention_save_name)
            self.reduce_state = torch.load(config.reduce_state_save_name)

        elif config.fine_tune:       
            model_path = config.root_path + '/saved_model/pgn/encoder.pt'     
            if os.path.exists(model_path):
                print('Loading model: ', model_path)
                self.encoder = torch.load(model_path)

            model_path = config.root_path + '/saved_model/pgn/decoder.pt'  
            if os.path.exists(model_path):
                print('Loading model: ', model_path)
                self.decoder = torch.load(model_path)

            model_path = config.root_path + '/saved_model/pgn/attention.pt'
            if os.path.exists(model_path):
                print('Loading model: ', model_path)
                self.attention = torch.load(model_path)

            model_path = config.root_path + '/saved_model/pgn/reduce_state.pt'
            if os.path.exists(model_path):
                print('Loading model: ', model_path)
                self.attention = torch.load(model_path)

#     @timer('final dist')
    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights, max_oov):
        """Calculate the final distribution for the model.
        Args:
            x: (batch_size, seq_len)
            p_gen: (batch_size, 1)
            p_vocab: (batch_size, vocab_size)
            attention_weights: (batch_size, seq_len)
            max_oov: (Tensor or int): The maximum sequence length in the batch.
        Returns:
            final_distribution (Tensor):
            The final distribution over the extended vocabualary.
            The shape is (batch_size, )
        """
        if not config.pointer:
            return p_vocab

        batch_size = x.size()[0]

        # Clip the probabilities.
        p_gen = torch.clamp(p_gen, 0.001, 0.999)

        # 根据论文公式(9)计算vocabulary distribution部分的权重
        p_vocab_weighted = p_gen * p_vocab

        # 根据论文公式(9)计算attention distribution部分的权重
        # (batch_size, seq_len)
        attention_weighted = (1 - p_gen) * attention_weights

        # 获取扩展的vocab probability distribution
        # extended_size = len(self.vocab) + max_oovs
        extension = torch.zeros((batch_size, max_oov)).float().to(self.DEVICE)

        # (batch_size, extended_vocab_size)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)

        # 根据论文公式(9)把attention部分的权重添加到vocab部分的权重
        # final distribution = attention distribution + vocab distribution
        final_distribution = p_vocab_extended.scatter_add_(
            dim=1, index=x, src=attention_weighted)

        return final_distribution

#     @timer('model forward')
    def forward(self, x, x_len, y, len_oovs, batch, num_batches, teacher_forcing):
        """Define the forward propagation for the seq2seq model.
        Args:
            x (Tensor):
                Input sequences as source with shape (batch_size, seq_len)
            x_len ([int): Sequence length of the current batch.
            y (Tensor):
                Input sequences as reference with shape (bacth_size, y_len)
            len_oovs (Tensor):
                The numbers of out-of-vocabulary words for samples in this batch.
            batch (int): The number of the current batch.
            num_batches(int): Number of batches in the epoch.
            teacher_forcing(bool): teacher_forcing or not
        Returns:
            batch_loss (Tensor): The average loss of the current batch.
        """
        # 处理输入序列中的oov token
        x_copy = replace_oovs(x, self.vocab)

        # ⽣成输⼊序列x的padding mask
        x_padding_masks = torch.ne(x, 0).byte().float()

        # 双向lstm的output和hidden states
        encoder_output, encoder_states = self.encoder(x_copy)

        # 对encoder的hidden states进行降维
        decoder_states = self.reduce_state(encoder_states)

        # 初台化coverage vector.
        coverage_vector = torch.zeros(x.size()).to(self.DEVICE)

        step_losses = []

        # use ground true to set x_t as first step data for decoder input 
        decoder_input_t = y[:, 0]        

        # 计算每个step的损失        
        for t in range(y.shape[1]-1):

            #对于每⼀个time step
            #以输⼊序列y的y[:, t]作为输⼊，y[:, t+1]作为target，计算attention

            # use ground true to set x_t ,if teacher_forcing is True
            if teacher_forcing:
                decoder_input_t = y[:, t]

            decoder_input_t = replace_oovs(decoder_input_t, self.vocab)

            decoder_target_t = y[:, t+1]

            # 通过attenion获取context vector, attentionweights, coverage vector
            context_vector, attention_weights, coverage_vector = \
                self.attention(decoder_states, encoder_output,
                               x_padding_masks, coverage_vector)

            # 获取decoder的vocab distribution, hidden states, p_gen
            # 找到target对应的词在p_vocab中对应的概率
            p_vocab, decoder_states, p_gen = self.decoder(decoder_input_t.unsqueeze(1),
                                                          decoder_states,
                                                          context_vector)

            final_dist = self.get_final_distribution(x,
                                                     p_gen,
                                                     p_vocab,
                                                     attention_weights,
                                                     torch.max(len_oovs))

            # 如果不是PGN就把decoder_target_t中的oov词为替换为UNK
            if not config.pointer:
                decoder_target_t = replace_oovs(decoder_target_t, self.vocab)

            target_probs = torch.gather(final_dist, 1, decoder_target_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)

            # Apply a mask such that pad zeros do not affect the loss
            mask = torch.ne(decoder_target_t, 0).byte()

            # base line model的损失
            # 加config.eps防止log(0)
            loss = -torch.log(target_probs + config.eps)

            # 把coverage的损失加进来
            if config.coverage:
                ct_min = torch.min(attention_weights, coverage_vector)
                cov_loss = torch.sum(ct_min, dim=1)
                loss = loss + config.LAMBDA * cov_loss

            mask = mask.float()
            loss = loss * mask

            step_losses.append(loss)

        # 计算每个序列的损失
        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)

        # get the non-padded length of each sequence in the batch
        seq_len_mask = torch.ne(y, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        # get batch loss by dividing the loss of each batch
        # by the target sequence length and mean
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss