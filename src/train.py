'''
@Author: dzy
@Date: 2021-09-13 11:07:48
@LastEditTime: 2021-09-26 20:25:17
@LastEditors: dzy
@Description: Helper functions or classes used for the model.
@FilePath: /JDProductSummaryGeneration/src/train.py
'''

import pickle
import os
import sys
import pathlib

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tensorboardX import SummaryWriter

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

from dataset import PairDataset
from model import PGN
import config
from evaluate import evaluate
from dataset import collate_fn, SampleDataset
from utils import ScheduledSampler, config_info

def train(dataset, val_dataset, vocab, start_epoch=0):
    """Train the model, evaluate it and store it.
    Args:
        dataset (dataset.PairDataset): The training dataset.
        val_dataset (dataset.PairDataset): The evaluation dataset.
        v (vocab.Vocab): The vocabulary built from the training dataset.
        start_epoch (int, optional): The starting epoch number. Defaults to 0.
    """
    DEVICE = torch.device("cuda" if config.is_cuda else "cpu")

    model = PGN(vocab)
    model.load_model()
    model.to(DEVICE)

    if config.fine_tune:
        # fine-tuning模式时除了attention.wc的参数外
        # 其他所有模型的参数都不更新梯度
        print('Fine-tuning mode.')
        for name, params in model.named_parameters():
            if name != 'attention.wc.weight':
                params.requires_grad=False    

    print("loading data")
    train_data = SampleDataset(dataset.pairs, vocab)
    val_data = SampleDataset(val_dataset.pairs, vocab)

    print("initializing optimizer")

    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(),
                              lr=config.learning_rate,
                              )
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    # 验证损失初始化为正无穷大
    val_losses = np.inf

    # 如果之前有保存损失则加载
    if (os.path.exists(config.losses_path)):
        with open(config.losses_path, 'rb') as f:
            val_losses = pickle.load(f)

    # SummaryWriter保存日志作为TensorboardX可视化
    writer = SummaryWriter(config.log_path)

    # scheduled_sampler : A tool for choosing teacher_forcing or not
    num_epochs =  len(range(start_epoch, config.epochs))
    scheduled_sampler = ScheduledSampler(num_epochs)    

    if config.scheduled_sampling:
        print('scheduled_sampling mode.')

    # 用tqdm显示训练进度
    with tqdm(total=config.epochs) as epoch_progress:
        for epoch in range(start_epoch, config.epochs):

            # 设置为训练模式 
            model.train()  

            # epoch中所有batch的损失
            batch_losses = []

            # train_dataloader中的batch数量
            num_batches = len(train_dataloader)

            # set a teacher_forcing signal
            if config.scheduled_sampling:
                teacher_forcing = scheduled_sampler.teacher_forcing(epoch - start_epoch)
            else:
                teacher_forcing = True

            print('teacher_forcing = {}'.format(teacher_forcing))           

            with tqdm(total = num_batches // config.update_loss_batch) as batch_progress:
                for batch, data in enumerate(tqdm(train_dataloader)):

                    x, y, x_len, y_len, oov, len_oovs = data

                    assert not np.any(np.isnan(x.numpy()))

                    # 在GPU上进行
                    if config.is_cuda:
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        x_len = x_len.to(DEVICE)
                        len_oovs = len_oovs.to(DEVICE)

                    # 梯度清零
                    optimizer.zero_grad()  

                    # 输出模型损失
                    loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=num_batches, 
                        teacher_forcing=teacher_forcing)

                    # 添加到batch的损失列表
                    batch_losses.append(loss.item())
                    loss.backward()  

                    # 进行梯度裁剪防止梯度爆炸
                    clip_grad_norm_(model.encoder.parameters(),
                                    config.max_grad_norm)
                    clip_grad_norm_(model.decoder.parameters(),
                                    config.max_grad_norm)
                    clip_grad_norm_(model.attention.parameters(),
                                    config.max_grad_norm)

                    # 更新参数
                    optimizer.step() 

                    # 每隔多少个batch更新loss显示
                    if (batch % config.update_loss_batch) == 0:
                        batch_progress.set_description(f'Epoch {epoch}')
                        batch_progress.set_postfix(Batch=batch,
                                                   Loss=loss.item())
                        batch_progress.update()

                        # Write loss for tensorboard.
                        writer.add_scalar(f'Average loss for epoch {epoch}',
                                          np.mean(batch_losses),
                                          global_step=batch)

            # 计算每个epoch中所有batch的平均损失
            epoch_loss = np.mean(batch_losses)

            epoch_progress.set_description(f'Epoch {epoch}')
            epoch_progress.set_postfix(Loss=epoch_loss)
            epoch_progress.update()

            # 用验证数据集进行验证，返回平均验证损失
            avg_val_loss = evaluate(model, val_data, epoch)

            print('training loss:{}'.format(epoch_loss),
                  'validation loss:{}'.format(avg_val_loss))

            # 更新并保存最小验证集损失
            if (avg_val_loss < val_losses):                
                # 保存验证损失小的模型
                torch.save(model.encoder, config.encoder_save_name)
                torch.save(model.decoder, config.decoder_save_name)
                torch.save(model.attention, config.attention_save_name)
                torch.save(model.reduce_state, config.reduce_state_save_name)

                # 更新验证损失
                val_losses = avg_val_loss

            # 保存验证集损失
            with open(config.losses_path, 'wb') as f:
                pickle.dump(val_losses, f)

    writer.close()

if __name__ == "__main__":

    DEVICE = torch.device('cuda') if config.is_cuda else torch.device('cpu')

    dataset = PairDataset(config.data_path,
                          max_src_len=config.max_src_len,
                          max_tgt_len=config.max_tgt_len,
                          truncate_src=config.truncate_src,
                          truncate_tgt=config.truncate_tgt)

    val_dataset = PairDataset(config.val_data_path,
                              max_src_len=config.max_src_len,
                              max_tgt_len=config.max_tgt_len,
                              truncate_src=config.truncate_src,
                              truncate_tgt=config.truncate_tgt)

    vocab = dataset.build_vocab(embed_file=config.embed_file)

    train(dataset, val_dataset, vocab, start_epoch=0)