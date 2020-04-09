#import tensorflow as tf
import numpy as np
import data_processing
import config
import data_utils
#import seq2seq_wrapper
from os import path

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#load data and split into train and test sets
idx_headings, idx_descriptions = data_processing.process_data()
print('Printing 1 idx_headings -')
print(idx_headings[1])
print('Printing 1 idx_description - ')
print(idx_descriptions[1])


article_metadata = data_processing.unpickle_articles()
print('Printing Article Meta Data from Pickled File - ')
print(article_metadata)

(x_train, x_test), (y_train, y_test), (x_valid, y_valid) = data_utils.split_data(idx_descriptions, idx_headings)
print('Printing x_train and its shape - ')
print(x_train)
print(x_train.shape)
print('Printing x_test - ')
print(x_test)
print(x_test.shape)

print('Printing y_train - ')
print(y_train)
print(y_train.shape)
print('Printing y_test - ')
print(y_test)
print(y_test.shape)

print('Printing x_valid - ')
print(x_valid)
print(x_valid.shape)
print('Printing y_valid -')
print(y_valid)
print(y_valid.shape)

#define parameters
xseq_length = x_train.shape[-1]
print("xseq_length = ", xseq_length)
yseq_length = y_train.shape[-1]
print("yseq_length = ", yseq_length)
batch_size = config.batch_size
xvocab_size = len(article_metadata['idx2word'])
print('xvocab size = ', xvocab_size)
yvocab_size = xvocab_size
print("printing y_vocab size = ", yvocab_size)
checkpoint_path = path.join(config.path_outputs, 'checkpoint')
print(checkpoint_path)

#define model



#model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_length,
#                                yseq_len=yseq_length,
#                                xvocab_size=xvocab_size,
#                                yvocab_size=yvocab_size,
#                                emb_dim=config.embedding_dim,
#                                num_layers=3,
#                                ckpt_path=checkpoint_path)

val_batch_gen = data_utils.generate_random_batch(x_valid, y_valid, config.batch_size)
train_batch_gen = data_utils.generate_random_batch(x_train, y_train, config.batch_size)





#sess = model.restore_last_session()
#sess = model.train(train_batch_gen, val_batch_gen)