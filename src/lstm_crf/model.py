# -*- coding:utf-8 -*-

import numpy as np

import torch
from torch import nn

from albert.model.modeling_albert import BertConfig, BertModel
from albert.configs.base import config
from lstm_crf.crf import CRF
from sklearn.metrics import f1_score, classification_report

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# import torchsnooper

def log_sum_exp(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)


class BiLSTMCRF(nn.Module):

    def __init__(
            self,
            tag_map={ 'B_T': 0,
                   'I_T': 1,
                   'B_LOC': 2,
                   'I_LOC': 3,
                   'B_ORG': 4,
                   'I_ORG': 5,
                   'B_PER': 6,
                   'I_PER': 7,
                        'O': 8},
            batch_size=20,
            hidden_dim=128,
            dropout=1.0,
            embedding_dim=100
    ):
        super(BiLSTMCRF, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.tag_size = len(tag_map)  # 标签个数
        self.tag_map = tag_map

        bert_config = BertConfig.from_pretrained(str(config['albert_config_path']), share_type='all')
        self.word_embeddings = BertModel.from_pretrained(config['bert_dir'], config=bert_config)
        self.word_embeddings.to(DEVICE)
        self.word_embeddings.eval()

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size)

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2, device=DEVICE),
                torch.randn(2, self.batch_size, self.hidden_dim // 2, device=DEVICE))

    def forward(self, input_ids, attention_mask):
        self.batch_size = input_ids.size(0)
        self.hidden = self.init_hidden()
        with torch.no_grad():
            embeddings = self.word_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        # 因为在albert中的config中设置了"output_hidden_states":"True","output_attentions":"True"，所以返回所有层
        # 也可以只返回最后一层
        all_hidden_states, all_attentions = embeddings[-2:]  # 这里获取所有层的hidden_satates以及attentions
        embeddings = all_hidden_states[-2]  # 倒数第二层hidden_states的shape
        lstm_out, _ = self.lstm(embeddings, self.hidden)
        output = self.hidden2tag(lstm_out)
        return output

    def loss_fn(self, bert_encode, output_mask, tags): #bert_encode是bert的输出
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        # 以下是用于主程序中的评估eval_1(); acc_f1,class_report的评估
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct / y_pred.shape[0]
        print('acc: {}'.format(acc))
        print('f1: {}'.format(f1))
        return acc, f1

    def class_report(self, y_pred, y_true):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)
