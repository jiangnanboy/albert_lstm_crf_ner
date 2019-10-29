# -*- coding:utf-8 -*-

import copy

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
# import torchsnooper  # 用于调试，运行时打印每行代码中tensor的信息

from albert.model.modeling_albert import BertConfig, BertModel
from albert.configs.base import config

START_TAG = "[CLS]"
STOP_TAG = "[SEP]"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_sum_exp(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)


class BiLSTMCRF(nn.Module):
    def __init__(
            self,
            tag_map={'[PAD]': 0,
                     'O': 1,
                     'B_T': 2,
                     'I_T': 3,
                     'B_LOC': 4,
                     'I_LOC': 5,
                     'B_ORG': 6,
                     'I_ORG': 7,
                     'B_PER': 8,
                     'I_PER': 9,
                     '[CLS]': 10,
                     '[SEP]': 11},
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
        # 标签转移概率矩阵
        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size, device=DEVICE)
        )
        self.transitions.detach()[:, self.tag_map[START_TAG]] = -1000.
        self.transitions.detach()[self.tag_map[STOP_TAG], :] = -1000.
        # self.transitions = self.transitions.to(DEVICE)
        bert_config = BertConfig.from_pretrained(str(config['albert_config_path']), share_type='all')
        self.word_embeddings = BertModel.from_pretrained(config['bert_dir'], config=bert_config)
        self.word_embeddings.to(DEVICE)
        self.word_embeddings.eval()

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2, device=DEVICE),
                torch.randn(2, self.batch_size, self.hidden_dim // 2, device=DEVICE))

    def __get_lstm_features(self, sentence, mask):
        self.batch_size = sentence.size(0)
        self.hidden = self.init_hidden()
        length = sentence.shape[1]
        embeddings = self.word_embeddings(input_ids=sentence, attention_mask=mask)
        all_hidden_states, all_attentions = embeddings[-2:]  # 这里获取所有层的hidden_satates以及attentions
        embeddings = all_hidden_states[-2]  # 倒数第二层hidden_states的shape

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        lstm_out = lstm_out.reshape(self.batch_size, -1, self.hidden_dim)  # batch,embedding_dim,hidden_dim

        # lstm_out = lstm_out.reshape(-1, self.hidden_dim)
        # l_out = self.hidden2tag(lstm_out)
        # lstm_feats = l_out.reshape(self.batch_size, length, -1)
        # print('after lstm_feats shape:{}'.format(lstm_feats.shape))

        logits = self.hidden2tag(lstm_out)
        return logits

    def real_path_score_(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = 0
        tags = torch.cat([torch.tensor([self.tag_map[START_TAG]], dtype=torch.long).to(DEVICE), tags])
        tags = tags.to(DEVICE)
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i], tags[i + 1]] + feat[tags[i + 1]]
        score = score + self.transitions[tags[-1], self.tag_map[STOP_TAG]]
        return score

    def real_path_score(self, logits, label):
        '''
        caculate real path score
        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * len_sent]

        Score = Emission_Score + Transition_Score
        Emission_Score = logits(0, label[START]) + logits(1, label[1]) + ... + logits(n, label[STOP])
        Transition_Score = Trans(label[START], label[1]) + Trans(label[1], label[2]) + ... + Trans(label[n-1], label[STOP])
        '''
        score = 0
        label = torch.cat([torch.tensor([self.tag_map[START_TAG]], dtype=torch.long).to(DEVICE), label])
        label = label.to(DEVICE)
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score
        score += self.transitions[label[-1], self.tag_map[STOP_TAG]]
        return score

    def total_score(self, logits, label):
        """
        caculate total score

        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * tag_size]

        SCORE = log(e^S1 + e^S2 + ... + e^SN)
        """
        obs = []
        previous = torch.full((1, self.tag_size), 0, device=DEVICE)
        for index in range(len(logits)):
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag_map[STOP_TAG]]
        # caculate total_scores
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores

    def neg_log_likelihood(self, sentences, masks, tags):
        logits = self.__get_lstm_features(sentences, masks)
        real_path_score = 0
        total_score = 0
        for logit, tag in zip(logits, tags):
            real_path_score += self.real_path_score(logit, tag)
            total_score += self.total_score(logit, tag)
        return total_score - real_path_score

    def forward(self, sentences, masks):
        """
        :params sentences sentences to predict
        :params lengths represent the ture length of sentence, the default is sentences.size(-1)
        """
        logits = self.__get_lstm_features(sentences, masks)
        scores = []
        paths = []
        for logit in logits:
            score, path = self.__viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores, paths

    def __viterbi_decode(self, logits):
        backpointers = []
        trellis = torch.zeros(logits.size(), device=DEVICE)
        backpointers = torch.zeros(logits.size(), dtype=torch.long)

        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi

    def __viterbi_decode_v1(self, logits):
        init_prob = 1.0
        trans_prob = self.transitions.t()
        prev_prob = init_prob
        path = []
        for index, logit in enumerate(logits):
            if index == 0:
                obs_prob = logit * prev_prob
                prev_prob = obs_prob
                prev_score, max_path = torch.max(prev_prob, -1)
                path.append(max_path.cpu().tolist())
                continue
            obs_prob = (prev_prob * trans_prob).t() * logit
            max_prob, _ = torch.max(obs_prob, 1)
            _, final_max_index = torch.max(max_prob, -1)
            prev_prob = obs_prob[final_max_index]
            prev_score, max_path = torch.max(prev_prob, -1)
            path.append(max_path.cpu().tolist())
        return prev_score.cpu().tolist(), path
