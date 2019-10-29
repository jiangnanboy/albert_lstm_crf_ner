# -*- coding:utf-8 -*-

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from configs.base import config
from albert.model.tokenization_bert import BertTokenizer

import os

class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask

class DataFormat():
    def __init__(self, max_length=100, batch_size=20, data_type='train'):
        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size
        self.max_length = max_length
        self.data_type = data_type
        self.train_data = []
        self.tag_map = {'[PAD]': 0,
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
                   '[SEP]': 11}
        base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        if data_type == "train":
            self.data_path = base_path + '/data/ner_data/train/'
        elif data_type == "dev":
            self.data_path = base_path + "/data/ner_data/dev/"
        elif data_type == "test":
            self.data_path = base_path + "/data/ner_data/test/"

        self.read_corpus(self.data_path + 'source.txt', self.data_path + 'target.txt', self.max_length, self.tag_map)
        self.train_dataloader= self.prepare_batch(self.train_data, self.batch_size)

    def read_corpus(self, train_file_data, train_file_tag, max_length, label_dic):
        """
        :param train_file_data:训练数据
        :param train_file_tag: 训练数据对应的标签
        :param max_length: 训练数据每行的最大长度
        :param label_dic: 标签对应的索引
        :return:
        """
        VOCAB = config['albert_vocab_path']  # your path for model and vocab
        tokenizer = BertTokenizer.from_pretrained(VOCAB)
        with open(train_file_data, 'r', encoding='utf-8') as file_train:
            with open(train_file_tag, 'r', encoding='utf-8') as file_tag:
                train_data = file_train.readlines()
                tag_data = file_tag.readlines()
                for text, label in zip(train_data, tag_data):
                    tokens = text.split()
                    label = label.split()
                    if len(tokens) > max_length:  # 大于最大长度进行截断
                        tokens = tokens[0:max_length]
                        label = label[0:max_length]
                    tokens_cs =  ' '.join(tokens)
                    label_cs =  ' '.join(label)
                    # token -> index
                    tokenized_text = tokenizer.tokenize(tokens_cs)  # 用tokenizer对句子分词
                    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表

                    # tag -> index
                    label_ids = [label_dic[i] for i in label_cs.split()]
                    input_mask = [1] * len(input_ids)

                    while len(input_ids) < max_length:
                        input_ids.append(0)
                        input_mask.append(0)
                        label_ids.append(0)
                    assert len(input_ids) == max_length
                    assert len(input_mask) == max_length
                    assert len(label_ids) == max_length
                    feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
                    self.train_data.append(feature)

    def prepare_batch(self, train_data, batch_size):
        '''
            prepare data for batch
        '''
        train_ids = torch.LongTensor([temp.input_id for temp in train_data])
        train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
        train_tags = torch.LongTensor([temp.label_id for temp in train_data])
        train_dataset = TensorDataset(train_ids, train_masks, train_tags)
        # train_sampler = RandomSampler(train_dataset)
        return DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

