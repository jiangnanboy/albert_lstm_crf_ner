# -*- coding: utf-8 -*-
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from configs.base import config
from albert.model.modeling_albert import BertConfig, BertModel
from albert.model.tokenization_bert import BertTokenizer

import os

class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask

def read_corpus(train_file_data, train_file_tag, max_length, label_dic):
    """
    :param train_file_data:训练数据
    :param train_file_tag: 训练数据对应的标签
    :param max_length: 训练数据每行的最大长度
    :param label_dic: 标签对应的索引
    :return:
    """

    VOCAB = config['albert_vocab_path']  # your path for model and vocab
    tokenizer = BertTokenizer.from_pretrained(VOCAB)
    result = []
    with open(train_file_data, 'r', encoding='utf-8') as file_train:
        with open(train_file_tag, 'r', encoding='utf-8') as file_tag:
            train_data = file_train.readlines()
            tag_data = file_tag.readlines()
            for text, label in zip(train_data, tag_data):
                tokens = text.split()
                label = label.split()
                if len(tokens) > max_length-2: #大于最大长度进行截断
                    tokens = tokens[0:(max_length-2)]
                    label = label[0:(max_length-2)]
                tokens_cs ='[CLS] ' + ' '.join(tokens) + ' [SEP]'
                label_cs = "[CLS] " + ' '.join(label) + ' [SEP]'
                # token -> index
                tokenized_text = tokenizer.tokenize(tokens_cs)  # 用tokenizer对句子分词
                input_ids  = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表

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
                result.append(feature)
    return result

device = torch.device('cuda' if torch.cuda.is_available()  else "cpu")
MAX_LEN = 10
if __name__ == '__main__':
    bert_config = BertConfig.from_pretrained(str(config['albert_config_path']), share_type='all')
    model = BertModel.from_pretrained(config['bert_dir'],config=bert_config)
    model.to(device)
    model.eval()

    base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    train_source = base_path + '/data/ner_data/dev/source.txt'
    train_target = base_path + '/data/ner_data/dev/target.txt'
    tag_dic = {'[PAD]':0,
                'O':1,
                'B_T':2,
                'I_T':3,
                'B_LOC':4,
                'I_LOC':5,
                'B_ORG':6,
                'I_ORG':7,
                'B_PER':8,
                'I_PER':9,
                '[CLS]':101,
                '[SEP]':102 }
    train_data = read_corpus(train_file_data=train_source, train_file_tag=train_target,max_length=15, label_dic=tag_dic)
    train_ids = torch.LongTensor([temp.input_id for temp in train_data])
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
    train_tags = torch.LongTensor([temp.label_id for temp in train_data])
    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=10)

    with torch.no_grad():
        '''
        note:
        一.
        如果设置："output_hidden_states":"True"和"output_attentions":"True"
        输出的是： 所有层的 sequence_output, pooled_output, (hidden_states), (attentions)
        则 all_hidden_states, all_attentions = model(input_ids)[-2:]

        二.
        如果没有设置：output_hidden_states和output_attentions
        输出的是：最后一层  --> (output_hidden_states, output_attentions)
       '''
        for index, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            last_hidden_state = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            print(len(last_hidden_state))
            all_hidden_states, all_attentions = last_hidden_state[-2:]  # 这里获取所有层的hidden_satates以及attentions
            print(all_hidden_states[-2].shape)  # 倒数第二层hidden_states的shape
            print(all_hidden_states[-2])# 倒数第二层 shape = (batch_size, seq_length, hidden_sates)