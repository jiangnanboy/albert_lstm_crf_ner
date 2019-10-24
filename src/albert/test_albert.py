import torch
from configs.base import config
from model.modeling_albert import BertConfig, BertModel
from model.tokenization_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


import os

device = torch.device('cuda' if torch.cuda.is_available()  else "cpu")
MAX_LEN = 10
if __name__ == '__main__':
    bert_config = BertConfig.from_pretrained(str(config['albert_config_path']), share_type='all')
    base_path = os.getcwd()
    VOCAB = base_path + '/configs/vocab.txt'  # your path for model and vocab
    tokenizer = BertTokenizer.from_pretrained(VOCAB)

    # encoder text
    tag2idx={'[SOS]':101, '[EOS]':102, '[PAD]':0, 'B_LOC':1, 'I_LOC':2, 'O':3}
    sentences = ['我是中华人民共和国国民', '我爱祖国']
    tags = ['O O B_LOC I_LOC I_LOC I_LOC I_LOC I_LOC O O', 'O O O O']

    tokenized_text = [tokenizer.tokenize(sent) for sent in sentences]
    #利用pad_sequence对序列长度进行截断和padding
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_text], #没法一条一条处理，只能2-d的数据，即多于一条样本，但是如果全部加载到内存是不是会爆
                              maxlen=MAX_LEN-2,
                              truncating='post',
                              padding='post',
                              value=0)

    tag_ids = pad_sequences([[tag2idx.get(tok) for tok in tag.split()] for tag in tags],
                            maxlen=MAX_LEN-2,
                            padding="post",
                            truncating="post",
                            value=0)

    #bert中的句子前后需要加入[CLS]:101和[SEP]:102
    input_ids_cls_sep = []
    for input_id in input_ids:
        linelist = []
        linelist.append(101)
        flag = True
        for tag in input_id:
            if tag > 0:
                linelist.append(tag)
            elif tag == 0 and flag:
                linelist.append(102)
                linelist.append(tag)
                flag = False
            else:
                linelist.append(tag)
        if tag > 0:
            linelist.append(102)
        input_ids_cls_sep.append(linelist)

    tag_ids_cls_sep = []
    for tag_id in tag_ids:
        linelist = []
        linelist.append(101)
        flag = True
        for tag in tag_id:
            if tag > 0:
                linelist.append(tag)
            elif tag == 0 and flag:
                linelist.append(102)
                linelist.append(tag)
                flag = False
            else:
                linelist.append(tag)
        if tag > 0:
            linelist.append(102)
        tag_ids_cls_sep.append(linelist)

    attention_masks = [[int(tok > 0) for tok in line] for line in input_ids_cls_sep]

    print('---------------------------')
    print('input_ids:{}'.format(input_ids_cls_sep))
    print('tag_ids:{}'.format(tag_ids_cls_sep))
    print('attention_masks:{}'.format(attention_masks))


    # input_ids = torch.tensor([tokenizer.encode('我 是 中 华 人 民 共 和 国 国 民', add_special_tokens=True)]) #为True则句子首尾添加[CLS]和[SEP]
    # print('input_ids:{}, size:{}'.format(input_ids, len(input_ids)))
    # print('attention_masks:{}, size:{}'.format(attention_masks, len(attention_masks)))

    inputs_tensor = torch.tensor(input_ids_cls_sep)
    tags_tensor = torch.tensor(tag_ids_cls_sep)
    masks_tensor = torch.tensor(attention_masks)

    train_data = TensorDataset(inputs_tensor, tags_tensor, masks_tensor)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=2)

    model = BertModel.from_pretrained(config['bert_dir'],config=bert_config)
    model.to(device)
    model.eval()
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
            last_hidden_state = model(input_ids = b_input_ids,attention_mask = b_input_mask)
            print(len(last_hidden_state))
            all_hidden_states, all_attentions = last_hidden_state[-2:] #这里获取所有层的hidden_satates以及attentions
            print(all_hidden_states[-2].shape)#倒数第二层hidden_states的shape
            print(all_hidden_states[-2])
