# -*- coding:utf-8 -*-

import yaml
import sys
import torch
import torch.optim as optim
from lstm_crf.data_format import DataFormat
from lstm_crf.model import BiLSTMCRF
from lstm_crf.utils import f1_score, get_tags, format_result
from configs.base import config
from albert.model.tokenization_bert import BertTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NER(object):
    
    def __init__(self, exec_type="train"):
        self.load_config()
        self.__init_model(exec_type)

    def __init_model(self, exec_type):
        if exec_type == "train":
            self.train_data = DataFormat(batch_size=self.batch_size, max_length=self.max_legnth, data_type='train')
            self.dev_data = DataFormat(batch_size=16, max_length=self.max_legnth, data_type="dev")

            self.model = BiLSTMCRF(
                tag_map=self.train_data.tag_map,
                batch_size=self.batch_size,
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )
            self.restore_model()

        elif exec_type == "predict":
            self.model = BiLSTMCRF(
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )
            self.restore_model()

    def load_config(self):
        try:
            fopen = open("models/config.yml")
            config = yaml.load(fopen)
            fopen.close()
        except Exception as error:
            print("Load config failed, using default config {}".format(error))
            fopen = open("models/config.yml", "w")
            config = {
                "embedding_size": 768,
                "hidden_size": 128,
                "batch_size": 64,
                "max_length":128,
                "dropout":0.5,
                "model_path": "models/",
                "tasg": ["ORG", "PER", "LOC", 'T']
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.max_legnth = config.get('max_length')
        self.model_path = config.get("model_path")
        self.tags = config.get("tags")
        self.dropout = config.get("dropout")

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("self.model:{}".format(self.model))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    def train(self):
        self.model.to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # optimizer = optim.SGD(ner_model.parameters(), lr=0.01)
        total_size = self.train_manager.train_dataloader.__len__()
        for epoch in range(5):
            index = 0
            for batch in self.train_manager.train_dataloader:
                self.model.train()
                index += 1
                self.model.zero_grad()  # 与optimizer.zero_grad()作用一样
                batch = tuple(t.to(DEVICE) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_out_masks = batch

                bert_encode = self.model(b_input_ids, b_input_mask)
                loss = self.model.loss_fn(bert_encode=bert_encode, tags=b_labels, output_mask=b_out_masks)
                progress = ("█" * int(index * 25 / total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                    epoch, progress, index, total_size, loss.item()))
                loss.backward()
                optimizer.step()

                # predicts = self.model.predict(bert_encode, b_out_masks)
                # b_labels = b_labels.view(1, -1)
                # b_labels = b_labels[b_labels != -1]
                # self.model.acc_f1(predicts, b_labels)
                self.eva1_1()
                print("-" * 50)
        torch.save(self.model.state_dict(), self.model_path + 'params.pkl')

    def eva1_1(self):
        self.model.eval()
        count = 0
        y_predicts, y_labels = [], []
        eval_loss, eval_acc, eval_f1 = 0, 0, 0
        with torch.no_grad():
            for step, batch in enumerate(self.dev_manager.train_dataloader):
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, input_mask, label_ids, output_mask = batch
                bert_encode = self.model(input_ids, input_mask)
                eval_los = self.model.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
                eval_loss = eval_los + eval_loss
                count += 1
                predicts = self.model.predict(bert_encode, output_mask)
                y_predicts.append(predicts)

                label_ids = label_ids.view(1, -1)
                label_ids = label_ids[label_ids != -1]
                y_labels.append(label_ids)

            eval_predicted = torch.cat(y_predicts, dim=0)
            eval_labeled = torch.cat(y_labels, dim=0)
            self.model.acc_f1(eval_predicted, eval_labeled)
            self.model.class_report(eval_predicted, eval_labeled)

    def predict_1(self):
        self.model.eval()  # 取消batchnorm和dropout,用于评估阶段
        self.model.to(DEVICE)
        VOCAB = config['albert_vocab_path']  # your path for model and vocab
        tokenizer = BertTokenizer.from_pretrained(VOCAB)
        while True:
            with torch.no_grad():
                input_str = input("请输入文本: ")
                tokens = ' '.join(list(input_str))
                sentences = '[CLS] ' + tokens + ' [SEP]'
                tokenized_text = tokenizer.tokenize(sentences)  # 用tokenizer对句子分词
                input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
                input_mask = [1] * len(input_ids)
                output_mask = [1] * len(tokens)
                output_mask = [0] + output_mask + [0]
                input_ids_tensor = torch.LongTensor(input_ids).view(-1, 1)
                input_mask_tensor = torch.LongTensor(input_mask).reshape(-1, 1)
                output_mask_tensor = torch.LongTensor(output_mask).reshape(-1,1)
                input_ids_tensor = input_ids_tensor.to(DEVICE)
                input_mask_tensor = input_mask_tensor.to(DEVICE)
                output_mask_tensor = output_mask_tensor.to(DEVICE)
                bert_encode = self.model(input_ids_tensor, input_mask_tensor)
                paths = self.model.predict(bert_encode, output_mask_tensor)
                print('paths:{}'.format(paths))


    def eval_2(self):
        self.model.eval()
        with torch.no_grad():
            index = 0
            #org
            R_ORG = 0
            P_ORG = 0
            F1_ORG = 0
            #per
            R_PER = 0
            P_PER = 0
            F1_PER = 0
            #loc
            R_LOC = 0
            P_LOC = 0
            F1_LOC = 0
            #time
            R_T = 0
            P_T = 0
            F1_T = 0

            for batch in self.dev_data.train_dataloader:
                index += 1
                batch = tuple(t.to(DEVICE) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                if len(b_input_ids) < self.dev_data.batch_size:
                    break
                _,paths = self.model(b_input_ids, b_input_mask)
                print("\teval")
                for tag in self.tags:
                    recall, precision, f1 = f1_score(b_labels, paths, tag, self.model.tag_map)
                    if tag == 'ORG':
                        R_ORG += recall
                        P_ORG += precision
                        F1_ORG += f1
                    elif tag == 'PER':
                        R_PER += recall
                        P_PER += precision
                        F1_PER += f1
                    elif tag == 'LOC':
                        R_LOC += recall
                        P_LOC += precision
                        F1_LOC += f1
                    elif tag == 'T':
                        R_T += recall
                        P_T += precision
                        F1_T += f1
            print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format("ORG", R_ORG/index, P_ORG/index, F1_ORG/index))
            print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format("PER", R_PER/index, P_PER/index, F1_PER/index))
            print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format("LOC", R_LOC/index, P_LOC/index, F1_LOC/index))
            print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format("T", R_T/index, P_T/index, F1_T/index))

    '''
    注意：
        1.在模型中有BN层或者dropout层时，在训练阶段和测试阶段必须显式指定train()
            和eval()。
        2.一般来说，在验证或者是测试阶段，因为只是需要跑个前向传播(forward)就足够了，
            因此不需要保存变量的梯度。保存梯度是需要额外显存或者内存进行保存的，占用了空间，
            有时候还会在验证阶段导致OOM(Out Of Memory)错误，因此我们在验证和测试阶段，最好显式地取消掉模型变量的梯度。
            使用torch.no_grad()这个上下文管理器就可以了。
    '''
    def predict_2(self, input_str=""):
        self.model.eval() #取消batchnorm和dropout,用于评估阶段
        self.model.to(DEVICE)
        VOCAB = config['albert_vocab_path']  # your path for model and vocab
        tokenizer = BertTokenizer.from_pretrained(VOCAB)

        with torch.no_grad():
            if not input_str:
                input_str = input("input text: ")
            input_vec = ' '.join(list(input_str))
            tokenized_text = tokenizer.tokenize(input_vec)  # 用tokenizer对句子分词
            input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
            input_mask = [1] * len(input_ids)

            input_ids_tensor = torch.LongTensor(input_ids)
            input_mask_tensor = torch.LongTensor(input_mask)

            _, paths = self.model(input_ids_tensor, input_mask_tensor)
            print("score:{}".format(_))
            print('paths:{}'.format(paths))
            entities = []
            for tag in self.tags:
                tags = get_tags(paths[0], tag, self.model.tag_map)
                entities += format_result(tags, input_str, tag)
        return entities

if __name__ == "__main__":
    if sys.argv[1] == "train":
        ner = NER("train")
        ner.train()
    elif sys.argv[1] == "predict":
        ner = NER("predict")
        print(ner.predict_1())
