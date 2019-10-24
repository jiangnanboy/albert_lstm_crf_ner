# Albert+BI-LSTM+CRF的实体识别 Pytorch
### outline
![lstm_crf的模型结构](https://raw.githubusercontent.com/jiangnanboy/albert_lstm_crf_ner/master/pics/lstm_crf_layers.png)

**lstm_crf**

![albert_lstm的模型结构](https://raw.githubusercontent.com/jiangnanboy/albert_lstm_crf_ner/master/pics/albert_lstm.png)

**albert_embedding_lstm**

1.这里将每个句子split成一个个字token，将每个token映射成一个数字，再加入masks,然后输入给albert产生句子矩阵表示，比如一个batch=10，句子最大长度为126，加上首尾标志[CLS]和[SEP]，max_length=128,albert_base_zh模型输出的数据shape为(batch,max_length,hidden_states)=(10,128,768)。

2.利用albert产生的表示作为lstm的embedding层。

3.没有对albert进行fine-tune。

### train
setp 1: albert/tfmodel_2_pymodel.py

1.将tensorflow预训练模型转化为pytorch可用模型。

2.本程序使用[albert_base_zh(小模型体验版)](https://storage.googleapis.com/albert_zh/albert_base_zh.zip), 参数量12M, 层数12，大小为40M。

3.转为pytorch模型后放在albert/pretrain/pytorch目录下。

4.模型的参数见albert/configs/目录。


setp 2: edit **models/config.yml**

    embedding_size: 768
	hidden_size: 128
	model_path: models/
	batch_size: 64
	max_length: 128
	dropout: 0.5
	tags:
  		- ORG
  		- PER
  		- LOC
  		- T

step 3: train

    python main.py train
	训练数据来自人民日报的标注数据
### predict

    python main.py predict
    input text: [CLS]“刘老根大舞台”被文化部、国家旅游局联合评为首批“国家文化旅游重点项目”[SEP]

### REFERENCES
-  https://github.com/huggingface/transformers
-  https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
-  https://github.com/yanwii/ChinsesNER-pytorch
-  https://github.com/lonePatient/albert_pytorch
-  https://github.com/brightmart/albert_zh
-  https://createmomo.github.io/

