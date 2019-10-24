from pathlib import Path
import os
BASE_DIR = os.path.abspath(os.path.dirname(os.getcwd()))
BASE_DIR += '/albert'
config = {

    'data_dir': BASE_DIR + '/dataset/lcqmc',
    'log_dir': BASE_DIR + '/outputs/logs',
    'figure_dir': BASE_DIR + "/outputs/figure",
    'outputs': BASE_DIR + '/outputs',
    'checkpoint_dir': BASE_DIR + "/outputs/checkpoints",
    'result_dir': BASE_DIR + "/outputs/result",

    'bert_dir':BASE_DIR + '/pretrain/pytorch/albert_base_zh', #预训练模型
    'albert_config_path': BASE_DIR + '/configs/albert_config_base.json',#基础版的预训练模型
    'albert_vocab_path': BASE_DIR + '/configs/vocab.txt'#bert需要词表
}

if __name__ == '__main__':
    print(config['albert_config_path'])
    print('./configs/albert_config_base.json')
    print(os.path.exists('../configs/albert_config_base.json'))
    base_path = Path('.')
    print(os.path.exists(base_path / 'configs/albert_config_base.json'))