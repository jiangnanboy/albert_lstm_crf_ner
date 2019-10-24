# -*- coding: utf-8 -*-

from convert_albert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

import os
if __name__ == '__main__':
    tf_checkpoint_path='pretrain/tf/albert_base_zh.zip '
    bert_config_file='configs/albert_config_base.json '
    pytorch_dump_path='pretrain/pytorch/albert_base_zh/pytorch_model.bin '
    share_type='all'
    convert_tf_checkpoint_to_pytorch(tf_checkpoint_path,bert_config_file,share_type,pytorch_dump_path)