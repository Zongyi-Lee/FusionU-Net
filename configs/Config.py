# -*- coding: utf-8 -*-

import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True # whether use cosineLR or not
n_channels = 3
n_labels = 1
epochs = 100
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 1000
early_stopping_patience = 50

pretrain = False
task_name = 'GlaS'
# task_name = 'MoNuSeg'
# task_name = 'PanNuke'
learning_rate = 1e-3
batch_size = 4


model_name = 'FusionUNet'

task_type = 'Training'

################
# For PanNuke Dataset
img_root = 'datasets/pannuke/images'
ann_root = 'datasets/pannuke/annotations'
split_path = 'datasets/pannuke/splits'

###############

train_dataset = './datasets/'+ task_name + '/Train_Folder/'
val_dataset = './datasets/'+ task_name + '/Val_Folder/'
test_dataset = './datasets/'+ task_name + '/Test_Folder/'
session_name       = task_type + '_' + time.strftime('%m.%d_%H_%M')
save_path          = 'results/' + task_name + '/' + model_name + '/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'


##########################################################################
# configs for UCtransNet
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config




# used in testing phase, copy the session name in training phase
test_session = "Training_06.19_22_51"