# -*- coding=utf-8 -*-
# @Time: 2023.05.21
# @Author: Zongyi Li
# train.py 



import os
import os.path as osp
import random

import logging
from ml_collections import ConfigDict
import numpy as np
import torch.optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from Load_Dataset import RandomGenerator,ValGenerator,ImageToImage2D
from trainer import train_one_epoch
import configs.Config as config
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, get_model


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)



##################################################################################
#=================================================================================
#          Main Loop: load model,
#=================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf= transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])

    post_fix = 'jpg' if config.task_name == 'KvasirSeg' else 'png'
    train_dataset = ImageToImage2D(config.train_dataset, train_tf,image_size=config.img_size, post_fix=post_fix)
    val_dataset = ImageToImage2D(config.val_dataset, val_tf,image_size=config.img_size, post_fix=post_fix)
        
        
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)

    lr = config.learning_rate
    logger.info(model_type)

    model = get_model(model_type)
    model = model.cuda()
    # if torch.cuda.device_count() > 1:
    #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    # model = nn.DataParallel(model, device_ids=[0])
    criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler =  None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)
        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                            optimizer, writer, epoch, lr_scheduler,model_type,logger)

        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch+1 > 5:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice,max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count,config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':

    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)

    logger.info("==== Settings ====")
    logger.info(f"Set random seed: {config.seed}")
    logger.info(f"Maximum training epoch: {config.epochs}, early stop tolerance: {config.early_stopping_patience}")
    logger.info(f"Dataset name {config.task_name}")
    logger.info(f"Initial learning rate: {config.learning_rate}, use consine lr scheduler: {config.cosineLR}")
    logger.info(f"training information saved at: {config.save_path}")

    model = main_loop(model_type=config.model_name, tensorboard=True)

