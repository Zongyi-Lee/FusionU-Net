# -*- coding:utf-8 -*-
'''
    @Time: 2023.05.25
    @Author: Zongyi Li
    
    test.py
        This file implements running test on tranined model. Most code
        are borrowed from UCTransNet test_model.py but make some simp-
        lification
'''


import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import jaccard_score

import configs.Config as config
from Load_Dataset import ValGenerator, ImageToImage2D
from utils import get_model


def get_dice_iou(pred, label):
    """

    Args:
        preds (np.array): (h, w)
        labels (np.array): (h, w)
    """
    label = label.astype(np.float32)
    pred = pred.astype(np.float32)

    dice_coeff = 2 * np.sum(label * pred) / (np.sum(label) + np.sum(pred) + 1e-5)
    iou = jaccard_score(label.reshape(-1), pred.reshape(-1))

    return dice_coeff, iou

    
def eval_batch(model, sample_batch):
    """use batch to 

    Args:
        model (_type_): _description_
        sample_batch (_type_): _description_
    """
    img_batch, label_batch = sample_batch['image'], sample_batch['label']
    assert img_batch.shape[0] == 1, 'We implement for evaluate on single image, set batch size to 1 please.' 

    preds = model(img_batch.cuda())
    pred_class = torch.where(preds > 0.5, torch.ones_like(preds), torch.zeros_like(preds))
    pred_class = pred_class.squeeze()   # (h, w)

    pred_class = pred_class.cpu().data.numpy()
    labels = label_batch.squeeze().data.numpy()

    dice_coeff, iou = get_dice_iou(pred= pred_class, label=labels)

    return dice_coeff, iou
    

def test(model, test_loader, test_num):
    model.eval()
    dice_coeffs = []
    ious = []
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            dice_coeff, iou = eval_batch(model, sampled_batch) 
            dice_coeffs.append(dice_coeff)
            ious.append(iou)
            torch.cuda.empty_cache()
            pbar.update()
   
    dice_avg = sum(dice_coeffs) / len(dice_coeffs)
    iou_avg = sum(ious) / len(ious)
    return dice_avg, iou_avg

    
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test_session = config.test_session
    
    if config.task_name == 'GlaS':
        model_path = "results/GlaS/"+config.model_name + "/" + test_session + \
            "/models/best_model-" + config.model_name + ".pth.tar"
        test_num = 80
    
    elif config.task_name == 'MoNuSeg':
        test_num = 14
        model_path = "results/MoNuSeg/" + config.model_name + "/" + test_session + \
            "/models/best_model-" + config.model_name + ".pth.tar"
    
    model = get_model(config.model_name)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    
    print('Model loaded!')
    test_transform = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, test_transform, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1)

    dice_avg, iou_avg = test(model, test_loader, test_num)

    print(f'Average Dice: {dice_avg}')
    print(f'Average IOU: {iou_avg}')


if __name__ == '__main__':
    main()