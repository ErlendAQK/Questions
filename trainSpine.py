# -*- coding: utf-8 -*-

import torch
import monai
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld, RandRotate90d, SqueezeDimd
from monai.inferers import sliding_window_inference

from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

# Inputdata
datadir = r'C:\Users\John\datasets\images'
segdir = r'C:\Users\John\datasets\images\labels\spine'
Nepochs = 50
pretrain_in ='fish.pth'
model_out = 'spine.pth'

def main(datadir, segdir, Nepochs, pretrain_in, model_out):
    # ====================== STARTER ======================
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # 1) --------- Leser inn bilder og labels ---------------------
    objekter = [fil.removesuffix('.nii.gz') for fil in os.listdir(segdir) if fil.endswith('.nii.gz')]
    bilder_inn = []
    for obj in objekter:
        path_img = os.path.join(datadir, obj + '.dcm.nii.gz')
        path_seg = os.path.join(segdir, obj + '.nii.gz')        
        if os.path.exists(path_img) & os.path.exists(path_seg):
            bilder_inn.append({'img': path_img, 'seg': path_seg})
        else:
            print(f"OPS! Fant ikke filene for '{obj}' - Hopper over.")
    
    # Split dataset into training and validation
    split_index = int(len(bilder_inn) * 0.8)
    train_data_dicts = bilder_inn[:split_index]
    val_data_dicts = bilder_inn[split_index:]
    
    # 2) --------- Transforms ---------------------
    train_transforms = Compose([
        LoadImaged(keys=["img", "seg"], reader='PydicomReader'),
        EnsureChannelFirstd(keys=["img", "seg"]),
        SqueezeDimd(keys=["img", "seg"], dim=-1),
        ScaleIntensityd(keys=["img"]),
        RandCropByPosNegLabeld(keys=["img", "seg"], label_key="seg", spatial_size=[256, 256], pos=1, neg=1, num_samples=2),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["img", "seg"], reader='PydicomReader'),
        EnsureChannelFirstd(keys=["img", "seg"]),
        SqueezeDimd(keys=["img", "seg"], dim=-1),
        ScaleIntensityd(keys=["img"]),
    ])
    
    train_ds = Dataset(data=train_data_dicts, transform=train_transforms)
    val_ds = Dataset(data=val_data_dicts, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    
    # 3) --------- Modell for ryggradssegmentering ---------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")         
        
    class SpineSegmentationModel2(nn.Module):
        def __init__(self):
            super(SpineSegmentationModel2, self).__init__()
            self.unet = UNet(
                spatial_dims=2,
                in_channels=1,  # 1-channel (grayscale input)
                out_channels=2,  # 2 classes (background, spine)
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ) 

        def forward(self, x):
            return self.unet(x)   
    
    #3a)------ --------- ---------
    model_spine = SpineSegmentationModel2().to(device) #Kjører load_state_dict internt
    model_spine.state_dict()
    model_spine.load_state_dict(torch.load(pretrain_in))


    # 4) --------- Loss function, optimizer og metrics ---------------
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_spine.parameters(), lr=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # 5) --------- Training loop ---------------
    val_interval = 2
    writer = SummaryWriter('runs_cascade') #(log_dir='runs/experiment_name')
    for epoch in range(Nepochs):
        print(f"Epoch {epoch+1}/{Nepochs}")
        
        model_spine.train()  # Set model to training mode
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["img"].to(device)
            labels = batch_data["seg"].to(device).long() #Konverterer til heltall
            
            optimizer.zero_grad()
            outputs = model_spine(inputs)
            
            # Compute loss
            loss = loss_function(outputs, labels.squeeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= step
        print(f"Training loss for epoch {epoch+1}: {epoch_loss}")
        writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
        
        # Validering
        if (epoch + 1) % val_interval == 0:
            model_spine.eval()
            with torch.no_grad():
                val_dice = 0
                val_steps = 0
                
                for val_data in val_loader:
                    val_steps += 1
                    val_inputs = val_data["img"].to(device)
                    val_labels = val_data["seg"].to(device)
                    val_outputs = sliding_window_inference(val_inputs, roi_size=(256, 256), sw_batch_size=1, predictor=model_spine)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                mean_dice = dice_metric.aggregate().item()
                dice_metric.reset()
                print(f"Validation Dice score for epoch {epoch+1}: {mean_dice}")
    
    # Lagre den trente modellen
    torch.save(model_spine.state_dict(), model_out)


    


if __name__ == '__main__':
    main(datadir, segdir, Nepochs, pretrain_in, model_out)