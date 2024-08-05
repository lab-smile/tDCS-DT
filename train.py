#Load Standard Libraries
import os
import glob
import nibabel as nib
import numpy as np
import math
import argparse

#Load Torch Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_transformer
import torchio as tio
import torch.nn.functional as F

#Load Monai Packages
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

#Load Custom 
from utils import create_dataloaders, MultiModel, vector_class, reg_ece

#Set Cuda Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(max_epoch, save_path_model):

    # Training loop
    IterationNum = 0
    for epoch in range(max_epoch): #args.num_epochs):
        
        torch.cuda.empty_cache()
        
        print("Training")
        # Training
        model.train()
        
        for batch in train_loader:
            
            # Input image
            input_data = batch['input_data'].to(device)

            # Label (Target) image
            label = batch['label'].to(device)

            # Vector to define electrode montage
            vector = batch['vector'].to(device)

            optimizer.zero_grad()

            output, output_class = model(input_data, vector)
            
            del input_data

            vector_label = vector_class(vector).to(device)#, device)
            
            del vector
            
            # Compute loss(es)
            loss1 = criterion1(output, label)
            loss2 = criterion2(output_class, vector_label)
            loss3 = reg_ece(output, label, device=device) #mscr_loss(output, label, device=device)
            loss = loss1 + .5*loss2 + .5*loss3 #loss2 + loss3 #+ .25*loss3

            print(f"Training Iteration {IterationNum}, Loss 1: {loss1}, Loss 2: {loss2}, Loss 3: {loss3}, Total: {loss}", flush=True)

            del label
            del vector_label
            del output
            del loss1
            del loss2
            del loss3
            torch.cuda.empty_cache()
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            #print(f"Training Iteration {IterationNum}, Loss 1: {loss1}, Loss 2: {loss2}, Loss 3: {loss3}, Total: {loss}", flush=True)
            IterationNum += 1

            torch.cuda.empty_cache()
            
        avg_mae = validation()
        
        # Save the model if it has the lowest MAE on the validation set
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_epoch = epoch
            torch.save(model.state_dict(), save_path_model)

        print(f"Epoch [{epoch+1}/{num_epochs}], MAE: {avg_mae}", flush=True)

    print(f"Best model achieved at Epoch {best_epoch+1}, Best MAE: {best_mae}", flush=True)
        
def validation():

    # Validation
    model.eval()
    total_mae = 0.0
    #total_dice = 0.0
    num_samples = 0
        
    with torch.no_grad():
        print("Validation")
        for batch in val_loader:

            #Currently doing it like this because inferer was only working on one image at once. I welcome help fixing this.
            for idx in range(batch['input_data'].size(0)):
                input_data = batch['input_data'][idx].unsqueeze(0).to(device)
                label = batch['label'][idx].unsqueeze(0).to(device)
                vector = batch['vector'][idx].unsqueeze(0).to(device)

                output = inferer(network=lambda input_data: (model(input_data, vector)[0]), inputs=input_data)

                mae = torch.abs(output - label).mean() # YY remove items() for using pytorch global reduce
                total_mae += mae
                num_samples += 1

                del input_data
                del label 
                del vector
                del output
                
                torch.cuda.empty_cache()

    # Calculate average MAE on the validation set
    avg_mae = total_mae / num_samples

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=int, default=256, help='horizontal slice size')
    parser.add_argument('-y', type=int, default=256, help='vertical slice size')
    parser.add_argument('-z', type=int, default=256, help='thickness size')
    parser.add_argument('-batch_size', type=int, default=1, help='thickness size')
    parser.add_argument('-root_dir', type=str, default='Data_3D', help='root data directory')
    parser.add_argument('-in_channels_img', type=int, default=1, help='number of image channels')
    parser.add_argument('-in_channels_vector', type=int, default=2, help='number of electrode montages')
    parser.add_argument('-out_channels', type=int, default=1, help='number of output image channels')
    parser.add_argument('-num_epochs', type=int, default=120, help='number of output image channels')
    parser.add_argument('-save_path', type=str, default='best_model.pth', help='save model')
    
    args = parser.parse_args()
    
    print('Setting Model Path for Saving...')
    os.makedirs('Models', exist_ok=True)
    save_path_model = os.path.join('Models', args.save_path)
    
    print('Loading Data...')
    # Set subdirectory paths based on the root directory
    train_subjects_path = os.path.join(args.root_dir, 'Train_ACT', 'train.npy')

    val_subjects_path = os.path.join(args.root_dir, 'Val_ACT', 'val.npy')

    # Get dataloaders
    train_loader, val_loader = create_dataloaders(args.batch_size, args.root_dir, train_subjects_path, val_subjects_path, args.x, args.y, args.z)

    print('Define Model...')
    # Validation Inferer
    inferer = SlidingWindowInferer(roi_size=(args.x,args.y,args.z),sw_batch_size=args.batch_size)#128,128,128), sw_batch_size=1)

    # Define Model
    model = MultiModel(args.in_channels_img, args.in_channels_vector, args.out_channels, args.x, args.y, args.z).to(device)
    
    criterion1 = nn.L1Loss() #nn.L1Loss()  # Using L1 Loss (MAE) because its been doing better than L2 Loss for this problem
    #criterion2 = DiceCELoss(to_onehot_y=True, softmax=True) 
    criterion2 = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)# 

    # Training parameters
    best_mae = float('inf')
    #best_dice = float('inf')
    best_epoch = -1
    
    torch.cuda.empty_cache()
    print('Begin Training...')
    train(args.num_epochs, save_path_model)


