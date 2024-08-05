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

def inference(x,y,z):
    
    cropped = RandSpatialCrop(roi_size=(1,x,y,z), random_size=False)
    ssim = SSIMMetric(spatial_dims=3)
    
    #metric = DiceMetric()                             
    accuracy = 0
    with torch.no_grad():
        for batch in test_loader:
            input_data = batch['input_data'].to(device)
            vector = batch['vector'].to(device)
            label = batch['label'].to(device)                    

            names = batch['Name']  # Extract paths from the batch

            # Run the SlidingWindowInferer
            output = inferer(network=lambda input_data: (model(input_data, vector)[0]), inputs=input_data)

            output_class = model(cropped(input_data), vector)[1]
            vector_label = vector_class(vector, device)

            output_back = model(cropped(input_data), vector)[2]

            #print(output_class)
            #print(vector_label)

            output_class = torch.argmax(F.softmax(output_class, dim=1), dim=1)

            if output_class == vector_label:
                accuracy += 1

            mscr = mscr_loss(output, label, device=device)

            #print(output.shape)
            #print(label.shape)

            a, b, c, d, e = output.shape

            mae = torch.abs(output - label).mean() 
            total_mae += mae
            total_mscr += mscr
            num_samples += input_data.size(0)

            mse = (output - label)**2
            mse = mse.mean()
            total_mse += mse

            ssim_results = ssim(output,label)
            total_ssim += ssim_results

            if c==256:
                old_mae += mae
                old_mscr += mscr
                old_mse += mse
                old_ssim += ssim_results
                old_samples += input_data.size(0)
            else:
                young_mae += mae
                young_mscr += mscr
                young_mse += mse
                young_ssim += ssim_results
                young_samples += input_data.size(0)

            index = names[0]
            nib.save(nib.Nifti1Image(output.squeeze().cpu().numpy(), np.eye(4)), os.path.join(results_dir, f"{index}_output_J.nii.gz"))
            nib.save(nib.Nifti1Image(output_back.squeeze().cpu().numpy(), np.eye(4)), os.path.join(results_dir, f"{index}_logit_J.nii.gz"))
            print(f"Finished Running: {names}, MAE: {mae}, Output Class: {output_class}, True Class: {vector_label}, MSCR: {mscr}, SSIM: {ssim_results}", flush=True)

    accuracy = accuracy / num_samples * 100

    avg_mae = total_mae / num_samples
    avg_mscr = total_mscr / num_samples
    avg_mse = total_mse / num_samples
    avg_ssim = total_ssim / num_samples
    print(f"MAE: {avg_mae}, Accuracy: {accuracy}, MSCR: {avg_mscr}, MSE: {avg_mse}, SSIM: {avg_ssim}", flush=True)

    if old_samples>0:
        old_mae = old_mae / old_samples
        old_mscr = old_mscr / old_samples
        old_mse = old_mse / old_samples
        old_ssim = old_ssim / old_samples
        print(f"Old - MAE: {old_mae}, Accuracy: {accuracy}, MSCR: {old_mscr}, MSE: {old_mse}, SSIM: {old_ssim}", flush=True)

    if young_samples>0:
        young_mae = young_mae / young_samples
        young_mscr = young_mscr / young_samples
        young_mse = young_mse / young_samples
        young_ssim = young_ssim / young_samples
        print(f"Young - MAE: {young_mae}, Accuracy: {accuracy}, MSCR: {young_mscr}, MSE: {young_mse}, SSIM: {young_ssim}", flush=True)

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
    parser.add_argument('-save_path', type=str, default='best_model.pth', help='saved model')
    parser.add_argument('-results_path', type=str, default='best_model', help='saved model')
    
    args = parser.parse_args()
    
    print('Loading Data...')
    # Set subdirectory paths based on the root directory
    test_subjects_path = os.path.join(args.root_dir, 'Test_ACT', 'test.npy')

    # Get dataloaders
    test_loader = create_test_dataloader(args.batch_size, args.root_dir, test_subjects_path, args.x, args.y, args.z)

    print('Define Model...')
    # Validation Inferer
    inferer = SlidingWindowInferer(roi_size=(args.x,args.y,args.z),sw_batch_size=args.batch_size)#128,128,128), sw_batch_size=1)

    # Define Model
    model = MultiModel(args.in_channels_img, args.in_channels_vector, args.out_channels, args.x, args.y, args.z).to(device)
    saved_state_dict = torch.load(args.save_path)
    model.load_state_dict(saved_state_dict)

    # Specify the results directory
    results_dir = os.path.join('Results', args.results_path)
    os.makedirs(results_dir, exist_ok=True)

    # Use MONAI's SlidingWindowInferer
    inferer = SlidingWindowInferer(roi_size=(x,y,z), sw_batch_size=1)

    total_mae = 0.0
    num_samples = 0
    results = []
    total_mscr = 0.0   
    total_mse = 0.0
    total_ssim = 0.0

    old_mae = 0.0
    young_mae = 0.0

    old_mscr = 0.0
    young_mscr = 0.0

    old_mse = 0.0
    young_mse = 0.0

    old_ssim = 0.0
    young_ssim = 0.0

    old_samples = 0
    young_samples = 0

    inference(args.x,args.y,args.z)