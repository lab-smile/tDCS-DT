#Load Standard Libraries
import os
import glob
import nibabel as nib
import numpy as np
import math

#Load Torch Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_transformer
import torchio as tio
import torch.nn.functional as F

#Load Monai Packages
from monai.networks.nets import SwinUNETR, EfficientNet
from monai.transforms import ScaleIntensityd, ToTensord, RandSpatialCropd, RandFlipd, RandRotate90d, RandShiftIntensityd, Compose, ThresholdIntensityd
from monai.data import Dataset
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

##########################################################################################################################

# Dataloading Tools

# Define Custom Data Usage
class CustomDataset(tio.data.SubjectsDataset):
    def __init__(self, root_dir, transform=None, subjectsIDS = 'file.npz'):
        self.root_dir = root_dir
        self.transform = transform
        self.subjectsIDS = subjectsIDS
        self.subjects = self.load_subjects(self.subjectsIDS)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        subject = self.subjects[index]

        if self.transform:
            subject = self.transform(subject)

        return {
            'input_data': subject['T1'], # Primary image data input for deep learning model
            'label': subject['label'],   # J current map for either F3F4 or C3Fp2 electrode montages
            #'seg': subject['seg'],       # Model Segmentation output
            'vector': subject['vector'], # Model input to indicate if J is from F3F4 [1,0] or C3Fp2 [0,1]
            'Name': subject['Name'],     # Extracted from image file. used to match data (this script) and save results (infer. script)
        }

    def load_subjects(self, subjectsIDS):
        subject_list = []

        # Manually specify the order of modalities so that the J files don't get mixed up
        modality_order = ['T1', 'J_F3F4', 'J_C3Fp2']#, 'seg11']
            
        subject_ids = np.load(subjectsIDS)
        #print(subject_ids)
        
        # Iterate over all subjects
        for subject_id in subject_ids:
            subject_images = {}
            #print(f"individual id: {subject_id}")

            for modality_name in modality_order:
                modality_path = os.path.join(self.root_dir, modality_name)
                # All data follows the following format within its subfolder
                image_path = os.path.join(modality_path, f'{subject_id}.nii')
                try:
                    # We will also skip subjects who have corrupted data
                    image_data = nib.load(image_path).get_fdata()
                except nib.filebasedimages.ImageFileError:
                    print(f"Cannot determine file type of {image_path}")
                    break
                    
                # Make 4D
                image_data = np.expand_dims(image_data, axis=0)  
                subject_images[modality_name] = image_data

            # If no errors: Create subjects for 'J_F3F4' and 'J_C3Fp2' - we create two subjects per one MRI
            subject_f3f4 = {
                'T1': subject_images['T1'],
                'label': subject_images['J_F3F4'],
                'vector': torch.tensor([[1], [0]], dtype=torch.float32),  # [1, 0] indicates 'J_F3F4'
                #'seg': subject_images['seg11'],
                'Name' : subject_id + '_F3F4',
            }
            subject_c3fp2 = {
                'T1': subject_images['T1'],
                'label': subject_images['J_C3Fp2'],
                'vector': torch.tensor([[0], [1]], dtype=torch.float32),  # [0, 1] indicates 'J_C3Fp2'
                #'seg': subject_images['seg11'],
                'Name' : subject_id + '_C3Fp2',
            }

            subject_list.append(subject_f3f4)
            subject_list.append(subject_c3fp2)
            #print("subject done")
        #print("all done")
        return subject_list

def create_dataloaders(batch_size, root_dir, train_subjects_path, val_subjects_path, x, y, z):
    # Define transformations 
    transform_train = Compose([
        ScaleIntensityd(minv=0, maxv=1, keys=['T1']),
        RandSpatialCropd(keys=['T1', 'label'], roi_size=(x, y, z), random_size=False),
        RandFlipd(keys=["T1", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["T1", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["T1", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["T1", "label"], prob=0.10, max_k=3),
        ToTensord(keys=['T1', 'vector', 'label'])
    ])

    transform_test = Compose([
        ScaleIntensityd(minv=0, maxv=1, keys=['T1']),
        RandSpatialCropd(keys=['T1', 'label'], roi_size=(x, y, z), random_size=False),
        ToTensord(keys=['T1', 'vector', 'label'])
    ])
    
    # Define datasets
    train_dataset = CustomDataset(root_dir=os.path.join(root_dir, 'Train'), transform=transform_train, subjectsIDS=train_subjects_path)
    val_dataset = CustomDataset(root_dir=os.path.join(root_dir, 'Val'), transform=transform_test, subjectsIDS=val_subjects_path)
    
    # Define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def create_test_dataloader(batch_size, root_dir, test_subjects_path, x, y, z):
    transform_test = Compose([
        ScaleIntensityd(minv=0, maxv=1, keys=['T1']),
        ThresholdIntensityd(keys=['label'], threshold=3.0, above=False, cval=3.0), 
        RandSpatialCropd(keys=['T1', 'label'], roi_size=(260, 311, 260), random_size=False),
        ToTensord(keys=['T1', 'vector', 'label']),#, 'seg']),
    ])

    test_dataset = CustomDataset(root_dir=os.path.join(root_dir, 'Test'), transform=transform_test, subjectsIDS=test_subjects_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

##########################################################################################################################

# Model

class MultiModel(nn.Module):
    def __init__(self, in_channels_img, in_channels_vector, out_channels,x,y,z):
        super(MultiModel, self).__init__()
        self.backbone = SwinUNETR(img_size=(x,y,z), in_channels=in_channels_img, out_channels=out_channels, spatial_dims=3)#, use_checkpoint=True)
        
        self.head = nn.Conv3d(out_channels + in_channels_vector, 1, kernel_size=1)

        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",    
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",   
            "r4_k5_s22_e6_i192_o320_se0.25",   
        ]
        self.montage = EfficientNet(blocks_args_str, num_classes=2, image_size=x, spatial_dims=3, in_channels=1)

    def forward(self, x_img, x_vector):
        backbone_output = self.backbone(x_img)
        
        # Ensure x_vector has the correct size: batch x 2 x 1
        x_vector = x_vector.unsqueeze(2).unsqueeze(3)
    
        # Concatenate along the channel dimension
        aa,bb,cc,dd,ee = backbone_output.shape #Batch x Channel x Length x Width x Height
        concatenated_input = torch.cat([backbone_output, x_vector.repeat(1, 1, cc, dd, ee)], dim=1)
        
        output = self.head(concatenated_input)
        
        output_class = self.montage(output)
        
        return output, output_class 
    
##########################################################################################################################

# Other

def reg_ece(predictions, targets, num_bins=10, device = 'cpu'):
    # Calculate absolute differences between predictions and targets
    errors = np.abs(predictions - targets)#.to(device)
    
    # Define confidence bins
    bin_boundaries = np.linspace(np.min(predictions), np.max(predictions), num_bins + 1)#.to(device)
    
    # Initialize arrays to store mean predictions and errors per bin
    mean_predictions = np.zeros(num_bins)#.to(device)
    mean_errors = np.zeros(num_bins)#.to(device)
    
    # Iterate over bins
    for i in range(num_bins):
        # Find indices where predictions fall into the current bin
        bin_indices = np.where((predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i+1]))#.to(device)
        
        # Calculate mean prediction and mean error for the current bin
        if len(bin_indices[0]) > 0:
            mean_predictions[i] = np.mean(predictions[bin_indices])#.to(device)
            mean_errors[i] = np.mean(errors[bin_indices])#.to(device)
    
    # Calculate weighted average of calibration errors
    weights = np.diff(bin_boundaries)[:-1]#).to(device)  # Width of each bin
    #mean_errors = mean_errors.cpu()
    mean_errors = mean_errors[:len(weights)]
    ece = torch.from_numpy(np.array(np.average(mean_errors, weights=weights))).to(device)
    
    return ece

def vector_class(vector, device='cpu'):
    # Create the desired meta tensor
    desired_meta_tensor = torch.zeros(vector.size(0), dtype=torch.long, device=device)

    # Set values to 0 where the metatensor has [[1.], [0.]] and 1 where the metatensor has [[0.], [1.]]
    desired_meta_tensor[(vector[:, 0, 0] == 1) & (vector[:, 1, 0] == 0)] = 0
    desired_meta_tensor[(vector[:, 0, 0] == 0) & (vector[:, 1, 0] == 1)] = 1

    return desired_meta_tensor