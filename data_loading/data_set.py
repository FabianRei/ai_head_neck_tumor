from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import functional as F
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from nibabel import ecat

import torchvision.transforms as transforms
import torchio as tio

def get_default_tio_transform(mode='train'):
    if mode == 'train':
        transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 255), percentiles=(0.5, 99.5)),
            tio.CropOrPad((128, 128, 140)),
            tio.RandomAffine(scales=0.1, degrees=10),
            tio.RandomElasticDeformation()
            ])
    elif mode == 'valid':
        transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 255), percentiles=(0.5, 99.5)),
            tio.CropOrPad((128, 128, 140)),
            ])
    return transform

class EcatDFDataset(Dataset):
    """Create dataset from data frame"""

    def __init__(self, df, target_col='statEFS', mode='train', cv_split=0, classes=["event_free', 'event_occurred"], transform=None,
                 path_data_col='blue_data_path', img_columns=['CT_img_path', 'CT_roi_path', 'PT_img_path', 'PT_roi_path'], pil_image=False):
        self.mode = mode
        if mode == 'train':
            limited_df = df[df.cv_split != cv_split].reset_index(drop=True)
        elif mode == 'valid':
            limited_df = df[df.cv_split == cv_split].reset_index(drop=True)
        elif mode == 'full':
            limited_df = df
        self.df = limited_df
        self.targets = self.df[target_col]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.transform = transform

        self.data_path = Path(self.df[path_data_col].iloc[0])
        self.imgs = [self.df[column].values for column in img_columns]
        
        self.pil_image = pil_image

    def __len__(self):
        return len(self.df)


    def read_ecat(self, img_path):
        img = ecat.load(self.data_path/img_path)
        data = np.transpose(img.get_fdata(), [3, 0, 1, 2])
        return data
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = [self.read_ecat(img[idx]) for img in self.imgs]
        sample = np.concatenate(sample, axis=0)
        if self.pil_image:
            sample = Image.fromarray(sample)
        else:
            sample = torch.tensor(sample)
        
        target = int(self.targets[idx])
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample.float(), target
    
    
    
    
if __name__ == '__main__':
    import pandas as pd
    from pathlib import Path
    path_df = Path.cwd() /'data' / 'clinical_data_has_CT_PT_21feb23.feather'
    df = pd.read_feather(path_df)
    data_path = Path(df.iloc[0].blue_data_path)
    example = df.iloc[0].CT_img_path
    img = ecat.load(data_path/example)
    data4d = img.get_fdata().squeeze()
    dataset_train = EcatDFDataset(df, mode='train', cv_split=0)
    x, y = dataset_train[10]
    
    ########################### test transform
    
    transform = tio.Compose([
        tio.Clamp(0, 255),


        tio.RandomAffine(scales=0.1, degrees=10),
        tio.RandomElasticDeformation()
        ])
    # Define the data augmentation transforms


    # Apply the data augmentation transforms to the 3D CT scan
    augmented_image = transform(x) # .int()

    # Convert the PyTorch tensor back to a numpy array
    augmented_image = augmented_image.numpy()

    # Save the augmented image
    np.save('augmented_3d_ct_scan.npy', augmented_image)
    print('done')
    