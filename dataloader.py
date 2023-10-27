import os
import re
import torch
import numpy as np
import pandas as pd
from glob import glob
from dipy.io.image import load_nifti
from torch.utils.data import Dataset, DataLoader

from monai.transforms import (
    Compose,
    Resize,
    NormalizeIntensity,
    RandAffine,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandAdjustContrast,
    RandScaleIntensity,
    RandFlip,
    RandRotate90,
    Rand3DElastic
)

class AFQ_Loader(Dataset):
    def __init__(self, path, csv_path):
        self.path = path
        self.sequences = ["SMT/*micro_fa*", "NODDI/*OD*", "NODDI/*ICVF*", "FWDTI/FA*", "FWDTI/RD*"]
        self.subjects = [i for i in os.listdir(self.path)]
        self.subjects.remove("sdmt.csv")

        self.df = pd.read_csv(csv_path).set_index("Unnamed: 0")
        
        self.transforms = Compose([
            Resize((112, 112, 112)),
            # Normalization (choose one based on your needs)
            NormalizeIntensity(),
            # ScaleIntensity(minv=0.0, maxv=1.0),
            # MinMaxScaleIntensity(mini=0.0, maxi=1.0, clip=True),

            # Spatial Augmentations
            #RandAffine(
            #    prob=0.5,
            #    translate_range=(50, 50, 50),
            #    rotate_range=(np.pi/36, np.pi/36, np.pi/36),
            #    scale_range=(0.1, 0.1, 0.1)
            #),
            #Rand3DElastic(
            #    prob=0.5,
            #    sigma_range=(5,7),
            #    magnitude_range=(1, 2)
            #),
            RandFlip(spatial_axis=[0, 1, 2], prob=0.5),
            RandRotate90(prob=0.5, max_k=3, spatial_axes=(0, 1)),

            # Intensity Augmentations
            #RandAdjustContrast(prob=0.5),
            RandScaleIntensity(factors=(0.8, 1.2), prob=0.5),
            #RandGaussianNoise(prob=0.5),
            #RandGaussianSmooth(prob=0.5)
        ])
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        img_shape = (5, 112, 112, 112)
        img_data = np.zeros(img_shape)
        
        subject_id = self.subjects[idx]
        print(subject_id)
        for i, sequence in enumerate(self.sequences):
            print(glob(os.path.join(self.path, subject_id, sequence)))
            file_path = glob(os.path.join(self.path, subject_id, sequence))[0]

            if os.path.exists(file_path):
                img_aux, _ = load_nifti(file_path)
                img_aux = self.transforms(
                    np.expand_dims(img_aux, axis=0)
                )
                img_data[i, ...] = img_aux[0, ...]
            else:
                print(f"File not found: {file_path}")

        if "FIS" in subject_id:  
            label = self.df[self.df.redcap_event_name == "followup1_arm_1"].loc[subject_id[:-3], "sdmt"]
        elif "REHEM" in subject_id:
            label = self.df[self.df.redcap_event_name == "baseline_arm_1"].loc[subject_id[:-3].lower(), "sdmt"]
        else:
            label = np.random.uniform(70, 80)
        #label = label[0] if label.size else 0
        label = label if str(label) != "nan" else 0 

        #return torch.from_numpy(img_data), self.label_dict[int(label)]
        return torch.from_numpy(img_data), label


