import os
import re
import torch
import numpy as np
import pandas as pd
from glob import glob
from dipy.io.image import load_nifti
from torch.utils.data import Dataset, DataLoader

from monai.transforms import Compose, Resize, RandFlip, RandZoom, NormalizeIntensity

class AFQ_Loader(Dataset):
    def __init__(self, path, csv_path):
        self.path = path
        self.sequences = ["micro_fa", "tensor_fa"]
        self.label_dict = {
            0:0,
            1:1,
            2:1,
            3:1,
            4:2,
            5:2,
            6:3,
            7:3, 
            8:3,
            9:3
        }
        self.subjects = [
            re.search(r'sub-(FIS|REHHV)_\d+_\d+', i).group() for i in glob(os.path.join(self.path, "*{0}*".format(self.sequences[0]))) if "REHEM" not in i
        ] + [
            re.search(r'sub-REHEM\d+', i).group() for i in glob(os.path.join(self.path, "*{0}*".format(self.sequences[0]))) if "REHEM" in i
        ]

        self.df = pd.read_csv(csv_path).set_index("Unnamed: 0")

        #self.transform = Compose([
        #    Resize((112, 112, 112)),
        #    RandFlip(prob=0.2, spatial_axis=3),
        #    RandZoom(prob=0.1, min_zoom=0.85, max_zoom=1.15)
        #    ])
        self.transform = Compose([
            Resize((112, 112, 112)),
            NormalizeIntensity(),
            RandZoom(prob=0.1, min_zoom=0.85, max_zoom=1.15)
            ])
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        img_shape = (2, 112, 112, 112)
        img_data = np.zeros(img_shape)
        
        subject_id = self.subjects[idx]

        for i, sequence in enumerate(self.sequences):
            file_path = os.path.join(self.path, f"{subject_id}_model-{sequence}.nii.gz")

            if os.path.exists(file_path):
                img_aux, _ = load_nifti(file_path)
                img_aux = self.transform(
                    np.expand_dims(img_aux, axis=0)
                )
                img_data[i, ...] = img_aux[0, ...]
            else:
                print(f"File not found: {file_path}")

        if "FIS" in subject_id:  
            label = self.df[self.df.redcap_event_name == "followup1_arm_1"].loc[subject_id[4:-3], "sdmt"]
        elif "REHHV" in subject_id:
            label = np.random.uniform(70, 80)
        else:
            sub_name = "rehem_" + str(int(subject_id[-4:-2])) if int(subject_id[-4:-2]) > 9 else "rehem_0" + str(int(subject_id[-4:-2]))
            sub_name = sub_name if int(subject_id[-4:-2]) != 0 else "rehem_" + str(int(subject_id[-5:-2]))
            label = self.df[self.df.redcap_event_name == "baseline_arm_1"].loc[sub_name, "sdmt"]
        #label = label[0] if label.size else 0
        label = label if str(label) != "nan" else 0 

        #return torch.from_numpy(img_data), self.label_dict[int(label)]
        return torch.from_numpy(img_data), label


