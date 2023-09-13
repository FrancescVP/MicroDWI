import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import nibabel as nib

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler

from dataloader import AFQ_Loader

import sys
sys.path.append("models")
from DenseNet3D import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from ResNet3D import ResNet50, ResNet101, ResNet152

import sys
sys.path.append("models")
from DenseNet3D import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from ResNet3D import ResNet50, ResNet101, ResNet152

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize your DataLoader here, just as you did before
    dataset_path = "/home/fran/DATA/AFQ"
    csv_path = os.path.join(dataset_path, "sdmt.csv")
    full_dataset = AFQ_Loader(dataset_path, csv_path)

    # Initialize k-Fold cross-validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print('\n--------------------------------')
        print(f'FOLD {fold}')
        print('--------------------------------\n')

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        trainloader = DataLoader(full_dataset, batch_size=2, sampler=train_subsampler)
        valloader = DataLoader(full_dataset, batch_size=1, sampler=val_subsampler)

        writer = SummaryWriter(f'runs/AFQ_experiment_fold_{fold}')

        model = ResNet50(num_classes=1, channels=2).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(100):
            model.train()
            epoch_loss = 0.0
            for i, (inputs, labels) in enumerate(tqdm(trainloader)):
                inputs, labels = inputs.float().to(device), labels.float().to(device)

                optimizer.zero_grad()

                outputs = model(inputs).squeeze(-1).unsqueeze(-1)
                loss = criterion(outputs, labels.unsqueeze(-1))

                epoch_loss += loss.item()

                writer.add_scalar('Training loss', epoch_loss / (i + 1), epoch * len(trainloader) + i)

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1} Training loss: {epoch_loss / (i+1):.3f}")

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(tqdm(valloader)):
                    inputs, labels = inputs.float().to(device), labels.float().to(device)

                    outputs = model(inputs).squeeze(-1).unsqueeze(-1)
                    loss = criterion(outputs, labels.unsqueeze(-1))

                    val_loss += loss.item()

            writer.add_scalar('Validation loss', val_loss / len(valloader), epoch)
            print(f"Epoch {epoch + 1} Validation loss: {val_loss / len(valloader):.3f}")

        writer.close()
