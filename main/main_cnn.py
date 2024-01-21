from models.cnn import CNNRegression
from datasets import EELGrass
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Subset, DataLoader
import toml
from trainers.trainer import Trainer
from torch import nn
from torch import optim

config = toml.load(open('../configs/cnn_regression.toml'))
dataset = EELGrass(csv_path=config['csv_path'],
                   imgs_path=config['imgs_path'])



kf = KFold(n_splits=config['num_folds'], shuffle=True, random_state=config['seed'])

for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
    print(f"Fold: {fold}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

    model = CNNRegression().to(device)
    if config['criterion'] == "MSELoss":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    if config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")

    trainer = Trainer(model, optimizer, criterion, train_loader, valid_loader,
                      lr=config['learning_rate'],
                      device=device)
    trainer.train_and_evaluate(fold, config)
