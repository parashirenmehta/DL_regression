import matplotlib.pyplot as plt

from models.cnn import CNNRegression
from datasets import EELGrass
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Subset, DataLoader
import toml
from trainers.trainer import Trainer
from torch import nn
from torch import optim
import numpy as np
from torchvision.transforms import transforms

config = toml.load(open('../configs/cnn_regression.toml'))
dataset = EELGrass(csv_path=config['csv_path'],
                   imgs_path=config['imgs_path'])

tensor_to_img = transforms.ToPILImage()


kf = KFold(n_splits=config['num_folds'], shuffle=True, random_state=config['seed'])
validation_splits = []

for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
    print(f"Fold: {fold}")
    # print(f"Train indices: {train_indices}")
    # print(f"Val indices: {val_indices}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)


    # save_path = 'C:/Users/paras/eelgrass_new/Validation_Splits/'

    if fold == 1 or fold == 6 or fold == 9:

        predicted = []
        true = []

        model = CNNRegression().to(device)
        model.load_state_dict(torch.load('C:/Users/paras/eelgrass/model_weights/CNNRegression/Fold' + str(fold) + '_weights.pth'))
        model.eval()
        model.to(device)

        for i, (images, covers) in enumerate(valid_loader):

            images = images.to(device)
            covers = covers.to(device)
            covers = covers.to(torch.float32)
            c = model(images)
            c = c.reshape((c.shape[0]))
            predicted.append(c)
            true.append(covers)

        plt.scatter(true, predicted)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('Fold ' + str(fold))
        # plt.savefig('C:/Users/paras/eelgrass_new/Validation_Splits/Fold' + str(fold) + '.png')
        # plt.close()
        plt.show()



    # model = CNNRegression().to(device)
    # if config['criterion'] == "MSELoss":
    #     criterion = nn.MSELoss()
    # else:
    #     criterion = nn.L1Loss()
    #
    # if config['optimizer'] == "Adam":
    #     optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    # else:
    #     optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    #
    # # total_params = sum(p.numel() for p in model.parameters())
    # # print(f"Total number of parameters: {total_params}")
    #
    # trainer = Trainer(model, optimizer, criterion, train_loader, valid_loader,
    #                   lr=config['learning_rate'],
    #                   device=device)
    # trainer.train_and_evaluate(fold, config)
