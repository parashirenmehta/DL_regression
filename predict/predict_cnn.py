import pandas as pd

from models.cnn import CNNRegression
from datasets import EELGrass

import torch
from torchvision.transforms import transforms
from PIL import Image
from datasets import EELGrass_Predict
from torch.utils.data import DataLoader
from torch import nn
import os
from models.cnn import CNNRegression
import toml

config = toml.load(open('../configs/cnn_regression.toml'))


def create_masks(fold, config, threshold=-1):
    sigmoid = nn.Sigmoid()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # seed = config['seed']

    # torch.manual_seed(seed)  # PyTorch random seed for CPU
    # torch.cuda.manual_seed(seed)  # PyTorch random seed for GPU

    tensor_to_image = transforms.Compose([
        transforms.ToPILImage()
    ])

    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    dataset = EELGrass_Predict('C:/Users/paras/eelgrass/data_formatted/cover/cropped/')

    # batch_size = config['batch_size']
    batch_size = 1

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = CNNRegression().to(device)
    model.load_state_dict(torch.load(
        config['save_model_weights'] + 'CNNRegression/' + 'Fold' + str(
            fold) + '_weights.pth'))
    model.eval()
    model.to(device)
    df = pd.DataFrame(columns=['image_filename', 'cover'])
    for i, (images, filenames) in enumerate(test_loader):

        images = images.to(device)
        c = model(images)

        for j in range(c.shape[0]):
            cover = c[j].cpu().detach().numpy()
            filename = filenames[j]
            df = pd.concat([df, pd.DataFrame([[filename, cover]], columns=['image_filename', 'cover'])])
            # img.save(save_path + filenames[j])

    df.to_csv('C:/Users/paras/eelgrass/Predicted_masks/cropped_covers_cnn_regression.csv', index=False)


create_masks(7, config)
