from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.transforms import transforms
from PIL import Image


class EELGrass(Dataset):
    def __init__(self, csv_path, imgs_path):

        self.csv_path = csv_path
        self.imgs_path = imgs_path
        self.df = pd.read_csv(csv_path)
        self.img_names = []
        self.covers = []
        self.images = []

        self.initial_transform = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ])

        for i in range(len(self.df)):
            img_name = self.df.iloc[i]['image_filename']
            if img_name.startswith('DSC'):
                img_name = img_name + '.jpeg'
            else:
                img_name = img_name + '.JPG'
            self.img_names.append(img_name)
            self.covers.append(self.df.iloc[i]['cover'])

        for filename in self.img_names:
            img_path = os.path.join(self.imgs_path, filename)
            img = Image.open(img_path)
            img = self.initial_transform(img)
            self.images.append(img)

    def __len__(self):
        return len(self.covers)

    def __getitem__(self, idx):
        return self.images[idx], self.covers[idx]


class EELGrass_Predict(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_names = []
        self.images = []
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ])
        for filename in os.listdir(self.img_path):
            self.img_names.append(filename)
            img = Image.open(os.path.join(self.img_path, filename))
            img = self.transform(img)
            self.images.append(img)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        return self.images[idx], self.img_names[idx]
