import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from data import CustomDataset
from data_fake import CustomDataset2
from data_real import CustomDataset3
from data_original import CustomDataset4
from data_fake_c import CustomDataset5
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=6, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
    
# Optimizers
discriminator = Discriminator()
generator = Generator()

optimizer_G = torch.optim.Adam(generator.parameters())
optimizer_D = torch.optim.Adam(discriminator.parameters())

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if cuda:
    generator.cuda()
    discriminator.cuda()

checkpoint_G = torch.load('generator_un.pth')
generator.load_state_dict(checkpoint_G['model_state_dict'])
optimizer_G.load_state_dict(checkpoint_G['optimizer_state_dict'])
generator.eval()

checkpoint_D = torch.load('discriminator_un.pth')
discriminator.load_state_dict(checkpoint_D['model_state_dict'])
optimizer_D.load_state_dict(checkpoint_D['optimizer_state_dict'])
discriminator.eval()

def sample_image(n_row):
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(-1, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, 'test.png', nrow=n_row, normalize=True)

sample_image(n_row=6)

# Generate Prediction Results

# dataloader = torch.utils.data.DataLoader(CustomDataset4('label.csv', 'images_n.csv'), batch_size = 999999, shuffle = True)

# for i, (imgs, labels) in enumerate(tqdm(dataloader)):
#     imgs = Variable(imgs.type(FloatTensor))
#     labels = Variable(labels.type(LongTensor))
#     result = discriminator(imgs, labels)
#     pd.DataFrame(result.cpu().detach().numpy(), columns=['result']).to_csv("classification_result_original.csv", index = False, header = None)
    
# dataloader = torch.utils.data.DataLoader(CustomDataset2('label.csv', 'images_n.csv'), batch_size = 999999, shuffle = True)

# for i, (imgs, labels) in enumerate(tqdm(dataloader)):
#     imgs = Variable(imgs.type(FloatTensor))
#     labels = Variable(labels.type(LongTensor))
#     result = discriminator(imgs, labels)
#     pd.DataFrame(result.cpu().detach().numpy(), columns=['result']).to_csv("classification_result_fake.csv", index = False, header = None)
    
# dataloader = torch.utils.data.DataLoader(CustomDataset3('label.csv', 'images_n.csv'), batch_size = 999999, shuffle = True)

# for i, (imgs, labels) in enumerate(tqdm(dataloader)):
#     imgs = Variable(imgs.type(FloatTensor))
#     labels = Variable(labels.type(LongTensor))
#     result = discriminator(imgs, labels)
#     pd.DataFrame(result.cpu().detach().numpy(), columns=['result']).to_csv("classification_result_real.csv", index = False, header = None)
    
# dataloader = torch.utils.data.DataLoader(CustomDataset5('label.csv', 'images_n.csv'), batch_size = 999999, shuffle = True)

# for i, (imgs, labels) in enumerate(tqdm(dataloader)):
#     imgs = Variable(imgs.type(FloatTensor))
#     labels = Variable(labels.type(LongTensor))
#     result = discriminator(imgs, labels)
#     pd.DataFrame(result.cpu().detach().numpy(), columns=['result']).to_csv("classification_result_fake_c.csv", index = False, header = None)


# Saving Sample Images from Peaks
    
# dataloader = torch.utils.data.DataLoader(CustomDataset4('label.csv', 'images_n.csv'), batch_size = 1, shuffle = True)

# for i, (imgs, labels) in enumerate(tqdm(dataloader)):
#     imgs = Variable(imgs.type(FloatTensor))
#     labels = Variable(labels.type(LongTensor))
#     result = discriminator(imgs, labels)
#     result_np = result.cpu().detach().numpy()
#     if (result_np>0.5326).all() & (result_np<0.533).all():
#         loc = 'peak_original1/'+str(i)+'.png'
#         save_image(imgs.data.reshape(1,1,128,128), loc, nrow=1, normalize=True)
#     if (result_np>0.711).all() & (result_np<0.7115).all():
#         loc = 'peak_original2/'+str(i)+'.png'
#         save_image(imgs.data.reshape(1,1,128,128), loc, nrow=1, normalize=True)
    
# dataloader = torch.utils.data.DataLoader(CustomDataset2('label.csv', 'images_n.csv'), batch_size = 1, shuffle = True)

# for i, (imgs, labels) in enumerate(tqdm(dataloader)):
#     imgs = Variable(imgs.type(FloatTensor))
#     labels = Variable(labels.type(LongTensor))
#     result = discriminator(imgs, labels)
#     result_np = result.cpu().detach().numpy()
#     if (result_np>0.1666).all() & (result_np<0.167).all():
#         loc = 'peak_fake/'+str(i)+'.png'
#         save_image(imgs.data.reshape(1,1,128,128), loc, nrow=1, normalize=True)

    
# dataloader = torch.utils.data.DataLoader(CustomDataset3('label.csv', 'images_n.csv'), batch_size = 1, shuffle = True)

# for i, (imgs, labels) in enumerate(tqdm(dataloader)):
#     imgs = Variable(imgs.type(FloatTensor))
#     labels = Variable(labels.type(LongTensor))
#     result = discriminator(imgs, labels)
#     result_np = result.cpu().detach().numpy()
#     if (result_np>0.5226).all() & (result_np<0.523).all():
#         loc = 'peak_real1/'+str(i)+'.png'
#         save_image(imgs.data.reshape(1,1,128,128), loc, nrow=1, normalize=True)
#     if (result_np>0.6995).all() & (result_np<0.7).all():
#         loc = 'peak_real2/'+str(i)+'.png'
#         save_image(imgs.data.reshape(1,1,128,128), loc, nrow=1, normalize=True)

dataloader = torch.utils.data.DataLoader(CustomDataset5('label.csv', 'images_n.csv'), batch_size = 1, shuffle = True)

for i, (imgs, labels) in enumerate(tqdm(dataloader)):
    imgs = Variable(imgs.type(FloatTensor))
    labels = Variable(labels.type(LongTensor))
    result = discriminator(imgs, labels)
    result_np = result.cpu().detach().numpy()
    if (result_np>0.115).all() & (result_np<0.117).all():
        loc = 'peak_fake_c/'+str(i)+'.png'
        save_image(imgs.data.reshape(1,1,128,128), loc, nrow=1, normalize=True)

