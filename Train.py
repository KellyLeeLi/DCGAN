# -*- coding: utf-8 -*-
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from Generator import Generator
from Discriminator import Discriminator
import time
import datetime

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

start = time.time()

# CUDA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)
gpu_count = torch.cuda.device_count()
print(f'Have {gpu_count} gpu')


# Random seed
manualSeed = 7777
print('Random Seed:', manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Attributes
dataroot = r'House_Room_Dataset/'

batch_size = 16
image_size = 64
G_out_D_in = 3
G_in = 100
G_hidden = 64
D_hidden = 64

epochs = 300
lr = 0.0001
beta1 = 0.5

img_list = []
G_losses = []
D_losses = []

# Data
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

# Create the dataLoader
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Weights
def weights_init(m):
    classname = m.__class__.__name__
    print('classname:', classname)

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Train
def train():
    # Create the generator
    netG = Generator(G_in, G_hidden, G_out_D_in).to(device)
    netG.apply(weights_init)
    print(netG)

    # Create the discriminator
    netD = Discriminator(G_out_D_in, D_hidden).to(device)
    netD.apply(weights_init)
    print(netD)

    # Loss fuG_out_D_intion
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, G_in, 1, 1, device=device)

    real_label = 1.0
    fake_label = 0
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    iters = 0
    print('Start!')

    for epoch in range(epochs):
        for i, data in enumerate(dataLoader, 0):
            # Update D network
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = netD(real_cpu).view(-1)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, G_in, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)

            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update G network
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataLoader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataLoader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()

                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    torch.save(netD, 'netD.pkl')
    torch.save(netG, 'netG.pkl')

    return G_losses, D_losses

file_name = f'lr={lr}_bs={batch_size}_ep={epochs}'
now = datetime.datetime.now()
now = now.strftime('%m%d_%H.%M')
parent_dir = 'fig'
path = os.path.join(parent_dir, now)
if not os.path.exists(path):
    try:
        os.makedirs(path)
    except:
        raise OSError("Can't create destination directory (%s)!" % (path)) 

# Plot
def plotImage(G_losses, D_losses):
    print('Start to plot!!')
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{path}/Loss_{file_name}.png')
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataLoader))

    # Plot the real images
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig(f'{path}/Out_{file_name}.png')
    plt.show()
    
    
    plt.figure(figsize=(20, 20))
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig(f'{path}/Out2_{file_name}.png')
    plt.show()
    

train()
plotImage(G_losses, D_losses)

end = time.time()

print("----- Execution time: %f s -----" % (end - start))
