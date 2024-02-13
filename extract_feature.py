

import os
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EDDeconvf(nn.Module):
    def __init__(self, cin=3, cout=2, zdim=512, nf=64, activation=nn.Tanh):
        super(EDDeconvf, self).__init__()       
                
        networkE = [nn.Conv2d(cin, nf, kernel_size=4, stride=1, padding=2, bias=False)] 
        networkE += [
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),      # 64x64 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),    # 32x32 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),    # 16x16 -> 8x8
            nn.GroupNorm(16*8, nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, nf*16, kernel_size=4, stride=2, padding=1, bias=False),   # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*16, nf*16, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*16, zdim, kernel_size=1, stride=1, padding=0, bias=False)   # 1x1 -> 1x1
            ]

        networkD = [
            nn.ConvTranspose2d(zdim, nf*16, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*16, nf*16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*16, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*8, nf*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*8, nf*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),   # 8x8 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),   # 16x16 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),     # 32x32 -> 64x64
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True)]
      
        networkD += [nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)]  
        networkD += [nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]

        if activation is not None:
            networkD += [activation()]
        self.networkE = nn.Sequential(*networkE)
        self.networkD = nn.Sequential(*networkD)

    def forward(self, input):
        feature = self.networkE(input)
        return feature


def main(opts):

    net = EDDeconvf().to(device)
    cp = torch.load(opts.model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(cp['netF_state_dict'])
    total_parameters = sum(param.numel() for param in net.parameters())
    print('# total parameters:', total_parameters)

    features = []
    with torch.no_grad():
        for img in tqdm(sorted(os.listdir(opts.img_dir))):
            if img.split('.')[-1] not in ['jpg', 'bmp']:
                continue
            img = Image.open(os.path.join(opts.img_dir, img))
            img = transforms.Resize(64)(img)
            img = transforms.ToTensor()(img)

            img = img.to(device)
            img = img * 2. -1.
            feature = net(img.unsqueeze(0))
            features.append(feature)

    if os.path.isdir(opts.save_dir) is False:
        os.mkdir(opts.save_dir)

    features = torch.cat(features).cpu().numpy()
    np.save(os.path.join(opts.save_dir, 'feature.npy'), features)
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SsupAU')
    parser.add_argument('--model_path', type=str, default='./checkpoint.pth')
    parser.add_argument('--img_dir',    type=str, default='./image')
    parser.add_argument('--save_dir',   type=str, default='./feature')
    args = parser.parse_args()

    main(args)
    print('finish ...')