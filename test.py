import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
from scipy import misc
import imageio
from tqdm import tqdm

from model import Back_PVT
from data import test_dataset
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = 'xxxx/'

model = Back_PVT(channel=32)
model = torch.nn.DataParallel(model) # Use DataParallel to load model trained on multiple GPUs
model.load_state_dict(torch.load('xx.pth'))
# load_matched_state_dict(model, pretrained_dict)
model = model.module # Remove the DataParallel wrapper

model.cuda()
model.eval()

test_datasets = ['PMD']

for dataset in test_datasets:
    save_path = 'xxx/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    print('processing, please wait for a moment')
    for i in tqdm(range(test_loader.size)):
        image, HH,WW,name = test_loader.load_data()
        image = image.cuda()
        res0,res1,res= model(image)
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        imageio.imsave(save_path+name, res)
