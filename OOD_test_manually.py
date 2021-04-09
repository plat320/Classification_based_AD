import os
import argparse


parser = argparse.ArgumentParser(description='Classifier training')
parser.add_argument('--num_classes', type=int, default=5, help='the # of classes')
parser.add_argument('--net_type', type=str, default="resnet", help='resnet | densenet')
parser.add_argument('--where', type=str, default="local", help='which machine')
parser.add_argument('--gpu', type=str, default=0, help='gpu index')
parser.add_argument('--mode', type=str, default="test", help='train | test')
parser.add_argument('--resume', default=True , help='load last checkpoint')
parser.add_argument('--board_clear', type=bool, default=False , help='clear tensorboard folder')
args = parser.parse_args()
print(args)


env = str(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = env
# if args.where == "server2":
#     image_dir = "/home/seonghun20/data/stanford_dog/top5"
# elif args.where == "server1":
#     image_dir = "/home/seonghun/anomaly/data/MVTec"
# elif args.where == "local":
#     image_dir = "/media/seonghun/data1/stanford_dog/top5"

if args.where == "server2":
    image_dir = "/home/seonghun20/data/animal_data"
elif args.where == "server1":
    image_dir = "/home/seonghun/anomaly/data/MVTec"
elif args.where == "local":
    image_dir = "/media/seonghun/data1/animal_data"

import torch
import torchvision
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import time
import math
import glob
import shutil
from torchvision import transforms
from torchvision.models import resnet34
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pylab import get_current_fig_manager
from pathlib import Path

import load_data
import models

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def main():
    start_epoch = 0

    save_model = "./pre_trained"
    tensorboard_dir = "./tensorboard/OOD"
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Hyper-parameters
    eps = 1e-8
    init_lr = 5e-4

    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    ### data config
    test_data = load_data.Dog_dataloader(image_dir = image_dir,
                                         num_class = args.num_classes,
                                         mode = "OOD")
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=2)


    ##### model, optimizer config
    if args.net_type == "resnet":
        model = models.resnet50(num_c=args.num_classes, pretrained=True)


    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma = 0.3)


    if args.resume == True:
        print("load checkpoint_last")
        checkpoint = torch.load(os.path.join(save_model, "resnet50.pth.tar"))

        ##### load model
        model.load_state_dict(checkpoint["model"])

    for i in range(args.num_classes):
        locals()["test_label{}".format(i)] = 0
    test_acc = 0
    MSP = torch.tensor([])
    with torch.no_grad():
        for i, (org_image, gt) in enumerate(test_loader):
            org_image = org_image.to(device)
            model = model.to(device).eval()
            gt = gt.type(torch.FloatTensor).to(device)
            #### forward path
            output = model(org_image)
            raw_image = unorm(org_image.squeeze(0)).cpu().detach()
            gt_label = torch.argmax(gt, dim=1).cpu().detach().tolist()
            output_label = torch.argmax(torch.sigmoid(output), dim=1).cpu().detach().tolist()
            for idx, label in enumerate(gt_label):
                if label == output_label[idx]:
                    locals()["test_label{}".format(label)] += 1
                MSP = torch.cat((MSP, (torch.softmax(output, dim=1).max().cpu()).unsqueeze(0)), dim=0)
                # print(torch.softmax(output, dim=1).max())
                # print("label : {}, predicted class : {}".format(label, output_label))
            test_acc += sum(torch.argmax(torch.sigmoid(output), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()



            # thismanager = get_current_fig_manager()
            # thismanager.window.SetPosition((500, 0))
            # plt.get_current_fig_manager().window.wm_geometry("+1000+100") # move the window
            # plt.imshow(raw_image.permute(1,2,0))
            # plt.show()
    thres_list = [0.501, 0.601, 0.701, 0.801, 0.901]
    print("total # of data : {}".format(test_data.num_image))
    for idx, thres in enumerate(thres_list):
        print(thres, end=" ")
        if idx == 0:
            print(torch.sum(MSP<thres))
        else:
            print(torch.sum(torch.mul((thres+0.1) >= MSP, thres < MSP)))



    print("test accuracy total : {:.4f}".format(test_acc/test_data.num_image))
    for num in range(args.num_classes):
        print("label{} : {:.4f}"
              .format(num, locals()["test_label{}".format(num)]/test_data.len_list[num])
              , end=" ")
    print("\n")
    time.sleep(0.001)

if __name__ == '__main__':
    main()
# python OOD_classification.py --batch_size 8 --num_epochs 400 --where server2 --gpu 2
