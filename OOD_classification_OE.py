import os
import argparse

#### python OOD_classification_OE.py --num_classes 4 -l 5e-3 -d top5 --net_type resnet50 --batch_size 32 --num_epochs 100 --where server2 --gpu 1
parser = argparse.ArgumentParser(description='Classifier training')
parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='batch size for data loader')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N', help='# of training epoch')
parser.add_argument('-l', '--init_lr', type=float, default=1e-5, metavar='N', help='initial learning rate')
parser.add_argument('-d', '--dataset', type=str, default="top5", metavar='N', help='animal | top5')
parser.add_argument('--load', type=bool, default=False , help='load pre-trained model')
parser.add_argument('--num_classes', type=int, default=4, help='the # of classes')
parser.add_argument('--net_type', type=str, required=True, help='resnet | densenet')
parser.add_argument('--where', type=str, default="local", help='which machine')
parser.add_argument('--gpu', type=str, default=0, help='gpu index')
parser.add_argument('--mode', type=str, default="train", help='train | test')
parser.add_argument('--board_clear', type=bool, default=False , help='clear tensorboard folder')
args = parser.parse_args()
print(args)


env = str(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = env
if args.where == "server2":
    if args.dataset == "animal":
        image_dir = "/home/seonghun20/data/animal_data"
    elif args.dataset == "top5":
        image_dir = "/home/seonghun20/data/stanford_dog/top5"
elif args.where == "server1":
    image_dir = "/home/seonghun/anomaly/data/MVTec"
elif args.where == "local":
    if args.dataset == "animal":
        image_dir = "/media/seonghun/data1/animal_data"
    elif args.dataset == "top5":
        image_dir = "/media/seonghun/data/stanford_dog/top5"
OOD_image_dir = "/home/seonghun20/data/OOD_data"

import torch
import torchvision
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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
from pathlib import Path

import load_data
import models


def main():
    start_epoch = 0

    pretrained_model = os.path.join("./pre_trained", args.dataset, args.net_type + ".pth.tar")
    save_model = "./save_model_animal_OE"
    tensorboard_dir = "./tensorboard/OOD_animal_OE"
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Hyper-parameters
    eps = 1e-8

    ### data config
    train_data = load_data.Dog_dataloader(image_dir = image_dir,
                                          num_class = args.num_classes,
                                          mode = "train")
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2)
    outlier_data = load_data.Dog_dataloader(image_dir = OOD_image_dir,
                                            num_class= 1,
                                            mode = "voc_data",
                                            repeat=10000
                                            )
    outlier_loader = torch.utils.data.DataLoader(outlier_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=2)

    test_data = load_data.Dog_dataloader(image_dir = image_dir,
                                         num_class = args.num_classes,
                                         mode = "test")
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=8,
                                              shuffle=False,
                                              num_workers=2)



    ##### model, optimizer config
    if args.net_type == "resnet50":
        model = models.resnet50(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "resnet34":
        model = models.resnet34(num_c=args.num_classes, pretrained=True)

    if args.load == True:
        print("loading model")
        checkpoint = torch.load(pretrained_model)

        ##### load model
        model.load_state_dict(checkpoint["model"])

    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           args.num_epochs * len(train_loader))



    #### loss config
    criterion = nn.BCEWithLogitsLoss()

    uniform = torch.ones((args.batch_size, args.num_classes))/args.num_classes
    uniform = uniform.to(device)

    #### create folder
    Path(os.path.join(save_model, env, args.net_type)).mkdir(exist_ok=True, parents=True)

    if args.board_clear == True:
        files = glob.glob(tensorboard_dir+"/*")
        for f in files:
            shutil.rmtree(f)
    i = 0
    while True:
        if Path(os.path.join(tensorboard_dir, str(i))).exists() == True:
            i += 1
        else:
            Path(os.path.join(tensorboard_dir, str(i))).mkdir(exist_ok=True, parents=True)
            break
    summary = SummaryWriter(os.path.join(tensorboard_dir, str(i)))


    test_acc = 0
    if args.load == True:
        with torch.no_grad():
            for i, (org_image, gt) in enumerate(test_loader):
                org_image = org_image.to(device)
                model = model.to(device).eval()
                gt = gt.type(torch.FloatTensor).to(device)

                #### forward path
                output = model(org_image)

                gt_label = torch.argmax(gt, dim=1).cpu().detach().tolist()
                output_label = torch.argmax(torch.sigmoid(output), dim=1).cpu().detach().tolist()

                test_acc += sum(torch.argmax(torch.sigmoid(output), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()

            print("initial model accuracy = {:.4f}".format(test_acc / test_data.num_image))
    # Start training
    j=0
    best_score=0
    score = 0
    for epoch in range(start_epoch, args.num_epochs):
        outlier_loader.dataset.offset = np.random.randint(outlier_data.num_image)
        for i in range(args.num_classes):
            locals()["train_label{}".format(i)] = 0
            locals()["test_label{}".format(i)] = 0
        total_class_loss = 0
        total_outlier_loss = 0
        train_acc = 0
        test_acc = 0
        stime = time.time()

        # for i, (org_image, gt) in enumerate(train_loader):
        for i, ((org_image, gt),(outlier_image, _)) in enumerate(zip(train_loader, outlier_loader)):
            #### initialized
            org_image += 0.01 * torch.randn_like(org_image)
            org_image = org_image.to(device)
            model = model.to(device).train()
            gt = gt.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()

            #### forward path
            output = model(org_image)

            #### calc loss
            class_loss = criterion(torch.sigmoid(output), gt)
            outlier_image += 0.01 * torch.randn_like(outlier_image)
            outlier_image = outlier_image.to(device)

            outlier_output = model(outlier_image)
            outlier_loss = criterion(outlier_output, uniform)

            loss = class_loss + 0.1 * outlier_loss
            #### calc accuracy
            train_acc += sum(torch.argmax(torch.sigmoid(output), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()

            gt_label = torch.argmax(gt, dim=1).cpu().detach().tolist()
            output_label = torch.argmax(torch.sigmoid(output), dim=1).cpu().detach().tolist()
            for idx, label in enumerate(gt_label):
                if label == output_label[idx]:
                    locals()["train_label{}".format(label)] += 1

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_class_loss += class_loss.item()
            total_outlier_loss += outlier_loss.item()


        with torch.no_grad():
            for i, (org_image, gt) in enumerate(test_loader):
                org_image = org_image.to(device)
                model = model.to(device).eval()
                gt = gt.type(torch.FloatTensor).to(device)

                #### forward path
                output = model(org_image)

                gt_label = torch.argmax(gt, dim=1).cpu().detach().tolist()
                output_label = torch.argmax(torch.sigmoid(output), dim=1).cpu().detach().tolist()
                for idx, label in enumerate(gt_label):
                    if label == output_label[idx]:
                        locals()["test_label{}".format(label)] += 1


                test_acc += sum(torch.argmax(torch.sigmoid(output), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()




        print('Epoch [{}/{}], Step {}, class_loss = {:.4f}, out_loss = {:.4f}, exe time: {:.2f}, lr: {:.4f}*e-4'
                  .format(epoch, args.num_epochs, i+1,
                          total_class_loss/len(train_loader),
                          total_outlier_loss/len(train_loader),
                          time.time() - stime,
                          scheduler.get_last_lr()[0] * 10 ** 4))

        print("train accuracy total : {:.4f}".format(train_acc/train_data.num_image))
        for num in range(args.num_classes):
            print("label{} : {:.4f}"
                  .format(num, locals()["train_label{}".format(num)]/train_data.len_list[num])
                  , end=" ")
        print()
        print("test accuracy total : {:.4f}".format(test_acc/test_data.num_image))
        for num in range(args.num_classes):
            print("label{} : {:.4f}"
                  .format(num, locals()["test_label{}".format(num)]/test_data.len_list[num])
                  , end=" ")
        print("\n")



        summary.add_scalar('loss/class_loss', total_class_loss/len(train_loader), epoch)
        summary.add_scalar('loss/outlier_loss', total_outlier_loss/len(train_loader), epoch)
        summary.add_scalar('acc/train_acc', train_acc/train_data.num_image, epoch)
        summary.add_scalar('acc/test_acc', test_acc/test_data.num_image, epoch)
        summary.add_scalar("learning_rate/lr", scheduler.get_last_lr()[0], epoch)
        time.sleep(0.001)
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'init_lr' : scheduler.get_last_lr()[0]
            }, os.path.join(save_model, env, args.net_type, 'checkpoint_last_OE.pth.tar'))

if __name__ == '__main__':
    main()
# python OOD_classification.py --batch_size 8 --num_epochs 400 --where server2 --gpu 2
