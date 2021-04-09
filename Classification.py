import os
where = "local"
env = "0"
if where == "server2":
    img_dir = "/home/seonghun20/data/MVTec"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
elif where == "server1":
    img_dir = "/home/seonghun/anomaly/data/MVTec"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
elif where == "local":
    image_dir = "/media/seonghun/data1/Recording/road_condition/"


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
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import resnet50
from tensorboardX import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import load_data


def L1_loss(pred, target):
    f_pred = pred.contiguous().view(-1)
    f_target = target.contiguous().view(-1)
    l1loss = (torch.abs(f_pred - f_target + 1e-8).sum()) / f_pred.shape[0]
    return l1loss


def FMloss(pred, target):
    loss = torch.mean((pred-target)**2)/2
    return loss



if __name__ == '__main__':

    load = False
    start_epoch = 0

    condition_list = ["dry", "wet"]
    data_version = "capture"                # capture, extract_road


    sample_dir = "./output"
    save_model = "./save_model"
    tensorboard_dir = "./tensorboard"
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_lr = 5e-4

    # Hyper-parameters
    eps = 1e-8
    num_epochs = 50
    batch_size = 8

    ### data config
    train_data = load_data.Mobticon_dataloader(image_dir = image_dir,
                                               condition_list = condition_list,
                                               version = data_version,
                                               mode = "train")
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    test_data = load_data.Mobticon_dataloader(image_dir = image_dir,
                                              condition_list = condition_list,
                                              version = data_version,
                                              mode = "test")
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)


    ##### model, optimizer config
    model = resnet50(pretrained=True)

    model.fc = nn.Linear(2048, len(condition_list))
    # optimizer = optim.SGD(model.parameters(), lr=init_lr)
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma = 0.3)


    if load == True:
        print("load checkpoint {}".format(check_epoch))

        checkpoint = torch.load(os.path.join(save_model, fol, "checkpoint{}.pth.tar".format(check_epoch)))
        if hidden_size != checkpoint["hidden_size"]:
            raise AttributeError("checkpoint's hidden size is not same")


        ##### load model
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        optimizer = optim.Adam(model.parameters(), lr = checkpoint["init_lr"])


    #### loss config
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()


    Path(os.path.join(save_model, env)).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(sample_dir, env)).mkdir(exist_ok=True, parents=True)

    i = 0
    while True:
        if Path(os.path.join(tensorboard_dir, str(i))).exists() == True:
            i += 1
        else:
            Path(os.path.join(tensorboard_dir, str(i))).mkdir(exist_ok=True, parents=True)
            break

    summary = SummaryWriter(os.path.join(tensorboard_dir, str(i)))
    # Start training
    j=0
    best_score=0

    score = 0

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        train_acc = 0
        test_acc = 0
        label_one = 0
        label_zero = 0
        stime = time.time()

        for i, (org_image, gt) in enumerate(train_loader):

            org_image = org_image.to(device)
            model = model.to(device).train()
            gt = gt.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            #### forward path
            output = model(org_image)

            #### calc loss

            class_loss = criterion(output, gt)

            #### calc accuracy
            train_acc += sum(torch.argmax(torch.sigmoid(output), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()


            with autograd.detect_anomaly():
                class_loss.backward()
                optimizer.step()
                scheduler.step(epoch)

            total_loss += class_loss.item()

            if i%50 == 49:
                print("loss = {:.4f}, train accuracy = {:.4f}".format(
                    total_loss/i, train_acc/((i+1)*batch_size)
                ))

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
                    if label==0:
                        label_zero += 1
                    else:
                        label_one += 1


            test_acc += sum(torch.argmax(torch.sigmoid(output), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()




        print('Epoch [{}/{}], Step {}, exe time: {:.2f}, lr: {:.4f}*e-4'
                  .format(epoch, num_epochs, i+1, time.time() - stime, scheduler.get_lr()[0] * 10 ** 4))

        print("dry accuracy {:.4f}, wet accuracy {:.4f}".format(
            label_zero/151, label_one/166
        ))

        print('loss = {:.4f}, train accuracy = {:.4f}, test accuracy = {:.4f}'
              .format(total_loss/len(train_loader),
                      train_acc/train_data.num_image,
                      test_acc/test_data.num_image
                      ))

        summary.add_scalar('loss/loss', total_loss/len(train_loader), epoch)
        summary.add_scalar('acc/train_acc', train_acc/train_data.num_image, epoch)
        summary.add_scalar('acc/test_acc', test_acc/test_data.num_image, epoch)
        summary.add_scalar("learning_rate/lr", scheduler.get_lr()[0], epoch)
        time.sleep(0.001)
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'init_lr' : scheduler.get_lr()[0]
            }, os.path.join(save_model, env, 'checkpoint_last.pth.tar'))
        scheduler.step(epoch)

