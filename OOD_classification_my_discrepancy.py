import os
import argparse


parser = argparse.ArgumentParser(description='Classifier training')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size for data loader')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N', help='# of training epoch')
parser.add_argument('--num_instances', type=int, default=2, metavar='N', help='# of minimum instances')
parser.add_argument('-l', '--init_lr', type=float, default=1e-5, metavar='N', help='initial learning rate')
parser.add_argument('-d', '--dataset', type=str, default="top5", metavar='N', help='animal | top5 | group2')
parser.add_argument('--OOD_num_classes', type=int, default=0, help='the # of OOD classes, if do not want training transfer, this value must be 0')
parser.add_argument('--load', default=False, action="store_true", help='load pre-trained model')
parser.add_argument('--num_classes', type=int, default=4, help='the # of classes')
parser.add_argument('--net_type', type=str, required=True, help='resnet | densenet')
parser.add_argument('--where', type=str, default="local", help='which machine')
parser.add_argument('--gpu', type=str, default=0, help='gpu index')
parser.add_argument('--nesterov', default=False, action="store_true", help='optimizer option nesterov')
parser.add_argument('--membership', default=False, action="store_true", help='membership loss enable')
parser.add_argument('--custom_sampler', default=False, action="store_true", help='use custom sampler')
parser.add_argument('--transfer', default=False, action="store_true", help='transfer method enable')
parser.add_argument('--soft_label', default=False, action="store_true", help='soft label enable')
parser.add_argument('--board_clear', default=False, action="store_true", help='clear tensorboard folder')
args = parser.parse_args()
print(args)


env = str(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = env
if args.where == "server2":
    if args.dataset == "animal":
        image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/animal_data"
    elif args.dataset == "top5":
        image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/stanford_dog/top5"
    elif args.dataset == "group2":
        image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/stanford_dog/group2/fitting"
    elif args.dataset == "caltech":
        image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/CALTECH256"
    elif args.dataset == "dog":
        image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/stanford_dog/original"
    elif args.dataset == "cifar10":
        image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/CIFAR10"
    elif args.dataset == "cifar100":
        image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/CIFAR100"
    OOD_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/COCO"
elif args.where == "server1":
    image_dir = "/home/seonghun/anomaly/data/MVTec"
elif args.where == "local":
    if args.dataset == "animal":
        image_dir = "/media/seonghun/data1/animal_data"
    elif args.dataset == "top5":
        image_dir = "/media/seonghun/data/stanford_dog/top5"
OOD_image_dir = "/home/seonghun20/data/OOD_data"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import glob
import shutil
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from pathlib import Path

import load_data
import models
from evaluate import evaluate







def Membership_loss(output, gt, num_classes):
    R_wrong = 0
    R_correct = 0
    gt_idx = torch.argmax(gt, dim=1)
    for batch_idx, which in enumerate(gt_idx):
        for idx in range(args.num_classes):
            output_sigmoid = torch.sigmoid(output)
            if which == idx:
                R_wrong += (1 - output_sigmoid[batch_idx][idx]) ** 2
            else:
                R_correct += output_sigmoid[batch_idx][idx] / (args.num_classes-1)
    return (R_wrong + R_correct) / output.shape[0]



def main():
    start_epoch = 0

    pretrained_model = os.path.join("./pre_trained", args.dataset, args.net_type + ".pth.tar")
    save_model = "./save_model_dis/pre_training"
    tensorboard_dir = "./tensorboard/OOD_dis/pre_training" + args.dataset
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Hyper-parameters
    eps = 1e-8

    ### data config
    train_dataset = load_data.Dog_metric_dataloader(image_dir = image_dir,
                                                    num_class = args.num_classes,
                                                    mode = "train",
                                                    soft_label=args.soft_label)
    if args.custom_sampler:
        MySampler = load_data.customSampler(train_dataset, args.batch_size, args.num_instances)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_sampler= MySampler,
                                                   num_workers=2)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

    test_dataset = load_data.Dog_dataloader(image_dir = image_dir,
                                         num_class = args.num_classes,
                                         mode = "test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=8,
                                              shuffle=False,
                                              num_workers=2)

    out_test_dataset = load_data.Dog_dataloader(image_dir=image_dir,
                                             num_class=args.num_classes,
                                             mode="OOD")
    out_test_loader = torch.utils.data.DataLoader(out_test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    if args.transfer:
        ### perfectly OOD data
        OOD_dataset = load_data.Dog_dataloader(image_dir = OOD_dir,
                                             num_class = args.OOD_num_classes,
                                             mode = "OOD")
        OOD_loader = torch.utils.data.DataLoader(OOD_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=2)



    ##### model, optimizer config
    if args.net_type == "resnet50":
        model = models.resnet50(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "resnet34":
        model = models.resnet34(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "vgg19":
        model = models.vgg19(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "vgg16":
        model = models.vgg16(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "vgg19_bn":
        model = models.vgg19_bn(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "vgg16_bn":
        model = models.vgg16_bn(num_c=args.num_classes, pretrained=True)

    if args.transfer:
        extra_fc = nn.Linear(2048, args.num_classes + args.OOD_num_classes)

    if args.load == True:
        print("loading model")
        checkpoint = torch.load(pretrained_model)

        ##### load model
        model.load_state_dict(checkpoint["model"])


    batch_num = len(train_loader) / args.batch_size if args.custom_sampler else len(train_loader)


    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           args.num_epochs * batch_num)

    #### loss config
    criterion = nn.BCEWithLogitsLoss()

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


    # Start training
    j=0
    best_score=0
    score = 0
    membership_loss = torch.tensor(0)
    transfer_loss = torch.tensor(0)
    for epoch in range(start_epoch, args.num_epochs):
        running_loss = 0
        running_membership_loss = 0
        running_transfer_loss = 0
        running_class_loss = 0
        train_acc = 0
        test_acc = 0
        stime = time.time()

        # for i, (train_data, OOD_data) in enumerate(zip(train_loader, OOD_loader)):
        for i, train_data in enumerate(train_loader):
            #### initialized
            org_image = train_data['input'] + 0.01 * torch.randn_like(train_data['input'])
            org_image = org_image.to(device)
            gt = train_data['label'].type(torch.FloatTensor).to(device)

            model = model.to(device).train()
            optimizer.zero_grad()



            #### forward path
            out1, out2 = model.pendis_forward(org_image)

            if args.membership:
                membership_loss = (Membership_loss(out2, gt, args.num_classes)
                                   + Membership_loss(out1, gt, args.num_classes)
                                   )
                running_membership_loss += membership_loss.item()

            if args.transfer:
                extra_fc = extra_fc.to(device).train()

                OOD_image = (OOD_data['input'] + 0.01 * torch.randn_like(OOD_data['input'])).to(device)
                OOD_gt = torch.cat((torch.zeros(args.batch_size, args.num_classes), OOD_data['label'].type(torch.FloatTensor))
                                   , dim=1).to(device)

                #### forward path
                _, feature = model.gen_forward(OOD_image)
                OOD_output = extra_fc(feature)
                transfer_loss = criterion(OOD_output, OOD_gt)
                running_transfer_loss += transfer_loss.item()


            #### calc loss
            class1_loss = criterion(out1, gt)
            class2_loss = criterion(out2, gt)
            class_loss = (class1_loss + class2_loss)

            total_loss = class_loss + membership_loss * 0.3 + transfer_loss


            #### calc accuracy
            train_acc += sum(torch.argmax(out1, dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()
            train_acc += sum(torch.argmax(out2, dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()


            total_loss.backward()
            optimizer.step()
            scheduler.step()

            running_class_loss += class_loss.item()
            running_loss += total_loss.item()


        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                org_image = test_data['input'].to(device)
                model = model.to(device).eval()
                gt = test_data['label'].type(torch.FloatTensor).to(device)

                #### forward path
                out1, out2 = model.pendis_forward(org_image)
                score_1 = nn.functional.softmax(out1, dim=1)
                score_2 = nn.functional.softmax(out2, dim=1)
                dist = torch.sum(torch.abs(score_1 - score_2), dim=1).reshape((org_image.shape[0], -1))
                if i == 0:
                    dists = dist
                    labels = torch.zeros((org_image.shape[0], ))
                else:
                    dists = torch.cat((dists, dist), dim=0)
                    labels = torch.cat((labels, torch.zeros((org_image.shape[0]))), dim=0)

                test_acc += sum(torch.argmax(torch.sigmoid(out1), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()
                test_acc += sum(torch.argmax(torch.sigmoid(out2), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()

            for i, out_org_data in enumerate(out_test_loader):
                out_org_image = out_org_data['input'].to(device)

                out1, out2 = model.pendis_forward(out_org_image)
                score_1 = nn.functional.softmax(out1, dim=1)
                score_2 = nn.functional.softmax(out2, dim=1)
                dist = torch.sum(torch.abs(score_1 - score_2), dim=1).reshape((out_org_image.shape[0], -1))

                dists = torch.cat((dists, dist), dim=0)
                labels = torch.cat((labels, torch.ones((out_org_image.shape[0]))), dim=0)

        roc = evaluate(labels.cpu(), dists.cpu(), metric='roc')
        print('Epoch{} AUROC: {:.3f}, test accuracy : {:.4f}'.format(epoch, roc, test_acc/test_dataset.num_image/2))





        print('Epoch [{}/{}], Step {}, total_loss = {:.4f}, class = {:.4f}, membership = {:.4f}, transfer = {:.4f}, exe time: {:.2f}, lr: {:.4f}*e-4'
                  .format(epoch, args.num_epochs, i+1,
                          running_loss/batch_num,
                          running_class_loss/batch_num,
                          running_membership_loss/batch_num,
                          running_transfer_loss/batch_num,
                          time.time() - stime,
                          scheduler.get_last_lr()[0] * 10 ** 4))
        print('exe time: {:.2f}, lr: {:.4f}*e-4'
                  .format(time.time() - stime,
                          scheduler.get_last_lr()[0] * 10 ** 4))

        print("train accuracy total : {:.4f}".format(train_acc/train_dataset.num_image/2))
        print("test accuracy total : {:.4f}".format(test_acc/test_dataset.num_image/2))



        summary.add_scalar('loss/total_loss', running_loss/batch_num, epoch)
        summary.add_scalar('loss/class_loss', running_class_loss/batch_num, epoch)
        summary.add_scalar('loss/membership_loss', running_membership_loss/batch_num, epoch)
        summary.add_scalar('acc/train_acc', train_acc/train_dataset.num_image/2, epoch)
        summary.add_scalar('acc/test_acc', test_acc/test_dataset.num_image/2, epoch)
        summary.add_scalar("learning_rate/lr", scheduler.get_last_lr()[0], epoch)
        time.sleep(0.001)
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'init_lr' : scheduler.get_last_lr()[0]
            }, os.path.join(save_model, env, args.net_type, 'checkpoint_last_pre.pth.tar'))

if __name__ == '__main__':
    main()
# python OOD_classification_my_discrepancy.py -d dog --num_classes 60 -l 5e-3 --net_type resnet50 --batch_size 64 --num_epochs 100 --where server2 --gpu 3