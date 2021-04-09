import os
import argparse

### python OOD_classification_animal.py --num_classes 5 -l 5e-3 --multi_lr 50 100 --net_type resnet50 --batch_size 32 --num_epochs 400 --where server2 --gpu 1


parser = argparse.ArgumentParser(description='Classifier training')
parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='batch size for data loader')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N', help='# of training epoch')
parser.add_argument('--num_instances', type=int, default=2, metavar='N', help='# of minimum instances')
parser.add_argument('-l', '--init_lr', type=float, default=1e-5, metavar='N', help='initial learning rate')
parser.add_argument('-d', '--dataset', type=str, default="top5", metavar='N', help='animal | top5 | group2 | caltech | dog')
parser.add_argument('--num_classes', type=int, default=4, help='the # of classes')
parser.add_argument('--OOD_num_classes', type=int, default=0, help='the # of OOD classes, if do not want training transfer, this value must be 0')
parser.add_argument('--net_type', type=str, required=True, help='resnet34 | resnet50 | vgg16 | vgg16_bn | vgg19 | vgg19_bn')
parser.add_argument('--where', type=str, default="local", help='which machine')
parser.add_argument('--gpu', type=str, default=0, help='gpu index')
parser.add_argument('--mode', type=str, default="train", help='train | test')
parser.add_argument('--resume', default=False, action="store_true", help='load last checkpoint')
parser.add_argument('--board_clear', default=False, action="store_true", help='clear tensorboard folder')
parser.add_argument('--nesterov', default=False, action="store_true", help='optimizer option nesterov')
parser.add_argument('--metric', default=False, action="store_true", help='triplet loss enable')
parser.add_argument('--membership', default=False, action="store_true", help='membership loss enable')
parser.add_argument('--custom_sampler', default=False, action="store_true", help='use custom sampler')
parser.add_argument('--transfer', default=False, action="store_true", help='transfer method enable')
parser.add_argument('--soft_label', default=False, action="store_true", help='soft label enable')
parser.add_argument('--not_test_ODIN', default=True, action="store_false", help='if do not want test ODIN, check this option')
args = parser.parse_args()
print(args)


env = str(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = env

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import glob
import shutil
from tensorboardX import SummaryWriter
from pathlib import Path

from ODIN_test import test_ODIN
from utils import (
    test,
    model_config,
    data_config,
    image_dir_config,
    board_clear,
    tensorboard_idx,

    Membership_Loss,
    Transfer_Loss,
    Metric_Loss,
)

image_dir, OOD_dir = image_dir_config(args.where, args.dataset)



def main():
    start_epoch = 0

    if args.metric:
        save_model = "./save_model_" + args.dataset + "_metric"
        tensorboard_dir = "./tensorboard/OOD_" + args.dataset
    else:
        save_model = "./save_model_" + args.dataset
        tensorboard_dir = "./tensorboard/OOD_" + args.dataset

    #### create folder
    Path(os.path.join(save_model, env, args.net_type)).mkdir(exist_ok=True, parents=True)

    if args.board_clear: board_clear(tensorboard_dir)

    idx = tensorboard_idx(tensorboard_dir)
    summary = SummaryWriter(os.path.join(tensorboard_dir, str(idx)))


    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    eps = 1e-8

    ### data config
    resize = (160,160)
    train_dataset, train_loader, test_dataset, test_loader, out_test_dataset, out_test_loader, OOD_dataset, OOD_loader = data_config(
        image_dir, OOD_dir, args.num_classes, args.OOD_num_classes, args.batch_size,
        args.num_instances, args.soft_label, args.custom_sampler, args.not_test_ODIN, args.transfer, resize)


    ##### model, optimizer config
    model = model_config(args.net_type, args.num_classes, args.OOD_num_classes)

    #### batch_number
    batch_num = len(train_loader) / args.batch_size if args.custom_sampler else len(train_loader)

    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           args.num_epochs * batch_num,
                                                           eta_min=args.init_lr/10)


    if args.resume:
        print("load checkpoint_last")
        checkpoint = torch.load(os.path.join(save_model, env, args.net_type, 'checkpoint_last.pth.tar'))

        ##### load model
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        optimizer = optim.SGD(model.parameters(), lr = checkpoint["init_lr"])

    #### loss config
    criterion = nn.BCEWithLogitsLoss()
    triplet = torch.nn.TripletMarginLoss(margin=0.5, p=2)

    # Start training
    j=0
    best_score=0
    score = 0
    triplet_loss = torch.tensor(0)          # for error control
    membership_loss = torch.tensor(0)
    transfer_loss = torch.tensor(0)
    for epoch in range(start_epoch, args.num_epochs):
        OOD_data = 0            # for error control
        total_loss = 0
        triplet_running_loss = 0
        membership_running_loss = 0
        transfer_running_loss = 0
        class_running_loss = 0
        train_acc = 0
        test_acc = 0
        stime = time.time()

        for i, train_data in enumerate(train_loader):
        # for i, (train_data, OOD_data) in enumerate(zip(train_loader, OOD_loader)):
            #### initialized
            model = model.to(device).train()
            optimizer.zero_grad()

            org_image = train_data['input'] + 0.01 * torch.randn_like(train_data['input'])
            org_image = org_image.to(device)
            gt = train_data['label'].type(torch.FloatTensor).to(device)

            #### forward path
            output, output_list = model.feature_list(org_image)


            #### calc loss
            if args.transfer:
                transfer_loss = Transfer_Loss(model, OOD_data, criterion, args.num_classes, device)

            if args.metric:
                triplet_loss = Metric_Loss(output_list, gt, triplet)

            if args.membership:
                membership_loss = Membership_Loss(output, gt, args.num_classes)

            class_loss = criterion(output, gt)


            #### backpropagation
            total_backward_loss = class_loss + triplet_loss + membership_loss + transfer_loss
            total_backward_loss.backward()
            optimizer.step()
            scheduler.step()


            #### calc accuracy and running loss update
            train_acc += sum(torch.argmax(torch.sigmoid(output), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()

            class_running_loss += class_loss.item()
            triplet_running_loss += triplet_loss.item()
            membership_running_loss += membership_loss.item()
            transfer_running_loss += transfer_loss.item()
            total_loss += total_backward_loss.item()


        #### test_classification
        with torch.no_grad():
            test_label, test_acc = test(model, test_loader, args.num_classes, device)

        #### print status
        print('Epoch [{}/{}], Step {}, exe time: {:.2f}, lr: {:.4f}*e-4'
                  .format(epoch, args.num_epochs, i+1,
                          time.time() - stime,
                          scheduler.get_last_lr()[0] * 10 ** 4))

        print('class_loss = {:.4f}, membership_loss = {:.4f}, transfer_loss = {:.4f}, total_loss = {:.4f}'
                  .format(class_running_loss/batch_num,
                          membership_running_loss/batch_num,
                          transfer_running_loss/batch_num,
                          total_loss/batch_num))

        if args.dataset == 'caltech' or args.dataset == 'dog':
            print("train accuracy total : {:.4f}".format(train_acc/train_dataset.num_image))
        else:
            print("train accuracy total : {:.4f}".format(train_acc/(batch_num*args.batch_size)))
        print("test accuracy total : {:.4f}".format(test_acc/test_dataset.num_image))
        #### class-wise test accuracy
        for label in range(args.num_classes):
            print("label{}".format(label), end=" ")
            print("{:.4f}%".format(test_label[label] / test_dataset.len_list[label] * 100), end=" ")
        print()
        print()

        #### test ODIN
        if epoch % 10 == 9 and args.not_test_ODIN:
            best_TNR, best_AUROC = test_ODIN(model, test_loader, out_test_loader, args.net_type, args)
            summary.add_scalar('AD_acc/AUROC', best_AUROC, epoch)
            summary.add_scalar('AD_acc/TNR', best_TNR, epoch)


        #### update tensorboard
        summary.add_scalar('loss/loss', total_loss/batch_num, epoch)
        summary.add_scalar('loss/membership_loss', membership_running_loss/batch_num, epoch)
        summary.add_scalar('loss/transfer_loss', transfer_running_loss/batch_num, epoch)
        summary.add_scalar('acc/train_acc', train_acc/train_dataset.num_image, epoch)
        summary.add_scalar('acc/test_acc', test_acc/test_dataset.num_image, epoch)
        summary.add_scalar("learning_rate/lr", scheduler.get_last_lr()[0], epoch)
        time.sleep(0.001)

        #### save model
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'init_lr' : scheduler.get_last_lr()[0]
            }, os.path.join(save_model, env, args.net_type, 'checkpoint_last.pth.tar'))
        scheduler.step()

if __name__ == '__main__':
    main()

# python OOD_classification_transfer.py -d group2 --num_classes 4 --OOD_num_classes 200 --gpu 2 --batch_size 32 --net_type vgg19 --where server2 -l 5e-3 --membership --transfer
# python OOD_classification_transfer.py -d caltech --num_classes 128 --gpu 3 --batch_size 32 --net_type resnet50 --where server2 -l 5e-3 --not_test_ODIN