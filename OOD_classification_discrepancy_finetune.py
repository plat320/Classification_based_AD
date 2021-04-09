import os
import argparse



parser = argparse.ArgumentParser(description='Classifier training')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size for data loader')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N', help='# of training epoch')
parser.add_argument('-l', '--init_lr', type=float, default=1e-5, metavar='N', help='initial learning rate')
parser.add_argument('-m', '--m', type=float, default=0.1, metavar='N', help='discrepancy loss magnitude')
parser.add_argument('-d', '--dataset', type=str, default="top5", metavar='N', help='animal | top5 | group2')
parser.add_argument('-p', '--pretrained_model', type=str, metavar='N', help='pretrained model directory default ./save_model_dis/pre_training/***/checkpoint.ckpt')
parser.add_argument('--load', default=True, action="store_false", help='load pre-trained model')
parser.add_argument('--num_classes', type=int, default=4, help='the # of classes')
parser.add_argument('--net_type', type=str, required=True, help='resnet50 | densenet')
parser.add_argument('--where', type=str, default="local", help='which machine')
parser.add_argument('--gpu', type=str, default=0, help='gpu index')
parser.add_argument('--nesterov', default=False, action="store_true", help='optimizer option nesterov')
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
import time
import glob
import shutil
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from pathlib import Path

import load_data
import models

from evaluate import evaluate


def DiscrepancyLoss(input_1, input_2, m = 1.2):
    soft_1 = nn.functional.softmax(input_1, dim=1)
    soft_2 = nn.functional.softmax(input_2, dim=1)
    entropy_1 = - soft_1 * nn.functional.log_softmax(input_1, dim=1)
    entropy_2 = - soft_2 * nn.functional.log_softmax(input_2, dim=1)
    entropy_1 = torch.sum(entropy_1, dim=1)
    entropy_2 = torch.sum(entropy_2, dim=1)

    loss = torch.nn.ReLU()(m - torch.mean(entropy_1 - entropy_2))
    return loss



def main():
    start_epoch = 0

    save_model = "./save_model_dis/fine"
    pretrained_model_dir = "./save_model_dis/pre_training"
    tensorboard_dir = "./tensorboard/OOD_dis/fine/" + args.dataset
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Hyper-parameters
    eps = 1e-8

    ### data config
    train_dataset = load_data.Dog_dataloader(image_dir = image_dir,
                                          num_class = args.num_classes,
                                          mode = "train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2)
    test_dataset = load_data.Dog_dataloader(image_dir = image_dir,
                                         num_class = args.num_classes,
                                         mode = "test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.num_classes,
                                              shuffle=True,
                                              num_workers=2)
    out_train_dataset = load_data.Dog_dataloader(image_dir=image_dir,
                                             num_class=args.num_classes,
                                             mode="OOD_val")
    out_train_loader = torch.utils.data.DataLoader(out_train_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=2)
    out_test_dataset = load_data.Dog_dataloader(image_dir=image_dir,
                                             num_class=args.num_classes,
                                             mode="OOD")
    out_test_loader = torch.utils.data.DataLoader(out_test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=2)




    ##### model, optimizer config
    if args.net_type == "resnet50":
        model = models.resnet50(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "resnet34":
        model = models.resnet34(num_c=args.num_classes, pretrained=True)

    if args.load == True:
        print("loading model")
        checkpoint = torch.load(os.path.join(pretrained_model_dir, args.pretrained_model, "checkpoint_last_pre.pth.tar"))

        ##### load model
        model.load_state_dict(checkpoint["model"])

    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           args.num_epochs * len(train_loader))

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
    for epoch in range(start_epoch, args.num_epochs):
        total_class_loss = 0
        total_dis_loss = 0
        train_acc = 0
        test_acc = 0
        stime = time.time()



        model.eval().to(device)
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                org_image = test_data['input'].to(device)
                gt = test_data['label'].type(torch.FloatTensor).to(device)

                out1, out2 = model.dis_forward(org_image)
                score_1 = nn.functional.softmax(out1, dim=1)
                score_2 = nn.functional.softmax(out2, dim=1)
                dist = torch.sum(torch.abs(score_1 - score_2), dim=1).reshape((org_image.shape[0], ))
                if i == 0:
                    dists = dist
                    labels = torch.zeros((org_image.shape[0],))
                else:
                    dists = torch.cat((dists, dist), dim=0)
                    labels = torch.cat((labels, torch.zeros((org_image.shape[0]))), dim=0)

                test_acc += sum(torch.argmax(torch.sigmoid(out1), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()
                test_acc += sum(torch.argmax(torch.sigmoid(out2), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()

            for i, out_org_data in enumerate(out_test_loader):
                out_org_image = out_org_data['input'].to(device)

                out1, out2 = model.dis_forward(out_org_image)
                score_1 = nn.functional.softmax(out1, dim=1)
                score_2 = nn.functional.softmax(out2, dim=1)
                dist = torch.sum(torch.abs(score_1 - score_2), dim=1).reshape((out_org_image.shape[0], -1))

                dists = torch.cat((dists, dist), dim=0)
                labels = torch.cat((labels, torch.ones((out_org_image.shape[0]))), dim=0)

        roc = evaluate(labels.cpu(), dists.cpu(), metric='roc')
        print('Epoch{} AUROC: {:.3f}, test accuracy : {:.4f}'.format(epoch, roc, test_acc/test_dataset.num_image/2))


        for i, (org_data, out_org_data) in enumerate(zip(train_loader, out_train_loader)):
            #### initialized
            org_image = org_data['input'].to(device)
            out_org_image = out_org_data['input'].to(device)
            model = model.to(device).train()
            gt = org_data['label'].type(torch.FloatTensor).to(device)
            optimizer.zero_grad()

            #### forward path
            out1, out2 = model.dis_forward(org_image)

            #### calc accuracy
            train_acc += sum(torch.argmax(out1, dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()
            train_acc += sum(torch.argmax(out2, dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()

            #### calc loss
            class1_loss = criterion(out1, gt)
            class2_loss = criterion(out2, gt)

            out1, out2 = model.dis_forward(out_org_image)
            dis_loss = DiscrepancyLoss(out1, out2, args.m)
            loss = class1_loss + class2_loss + dis_loss

            total_class_loss += class1_loss.item() + class2_loss.item()
            total_dis_loss += dis_loss.item()


            loss.backward()
            optimizer.step()
            scheduler.step()



        print('Epoch [{}/{}], Step {}, class_loss = {:.4f}, dis_loss = {:.4f}, exe time: {:.2f}, lr: {:.4f}*e-4'
                  .format(epoch, args.num_epochs, i+1,
                          total_class_loss/len(out_train_loader),
                          dis_loss/len(out_train_loader),
                          time.time() - stime,
                          scheduler.get_last_lr()[0] * 10 ** 4))




        summary.add_scalar('loss/class_loss', total_class_loss/len(train_loader), epoch)
        summary.add_scalar('loss/dis_loss', total_dis_loss/len(train_loader), epoch)
        summary.add_scalar('acc/roc', roc, epoch)
        summary.add_scalar("learning_rate/lr", scheduler.get_last_lr()[0], epoch)
        time.sleep(0.001)
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'init_lr' : scheduler.get_last_lr()[0]
            }, os.path.join(save_model, env, args.net_type, 'checkpoint_last_fine.pth.tar'))

if __name__ == '__main__':
    main()

# python OOD_classification_discrepancy_finetune.py -d animal --num_classes 5 --net_type resnet50 -p 1/resnet50 --gpu 1 --where server2