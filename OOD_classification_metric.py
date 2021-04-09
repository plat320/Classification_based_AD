import os
import argparse

### python OOD_classification_animal.py --num_classes 5 -l 5e-3 --multi_lr 50 100 --net_type resnet50 --batch_size 32 --num_epochs 400 --where server2 --gpu 1


parser = argparse.ArgumentParser(description='Classifier training')
parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='batch size for data loader')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N', help='# of training epoch')
parser.add_argument('--num_instances', type=int, default=2, metavar='N', help='# of minimum instances')
parser.add_argument('-l', '--init_lr', type=float, default=1e-5, metavar='N', help='initial learning rate')
parser.add_argument('-d', '--dataset', type=str, default="top5", metavar='N', help='animal | top5 | group2')
parser.add_argument('--num_classes', type=int, default=4, help='the # of classes')
parser.add_argument('--OOD_num_classes', type=int, default=4, help='the # of OOD classes')
parser.add_argument('--net_type', type=str, required=True, help='resnet | densenet')
parser.add_argument('--where', type=str, default="local", help='which machine')
parser.add_argument('--gpu', type=str, default=0, help='gpu index')
parser.add_argument('--mode', type=str, default="train", help='train | test')
parser.add_argument('--resume', default=False , help='load last checkpoint')
parser.add_argument('--board_clear', default=False, action="store_true", help='clear tensorboard folder')
parser.add_argument('--nesterov', default=False, action="store_true", help='optimizer option nesterov')
parser.add_argument('--metric', default=False, action="store_true", help='triplet loss enable')
parser.add_argument('--membership', default=False, action="store_true", help='membership loss enable')
parser.add_argument('--soft_label', default=False, action="store_true", help='soft label enable')
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


import torch
import copy
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import glob
import shutil
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from pathlib import Path
from collections import defaultdict

import load_data
import models
from ODIN_test import test_ODIN


class customSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source)
        assert batch_size % num_instances == 0, "batch_size cannot divided by num_instances"

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_gts_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        for idx in range(len(self.data_source)):
            self.index_dic[int(torch.argmax(self.data_source[idx]['label']))].extend([idx])

        self.gts = list(self.index_dic.keys())                      # all labels name

        self.length = 0
        for gt in self.gts:
            idxs = self.index_dic[gt]                               # return every indexes of gt
            num = len(idxs)                                         # # of indexes of gt
            if num < self.num_instances:
                num = self.num_instances                            # if num >= self.num_instances,
            self.length += num - num % self.num_instances           # self.length += overflow number


    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for gt in self.gts:
            idxs = copy.deepcopy(self.index_dic[gt])
            if len(idxs) < self.num_instances:                      # if idxs' # is under the num_instances
                idxs = np.random.choice(idxs, size = self.num_instances, replace = True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:           # if batch_idxs reach num_instances, save idx and gt
                    batch_idxs_dict[gt].append(batch_idxs)
                    batch_idxs = []

        avai_gts = copy.deepcopy(self.gts)
        selected_gts = copy.deepcopy(self.gts)
        final_idxs = []
        batch_idxs = []
        flag = 0                                                    # break flag
        while True:
            count = 0

            for gt in selected_gts:
                tmp_batch_idxs = batch_idxs_dict[gt].pop(0)
                batch_idxs.extend(tmp_batch_idxs)

            if len(batch_idxs) == self.batch_size:
                final_idxs.append(batch_idxs)
                batch_idxs = []

            for key, value in batch_idxs_dict.items():
                if isinstance(value, list):
                    count += len(value)
                    if len(value) == 0:
                        flag += 1
                if count < self.batch_size:
                    flag += 1
            if flag != 0:
                break


            iter_num = (self.batch_size - len(batch_idxs))//self.num_instances
            iter_num = len(self.gts) if iter_num > len(self.gts) else iter_num
            selected_gts = random.sample(avai_gts, iter_num)


        self.length = len(final_idxs)
        return iter(final_idxs)



    def __len__(self):
        return self.length




def main():
    start_epoch = 0

    if args.metric:
        save_model = "./save_model_" + args.dataset + "_metric"
        tensorboard_dir = "./tensorboard/OOD_" + args.dataset
    else:
        save_model = "./save_model_" + args.dataset
        tensorboard_dir = "./tensorboard/OOD_" + args.dataset

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Hyper-parameters
    eps = 1e-8


    ### data config
    train_dataset = load_data.Dog_metric_dataloader(image_dir = image_dir,
                                                    num_class = args.num_classes,
                                                    mode = "train",
                                                    soft_label=args.soft_label)
    MySampler = customSampler(train_dataset, args.batch_size, args.num_instances)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               # batch_size=args.batch_size,
                                               batch_sampler= MySampler,
                                               num_workers=2)

    test_dataset = load_data.Dog_dataloader(image_dir = image_dir,
                                         num_class = args.num_classes,
                                         mode = "test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=8,
                                              shuffle=False,
                                              num_workers=2)

    out_test_dataset = load_data.Dog_dataloader(image_dir = image_dir,
                                             num_class = args.num_classes,
                                             mode = "OOD")
    out_test_loader = torch.utils.data.DataLoader(out_test_dataset,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=2)



    ##### model, optimizer config
    if args.net_type == "resnet50":
        model = models.resnet50(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "resnet34":
        model = models.resnet34(num_c=args.num_classes, pretrained=True)


    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           args.num_epochs * len(train_loader)//50,
                                                           eta_min=args.init_lr/10)


    if args.resume == True:
        print("load checkpoint_last")
        checkpoint = torch.load(os.path.join(save_model, "checkpoint_last.pth.tar"))

        ##### load model
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        optimizer = optim.SGD(model.parameters(), lr = checkpoint["init_lr"])

    #### loss config
    criterion = nn.BCEWithLogitsLoss()
    triplet = torch.nn.TripletMarginLoss(margin=0.5, p=2)

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
    triplet_loss = torch.tensor(0)
    membership_loss = torch.tensor(0)
    for epoch in range(start_epoch, args.num_epochs):
        for i in range(args.num_classes):
            locals()["train_label{}".format(i)] = 0
            locals()["test_label{}".format(i)] = 0
        total_loss = 0
        triplet_running_loss = 0
        membership_running_loss = 0
        class_running_loss = 0
        train_acc = 0
        test_acc = 0
        stime = time.time()

        for i, train_data in enumerate(train_loader):
            #### initialized
            org_image = train_data['input'] + 0.01 * torch.randn_like(train_data['input'])
            org_image = org_image.to(device)
            model = model.to(device).train()
            gt = train_data['label'].type(torch.FloatTensor).to(device)
            optimizer.zero_grad()

            #### forward path
            output, output_list = model.feature_list(org_image)


            if args.metric:

                target_layer = output_list[-1]
                negative_list = []
                for batch_idx in range(args.batch_size):
                    gt_arg = gt.argmax(dim=1)
                    negative = (gt_arg != gt_arg[batch_idx])
                    if batch_idx == 0:
                        negative_tensor = target_layer[np.random.choice(np.where(negative.cpu().numpy() == True)[0], 1)[0]]
                        positive_tensor = target_layer[np.random.choice(np.delete(
                            np.where(~negative.cpu().numpy() == True)[0],np.where(np.where(~negative.cpu().numpy() == True)[0] == batch_idx)),
                            1)[0]]
                        negative_tensor = torch.unsqueeze(negative_tensor, dim=0)
                        positive_tensor = torch.unsqueeze(positive_tensor, dim=0)
                    else:
                        tmp_negative_tensor = target_layer[np.random.choice(np.where(negative.cpu().numpy() == True)[0], 1)[0]]
                        negative_tensor = torch.cat((negative_tensor, torch.unsqueeze(tmp_negative_tensor, dim=0)), dim=0)

                        tmp_positive_tensor =  target_layer[np.random.choice(np.delete(
                            np.where(~negative.cpu().numpy() == True)[0],np.where(np.where(~negative.cpu().numpy() == True)[0] == batch_idx)),
                            1)[0]]
                        positive_tensor = torch.cat((positive_tensor, torch.unsqueeze(tmp_positive_tensor, dim=0)), dim=0)

                triplet_loss = 0.5 * triplet(target_layer, positive_tensor, negative_tensor)


            if args.membership:
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
                membership_loss = (R_wrong + R_correct) / args.batch_size


            #### calc loss
            class_loss = criterion(output, gt)

            #### calc accuracy
            train_acc += sum(torch.argmax(torch.sigmoid(output), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()

            gt_label = torch.argmax(gt, dim=1).cpu().detach().tolist()
            output_label = torch.argmax(torch.sigmoid(output), dim=1).cpu().detach().tolist()
            for idx, label in enumerate(gt_label):
                if label == output_label[idx]:
                    locals()["train_label{}".format(label)] += 1

            total_backward_loss = class_loss + triplet_loss + membership_loss
            total_backward_loss.backward()
            optimizer.step()
            scheduler.step()

            class_running_loss += class_loss.item()
            triplet_running_loss += triplet_loss.item()
            membership_running_loss += membership_loss.item()
            total_loss += total_backward_loss.item()


        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                org_image = test_data['input'].to(device)
                model = model.to(device).eval()
                gt = test_data['label'].type(torch.FloatTensor).to(device)

                #### forward path
                output = model(org_image)

                gt_label = torch.argmax(gt, dim=1).cpu().detach().tolist()
                output_label = torch.argmax(torch.sigmoid(output), dim=1).cpu().detach().tolist()
                for idx, label in enumerate(gt_label):
                    if label == output_label[idx]:
                        locals()["test_label{}".format(label)] += 1


                test_acc += sum(torch.argmax(torch.sigmoid(output), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()





        print('Epoch [{}/{}], Step {}, class_loss = {:.4f}, membership_loss = {:.4f}, total_loss = {:.4f}, exe time: {:.2f}, lr: {:.4f}*e-4'
                  .format(epoch, args.num_epochs, i+1,
                          class_running_loss/len(train_loader),
                          membership_running_loss/len(train_loader),
                          total_loss/len(train_loader),
                          time.time() - stime,
                          scheduler.get_last_lr()[0] * 10 ** 4))

        print("train accuracy total : {:.4f}".format(train_acc/(len(MySampler)*args.batch_size)))
        # print("train accuracy total : {:.4f}".format(train_acc/train_dataset.num_image))
        for num in range(args.num_classes):
            print("label{} : {:.4f}"
                  .format(num, locals()["train_label{}".format(num)]/train_dataset.len_list[num])
                  , end=" ")
        print()
        print("test accuracy total : {:.4f}".format(test_acc/test_dataset.num_image))
        for num in range(args.num_classes):
            print("label{} : {:.4f}"
                  .format(num, locals()["test_label{}".format(num)]/test_dataset.len_list[num])
                  , end=" ")
        print("\n")

        if epoch % 10 == 9:
            best_TNR, best_AUROC = test_ODIN(model, test_loader, out_test_loader, args.net_type, args)
            summary.add_scalar('AD_acc/AUROC', best_AUROC, epoch)
            summary.add_scalar('AD_acc/TNR', best_TNR, epoch)


        summary.add_scalar('loss/loss', total_loss/len(train_loader), epoch)
        summary.add_scalar('loss/membership_loss', membership_running_loss/len(train_loader), epoch)
        summary.add_scalar('acc/train_acc', train_acc/train_dataset.num_image, epoch)
        summary.add_scalar('acc/test_acc', test_acc/test_dataset.num_image, epoch)
        summary.add_scalar("learning_rate/lr", scheduler.get_last_lr()[0], epoch)
        time.sleep(0.001)
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'init_lr' : scheduler.get_last_lr()[0]
            }, os.path.join(save_model, env, args.net_type, 'checkpoint_last.pth.tar'))
        scheduler.step()

if __name__ == '__main__':
    main()
# python OOD_classification_animal_metric.py -d animal --num_classes 5 --gpu 2 --batch_size 32 --net_type resnet50 --where server2 -l 5e-3
