import os
import argparse

### python OOD_classification.py --num_classes 4 -l 5e-3 --multi_lr 30 60 --net_type resnet50 --batch_size 32 --num_epochs 400 --where server2 --gpu 2

parser = argparse.ArgumentParser(description='Classifier training')
parser.add_argument('--net_type', type=str, required=True, help='vgg16 | vgg19')
parser.add_argument('-p', '--pre_trained_path', type=str, required=True, help='checkpoint path')
parser.add_argument('-s', '--seg_pre_trained_path', type=str, help='SegNet checkpoint path')
parser.add_argument('-d', '--dataset', type=str, default="top5", metavar='N', help='animal | top5 | group2 | caltech')
parser.add_argument('--g', type=int, default=1000, metavar='N', help='hyper-parameter')
parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='batch size for data loader')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N', help='# of training epoch')
parser.add_argument('-l', '--init_lr', type=float, default=1e-5, metavar='N', help='initial learning rate')
parser.add_argument('--num_classes', type=int, default=4, help='the # of classes')
parser.add_argument('--where', type=str, default="local", help='which machine')
parser.add_argument('--gpu', type=str, default=0, help='gpu index')
parser.add_argument('--mode', type=str, default="train", help='train | test')
parser.add_argument('--nesterov', default=False, action="store_true", help='optimizer option nesterov')
parser.add_argument('--board_clear', type=bool, default=False , help='clear tensorboard folder')
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn.functional as F
import torch.optim as optim
import time
import glob
import shutil
import random

from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from pathlib import Path

import load_data
import models
from evaluate import evaluate

class MSE:
    def __init__(self, g):
        self.g = g

    def loss(self, pred, target):
        none_zero = target * self.g

        result = torch.sum((pred * (target==0)) ** 2)
        result += self.g * torch.sum((pred * (target!=0) - target)**2)

        return result / (target.shape[0] * target.shape[1])


def test():
    tensorboard_dir = "./tensorboard/OOD_SegNet"

    ##### model, optimizer config
    if args.net_type == "vgg19_bn":
        model = models.vgg19_bn(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "vgg16_bn":
        model = models.vgg16_bn(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "resnet50":
        model = models.resnet50(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "resnet34":
        model = models.resnet34(num_c=args.num_classes, pretrained=True)

    SegNet_model = models.SegNet(num_classes=args.num_classes)

    ##### load pretrained model
    print("load checkpoint_last")
    checkpoint = torch.load(args.pre_trained_path)
    model.load_state_dict(checkpoint["model"])


    ##### load SegNet pretrained model
    print("load SegNet checkpoint_last")
    SegNet_model = models.SegNet(num_classes=args.num_classes)
    checkpoint2 = torch.load(args.seg_pre_trained_path)

    ##### load model
    SegNet_model.load_state_dict(checkpoint2["model"])


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
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=2)

    ### novelty data
    out_test_dataset = load_data.Dog_dataloader(image_dir = image_dir,
                                             num_class = args.num_classes,
                                             mode = "OOD")
    out_test_loader = torch.utils.data.DataLoader(out_test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=2)

    i = 0
    while True:
        if Path(os.path.join(tensorboard_dir, str(i))).exists() == True:
            i += 1
        else:
            Path(os.path.join(tensorboard_dir, str(i))).mkdir(exist_ok=True, parents=True)
            break
    summary = SummaryWriter(os.path.join(tensorboard_dir, str(i)))


    sliding_alpha = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    with torch.no_grad():
        stime = time.time()
        for i in range(args.num_classes):
            locals()["num"+str(i)] = 0
            locals()["mean" + str(i)] = torch.zeros((4096)).to(device)
        for train_data in train_loader:

            num = 0

            org_image = train_data['input'].to(device)
            gt = train_data['label'].to(device)
            model = model.to(device).eval()
            SegNet_model = SegNet_model.to(device).eval()

            target_image = torch.tensor([]).to(device)
            target_gt = torch.tensor([]).to(device)

            _, out_list = model.feature_list(org_image)
            feature_list = out_list[-1]
            if feature_list.shape[1] == 2048:
                feature_list = feature_list.permute(0, 2, 1, 3)
                feature_list = F.interpolate(feature_list, [4096, 1])
                feature_list = feature_list.permute(0, 2, 1, 3).squeeze(2).squeeze(2)

            ### calc mean value of classes
            for idx in range(org_image.shape[0]):
                locals()['num'+str(int(torch.argmax(gt[idx]).cpu()))] += 1
                locals()['mean'+str(int(torch.argmax(gt[idx]).cpu()))] += feature_list[idx]
                if bool(torch.isnan(locals()['mean'+str(int(torch.argmax(gt[idx]).cpu()))]).any()):
                    print("nan")

        for i in range(args.num_classes):
            locals()["mean" + str(i)] = locals()["mean" + str(i)] / locals()['num'+str(i)]

        print("pre-processing time {:.4f}".format(time.time()-stime))

        ### test OOD
        for alpha in sliding_alpha:
            OOD_final_output = torch.tensor([]).to(device)
            for OOD_data in out_test_loader:
                for i in range(args.num_classes):
                    locals()["image_set" + str(i)] = torch.tensor([]).to(device)
                org_image = OOD_data['input'].to(device)
                gt = OOD_data['label'].to(device)

                target_image = torch.tensor([]).to(device)
                target_gt = torch.tensor([]).to(device)

                _, out_list = model.feature_list(org_image)
                feature_list = out_list[-1]
                if feature_list.shape[1] == 2048:
                    feature_list = feature_list.permute(0, 2, 1, 3)
                    feature_list = F.interpolate(feature_list, [4096, 1])
                    feature_list = feature_list.permute(0, 2, 1, 3).squeeze(2).squeeze(2)

                ### batch rearrangement
                for idx in range(org_image.shape[0]):
                    tmp_image = feature_list[idx]
                    for i in range(args.num_classes):
                        inter_image = (1 - alpha) * tmp_image + alpha * locals()['mean'+str(i)]
                        temp_image = torch.unsqueeze(torch.cat((torch.unsqueeze(tmp_image, 0),
                                                                                  torch.unsqueeze(locals()['mean'+str(i)], 0),
                                                                                  torch.unsqueeze(inter_image, 0)),
                                                                                 dim=0),
                                                                       dim=0)

                        ### query / mean / inter set for all mean feature
                        locals()['image_set'+str(i)] = torch.cat((locals()['image_set'+str(i)], temp_image), dim=0)

                ### forward
                output=torch.tensor([]).to(device)
                for i in range(args.num_classes):
                    ### B x class for mean i
                    SegNet_output = SegNet_model(locals()['image_set'+str(i)])
                    output = torch.cat((output, torch.unsqueeze(SegNet_output, 0)), dim=0)
                OOD_final_output = torch.cat((OOD_final_output, torch.mean(output.max(2)[0], dim=0)), dim=0)



            ### test OOD
            inlier_final_output = torch.tensor([]).to(device)
            for inlier_data in test_loader:
                for i in range(args.num_classes):
                    locals()["image_set" + str(i)] = torch.tensor([]).to(device)
                org_image = inlier_data['input'].to(device)
                gt = inlier_data['label'].to(device)

                target_image = torch.tensor([]).to(device)
                target_gt = torch.tensor([]).to(device)

                _, out_list = model.feature_list(org_image)
                feature_list = out_list[-1]
                if feature_list.shape[1] == 2048:
                    feature_list = feature_list.permute(0, 2, 1, 3)
                    feature_list = F.interpolate(feature_list, [4096, 1])
                    feature_list = feature_list.permute(0, 2, 1, 3).squeeze(2).squeeze(2)

                ### batch rearrangement
                for idx in range(org_image.shape[0]):
                    tmp_image = feature_list[idx]
                    for i in range(args.num_classes):
                        inter_image = (1 - alpha) * tmp_image + alpha * locals()['mean'+str(i)]
                        temp_image = torch.unsqueeze(torch.cat((torch.unsqueeze(tmp_image, 0),
                                                                                  torch.unsqueeze(locals()['mean'+str(i)], 0),
                                                                                  torch.unsqueeze(inter_image, 0)),
                                                                                 dim=0),
                                                     dim=0)

                        ### query / mean / inter set for all mean feature
                        locals()['image_set'+str(i)] = torch.cat((locals()['image_set'+str(i)], temp_image), dim=0)

                ### forward
                output=torch.tensor([]).to(device)
                for i in range(args.num_classes):
                    ### B x class for mean i
                    output = torch.cat((output, torch.unsqueeze(SegNet_model(locals()['image_set'+str(i)]), 0)), dim=0)
                inlier_final_output = torch.cat((inlier_final_output, torch.mean(output.max(2)[0], dim=0)), dim=0)

            OOD_final_gt = torch.zeros_like(OOD_final_output)
            inlier_final_gt = torch.ones_like(inlier_final_output)

            final_scores = torch.cat((OOD_final_output, inlier_final_output), 0)
            final_gt = torch.cat((OOD_final_gt, inlier_final_gt), 0)

            roc = evaluate(final_gt, final_scores)
            summary.add_scalar('AUROC/AUROC', roc, int(alpha*100))
            summary.flush()

            print("alpha = {:.2f}, AUROC = {:.4f}".format(alpha, roc))








def train():
    start_epoch = 0

    save_model = "./save_model/SegNet"
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




    ##### model, optimizer config
    if args.net_type == "vgg19":
        model = models.vgg19(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "vgg16":
        model = models.vgg16(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "vgg19_bn":
        model = models.vgg19_bn(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "vgg16_bn":
        model = models.vgg16_bn(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "resnet50":
        model = models.resnet50(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "resnet34":
        model = models.resnet34(num_c=args.num_classes, pretrained=True)
    SegNet_model = models.SegNet(num_classes=args.num_classes)


    ##### load pretrained model
    print("load checkpoint_last")
    checkpoint = torch.load(args.pre_trained_path)

    ##### load model
    model.load_state_dict(checkpoint["model"])


    # optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=1e-5)
    optimizer = optim.SGD(SegNet_model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           args.num_epochs * len(train_loader),
                                                           )


    #### loss config
    MSE_loss = MSE(args.g)

    #### create folder
    Path(os.path.join(save_model, env, args.net_type)).mkdir(exist_ok=True, parents=True)



    # Start training
    j=0
    best_score=0
    score = 0

    for epoch in range(start_epoch, args.num_epochs):
        total_loss = 0
        train_acc = 0
        test_acc = 0
        stime = time.time()

        for i, train_data in enumerate(train_loader):
            #### initialized
            org_image = train_data['input'].to(device)
            if org_image.shape[0] !=args.batch_size:
                continue

            gt = train_data['label'].type(torch.FloatTensor).to(device)
            model = model.to(device).eval()
            SegNet_model = SegNet_model.to(device).train()



            target_image = torch.tensor([]).to(device)
            target_gt = torch.tensor([]).to(device)

            with torch.no_grad():
                _, out_list = model.feature_list(org_image)
                feature_list = out_list[-1]
                if feature_list.shape[1] == 2048:
                    feature_list = feature_list.permute(0, 2, 1, 3)
                    feature_list = F.interpolate(feature_list, [4096, 1])
                    feature_list = feature_list.permute(0, 2, 1, 3).squeeze(2).squeeze(2)



            ### batch rearrangement
            for idx in range(args.batch_size):
                if idx % 2 == 0:
                    tmp_image, tmp_gt = feature_list[idx], gt[idx]
                else:
                    alpha = random.random()
                    inter_image = alpha * feature_list[idx] + (1-alpha) * tmp_image
                    inter_gt = alpha * gt[idx] + (1-alpha) * tmp_gt
                    tmp_image = torch.unsqueeze(torch.cat((torch.unsqueeze(tmp_image, 0),
                                                           torch.unsqueeze(feature_list[idx], 0),
                                                           torch.unsqueeze(inter_image, 0)),
                                                          dim=0),
                                                dim=0)
                    target_image = torch.cat((target_image, tmp_image), dim=0)
                    target_gt = torch.cat((target_gt, torch.unsqueeze(inter_gt, 0)), dim=0)




            optimizer.zero_grad()

            #### forward path
            output = SegNet_model(target_image)

            #### calc loss
            cons_loss = MSE_loss.loss(output, target_gt)


            cons_loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += cons_loss.item()



        print('Epoch [{}/{}], Step {}, loss = {:.4f}, exe time: {:.2f}, lr: {:.4f}*e-4'
                  .format(epoch, args.num_epochs, i+1,
                          total_loss/len(train_loader),
                          time.time() - stime,
                          scheduler.get_last_lr()[0] * 10 ** 4))




        time.sleep(0.001)
        torch.save({
            'model': SegNet_model.state_dict(),
            'epoch': epoch,
            'init_lr' : scheduler.get_last_lr()[0]
            }, os.path.join(save_model, env,args.net_type, 'checkpoint_last.pth.tar'))

if __name__ == '__main__':
    if args.mode == "test":
        test()
    elif args.mode == "train":
        train()
# python OOD_classification_SegNet.py -d caltech --mode test --g 1000 -p /home/seonghun20/code/Mobticon/classification/save_model_caltech/1/vgg19_bn/checkpoint_last.pth.tar -s /home/seonghun20/code/Mobticon/classification/save_model/SegNet/1/vgg19_bn/checkpoint_last.pth.tar --num_classes 128 --num_epochs 50 --gpu 1 --batch_size 256 --net_type vgg19_bn --where server2 -l 5e-3
