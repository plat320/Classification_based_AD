#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26
import os
import argparse

parser = argparse.ArgumentParser(description='Classifier training')
parser.add_argument('-d', '--dataset', type=str, default="top5", metavar='N', help='animal | top5 | group2 | caltech | dog')
parser.add_argument('-m', '--model_path', required=True, type=str, metavar='N', help='checkpoint path')
parser.add_argument('-c','--num_classes', type=int, default=4, help='the # of classes')
parser.add_argument('-n','--net_type', type=str, required=True, help='resnet34 | resnet50 | vgg16 | vgg16_bn | vgg19 | vgg19_bn')
parser.add_argument('--where', type=str, default="local", help='which machine')
parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu index')
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
    elif args.dataset == "tmp":
        # image_dir = "/mnt/nas55/Personal/20seonghun/dataset/Mobticon_tmp/ICE_ANOMALY_dataset"
        # #### python OOD_classification_transfer.py -d tmp --num_classes 3 --gpu 3 --batch_size 32 --net_type resnet50 --where server2 -l 5e-3

        image_dir = "/mnt/nas55/Personal/20seonghun/dataset/Mobticon_tmp/SNOW_ANOMALY_dataset"
        #### python OOD_classification_transfer.py -d tmp --num_classes 3 --gpu 3 --batch_size 32 --net_type resnet50 --where server2 -l 5e-3

        # image_dir = "/mnt/nas55/Personal/20seonghun/dataset/Mobticon_tmp/ALL_CLASS_dataset"
        # #### python OOD_classification_transfer.py -d tmp --num_classes 4 --gpu 0 --batch_size 32 --net_type resnet50 --where server2 -l 5e-3 --not_test_ODIN

        # image_dir = "/mnt/nas55/Personal/20seonghun/dataset/Mobticon_tmp/ALL_WET_dataset"
        # #### python OOD_classification_transfer.py -d tmp --num_classes 3 --gpu 2 --batch_size 32 --net_type resnet50 --where server2 -l 5e-3 --not_test_ODIN

    OOD_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/COCO"
    # OOD_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/CUB_200"
elif args.where == "server1":
    image_dir = "/home/seonghun/anomaly/data/MVTec"
elif args.where == "local":
    if args.dataset == "animal":
        image_dir = "/media/seonghun/data1/animal_data"
    elif args.dataset == "top5":
        image_dir = "/media/seonghun/data/stanford_dog/top5"

#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

import torch
import torch.optim as optim
import numpy as np
import time
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path

import load_data
import models

import copy
import os.path as osp

import cv2
import matplotlib.cm as cm
import torch

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
    GradCAMmodule,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False

def denormalization (array, mins, range):
    A = []
    for x in array:
        m = [(float(xi) * range) + mins for xi in x]
        A.append(m)
    return A


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + 255*raw_image[...,::-1].astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(np.clip(gcam, 0,255)))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


def main():
    output_dir = "./save_fig"


    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    eps = 1e-8

    ### data config
    test_dataset = load_data.Dog_dataloader(image_dir=image_dir,
                                            num_class=args.num_classes,
                                            mode="test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=2)

    ### novelty data
    out_test_dataset = load_data.Dog_dataloader(image_dir=image_dir,
                                                num_class=args.num_classes,
                                                mode="OOD")
    out_test_loader = torch.utils.data.DataLoader(out_test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=2)

    ##### model, optimizer config
    if args.net_type == "resnet50":
        model = models.resnet50(num_c=args.num_classes, pretrained=True)
    elif args.net_type == "resnet34":
        model = models.resnet34(num_c=args.num_classes, num_cc=args.OOD_num_classes, pretrained=True)
    elif args.net_type == "vgg19":
        model = models.vgg19(num_c=args.num_classes, num_cc=args.OOD_num_classes, pretrained=True)
    elif args.net_type == "vgg16":
        model = models.vgg16(num_c=args.num_classes, num_cc=args.OOD_num_classes, pretrained=True)
    elif args.net_type == "vgg19_bn":
        model = models.vgg19_bn(num_c=args.num_classes, num_cc=args.OOD_num_classes, pretrained=True)
    elif args.net_type == "vgg16_bn":
        model = models.vgg16_bn(num_c=args.num_classes, num_cc=args.OOD_num_classes, pretrained=True)



    print("load checkpoint_last")
    checkpoint = torch.load(args.model_path)

    ##### load model
    model.load_state_dict(checkpoint["model"])
    start_epoch = checkpoint["epoch"]
    optimizer = optim.SGD(model.parameters(), lr=checkpoint["init_lr"])

    #### create folder
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    model = model.to(device).eval()
    # Start grad-CAM
    bp = BackPropagation(model = model)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    target_layer = "layer4"

    stime = time.time()

    gcam = GradCAM(model=model)


    grad_cam = GradCAMmodule(target_layer, output_dir)
    grad_cam.model_config(model)
    for j, test_data in enumerate(test_loader):
        #### initialized
        org_image = test_data['input'].to(device)
        target_class = test_data['label'].to(device)

        target_class = int(target_class.argmax().cpu().detach())
        result = model(org_image).argmax()
        print("number: {} pred: {} target: {}".format(j, result, target_class))
        result = int(result.cpu().detach())
        grad_cam.saveGradCAM(org_image, result, j)
        # grad_cam.saveGradCAM(org_image, target_class, j)



        # probs, ids = bp.forward(org_image)
        # print(ids.argmax())
        #
        # for i in range(args.num_classes):
        #     if i == target_class:
        #         bp.backward(ids = ids[:, [i]])
        #         gradients = bp.generate()
        #         # print("\t#{}:({:.5f})".format(j, probs[0, i]))
        #         #
        #         # save_gradient(
        #         #     filename=osp.join(
        #         #         output_dir,
        #         #         "{}vanilla.png".format(i),
        #         #     ),
        #         #     gradient=gradients[0],
        #         # )
        # bp.remove_hook()
        #
        # # print("Deconvolution:")
        #
        # deconv = Deconvnet(model=model)
        # _ = deconv.forward(org_image)
        #
        # for i in range(args.num_classes):
        #     if i == target_class:
        #         deconv.backward(ids=ids[:, [i]])
        #         gradients = deconv.generate()
        #
        #         # print("\t#{}:({:.5f})".format(j, probs[0, i]))
        #         #
        #         # save_gradient(
        #         #     filename=osp.join(
        #         #         output_dir,
        #         #         "{}-deconvnet.png".format(j),
        #         #     ),
        #         #     gradient=gradients[0],
        #         # )
        #
        # deconv.remove_hook()
        #
        # # =========================================================================
        # # print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")
        #
        # _ = gcam.forward(org_image)
        #
        # gbp = GuidedBackPropagation(model=model)
        # _ = gbp.forward(org_image)
        #
        # for i in range(args.num_classes):
        #     if i == target_class:
        #     # Guided Backpropagation
        #         gbp.backward(ids=ids[:, [i]])
        #         gradients = gbp.generate()
        #
        #         # Grad-CAM
        #         gcam.backward(ids=ids[:, [i]])
        #         regions = gcam.generate(target_layer=target_layer)
        #
        #         # print("\t#{}:({:.5f})".format(j, probs[0, i]))
        #         #
        #         # Guided Backpropagation
        #         # save_gradient(
        #         #     filename=osp.join(
        #         #         output_dir,
        #         #         "{}guided.png".format(j),
        #         #     ),
        #         #     gradient=gradients[0],
        #         # )
        #
        #         # Grad-CAM
        #         save_gradcam(
        #             filename=osp.join(
        #                 output_dir,
        #                 "{}-gradcam-{}.png".format(
        #                     j, target_layer
        #                 ),
        #             ),
        #             gcam=regions[0, 0],
        #             raw_image=inv_normalize(org_image[0]).permute((1,2,0)).cpu().detach().numpy(),
        #         )
        #
        #         # Guided Grad-CAM
        #         # save_gradient(
        #         #     filename=osp.join(
        #         #         output_dir,
        #         #         "{}-guided_gradcam-{}.png".format(
        #         #             j, target_layer
        #         #         ),
        #         #     ),
        #         #     gradient=torch.mul(regions, gradients)[0],
        #         # )




if __name__ == '__main__':
    main()

# python our_grad_cam.py -n resnet50 -d caltech -c 128 -g 2 -m /home/seonghun20/code/Mobticon/my_project/student_model/caltech/checkpoin_last.pth.tar --where server2