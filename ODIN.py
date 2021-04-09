import os
where = "local"
env = "3"
if where == "server2":
    image_dir = "/home/seonghun20/data/Recording/road_condition"
    os.environ["CUDA_VISIBLE_DEVICES"] = env
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
    # condition_list = ["dry", "wet"]
    data_version = "capture"                # capture, extract_road


    sample_dir = "./output"
    txt_dir = "./softmax_scores/"
    save_model = "./save_model/capture_model/0"
    check_name = "_last"
    tensorboard_dir = "./tensorboard"
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Hyper-parameters
    init_lr = 5e-4
    eps = 1e-8
    num_epochs = 50
    temp = 1000
    perturbation = 0.0012
    batch_size = 1

    ### data config
    test_data = load_data.Mobticon_dataloader(image_dir = image_dir,
                                              condition_list = condition_list.__add__(["anomaly_image"]),
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
        print("load checkpoint {}".format(check_name))

        checkpoint = torch.load(os.path.join(save_model, "checkpoint{}.pth.tar".format(check_name)))

        ##### load model
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]

    Path(txt_dir).mkdir(exist_ok=True)

    #### loss config
    criterion = nn.BCEWithLogitsLoss()

    # Start test
    in_file = open(os.path.join(txt_dir, "IN_ODIN.txt"), "w")
    out_file = open(os.path.join(txt_dir, "OUT_ODIN.txt"), "w")

    j=0
    best_score=0

    stime = time.time()

    for i, (org_image, gt) in enumerate(test_loader):

        org_image = org_image.to(device).requires_grad_(True)
        model = model.to(device).train()

        #### forward path
        output = model(org_image)
        label = torch.argmax(output, dim = 1)
        label = torch.eye(2)[label].to(device)

        #### no perturbation, no temperature scaling
        #### calc loss
        class_loss = criterion(output, label)
        class_loss.backward()

        gradient = torch.ge(org_image.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient[0][0] = (gradient[0][0]) / (0.229)
        gradient[0][1] = (gradient[0][1]) / (0.224)
        gradient[0][2] = (gradient[0][2]) / (0.225)

        tempInputs = torch.add(org_image.data, -perturbation, gradient)
        outputs = model(tempInputs)
        outputs = outputs / temp

        nnOutputs = outputs.data.cpu().numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        if torch.argmax(gt) == test_data.condition_num -1:
            out_file.write("{}, {}, {}\n".format(temp, perturbation, np.max(nnOutputs)))
        elif torch.argmax(gt) == 0:
            in_file.write("{}, {}, {}\n".format(temp, perturbation, np.max(nnOutputs)))
        else:
            in_file.write("{}, {}, {}\n".format(temp, perturbation, np.max(nnOutputs)))

    in_file.close()
    out_file.close()



    print('Step {}, exe time: {:.2f}, fps: {:.2f}'
              .format(i+1, time.time() - stime, test_data.num_image/(time.time() - stime)))

