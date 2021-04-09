import os
import random
import copy
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from collections import defaultdict



def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]





class MNIST_manual(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.folder_list = os.listdir(self.image_dir)

        self.image_list = []
        self.gt_list = []

        for gt in self.folder_list:
            locals()[str(gt)] = os.path.join(image_dir, str(gt))
            self.image_list.extend(listdir_fullpath(locals()[str(gt)]))
            self.gt_list.extend([gt] * len(os.listdir(locals()[str(gt)])))

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = self.preprocess(Image.open(self.image_list[idx]))

        return img, torch.tensor(int(self.gt_list[idx]))




class Dog_metric_dataloader(Dataset):
    def __init__(self, image_dir, num_class, mode, resize=(160,160), soft_label = False, repeat=1):
        self.soft_label = soft_label
        self.image_dir = os.path.join(image_dir, mode)
        self.class_list = sorted(os.listdir(self.image_dir))
        # assert len(self.class_list) == num_class, "num_class is not equal to dataset's class number"

        self.image_fol_list = []
        self.image_list = []
        self.gt_list = []
        self.len_list = []
        self.repeat = repeat

        for i, fol in enumerate(self.class_list):
            print("label {} : folder {}".format(i, fol), end="\t")
            locals()[fol] = os.path.join(self.image_dir, fol)
            self.image_fol_list.extend(listdir_fullpath(locals()[fol]))
            self.gt_list.extend([i]*len(os.listdir(locals()[fol])))
            self.len_list.append(len(os.listdir(locals()[fol])))

        print()

        for file in self.image_fol_list:
            self.image_list.append([np.asarray(Image.open(file).convert("RGB"))])


        if mode == "train":
            self.preprocess = transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


        self.num_image = len(self.image_list)

    def __len__(self):
        return len(self.image_list) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.image_list)
        img = self.preprocess(Image.fromarray(self.image_list[idx][0], 'RGB'))

        if self.soft_label:
            gt = [0.1/(len(self.class_list)-1)] * len(self.class_list)
            gt[self.gt_list[idx]] = 0.9
        else:
            gt = [0] * len(self.class_list)
            gt[self.gt_list[idx]] = 1


        return {'input' : img,
                'label' : torch.tensor(gt)}


class Dog_dataloader(Dataset):
    def __init__(self, image_dir, num_class, mode, resize=(160,160), repeat=1):
        self.image_dir = os.path.join(image_dir, mode)
        self.class_list = sorted(os.listdir(self.image_dir))
        # assert len(self.class_list) == num_class, "num_class is not equal to dataset's class number"

        self.image_list = []
        self.gt_list = []
        self.len_list = []
        self.repeat = repeat

        for i, fol in enumerate(self.class_list):
            print("label {} : folder {}".format(i, fol), end="\t")
            locals()[fol] = os.path.join(self.image_dir, fol)
            self.image_list.extend(listdir_fullpath(locals()[fol]))
            self.gt_list.extend([i]*len(os.listdir(locals()[fol])))
            self.len_list.append(len(os.listdir(locals()[fol])))
        print()
        if mode == "train":
            self.preprocess = transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


        self.num_image = len(self.image_list)

    def __len__(self):
        return len(self.image_list) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.image_list)

        img = Image.open(self.image_list[idx]).convert("RGB")
        img = self.preprocess(img)
        gt = [0] * len(self.class_list)
        gt[self.gt_list[idx]] = 1

        return {'input' : img,
                'label' : torch.tensor(gt)}



class modified_Dog_dataloader(Dataset):
    def __init__(self, image_dir, num_class, mode):
        self.image_dir = os.path.join(image_dir, mode)
        self.class_list = sorted(os.listdir(self.image_dir))
        assert len(self.class_list) == num_class, "num_class is not equal to dataset's class number"

        self.image_list = []
        self.gt_list = []
        self.len_list = []
        for i, fol in enumerate(self.class_list):
            print("label {} : folder {}".format(i, fol), end="\t")
            locals()[fol] = os.path.join(self.image_dir, fol)
            tmp = 0
            for each_fol in os.listdir(locals()[fol]):
                self.image_list.extend(listdir_fullpath(os.path.join(locals()[fol], each_fol)))
                self.gt_list.extend([i]*len(os.listdir(os.path.join(locals()[fol], each_fol))))
                tmp += len(os.listdir(os.path.join(locals()[fol], each_fol)))
            self.len_list.append(tmp)
        print()

        if mode == "train":
            self.preprocess = transforms.Compose([
                transforms.Resize((320,320)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.num_image = len(self.image_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])
        img = self.preprocess(img)
        gt = [0] * len(self.class_list)
        gt[self.gt_list[idx]] = 1
        return img, torch.tensor(gt)


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


class Mobticon_dataloader(Dataset):
    def __init__(self, image_dir, condition_list, version = "capture", mode = "train"):
        self.condition_num = len(condition_list)

        self.image_list = []
        self.gt_list = []
        for i, condition in enumerate(condition_list):
            locals()[condition] = os.path.join(image_dir, condition, version, mode)
            self.image_list.extend(listdir_fullpath(locals()[condition]))
            self.gt_list.extend([i]*len(os.listdir(locals()[condition])))

        self.preprocess = transforms.Compose([
            transforms.Resize((512, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.num_image = len(self.image_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])
        img = self.preprocess(img)
        gt = [0] * self.condition_num
        gt[self.gt_list[idx]] = 1
        return img, torch.tensor(gt)



class MNIST(Dataset):
    def __init__(self, imagedir, mode):
        self.imgdir = os.path.join(imagedir, mode)
        self.gtlist = sorted(os.listdir(self.imgdir))
        self.imglist = []
        for fol in self.gtlist:
            self.imglist.extend(sorted(listdir_fullpath(os.path.join(self.imgdir, fol))))
        self.gt = []
        for file in self.imglist:
            self.gt.append(file[file[:file.rfind("/")].rfind("/")+1:file.rfind("/")])
        self.len = len(self.imglist)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.imglist)

    def get_gtlist(self):
        return self.gtlist

    def __getitem__(self, idx):
        img = Image.open(self.imglist[idx]).convert("L")
        img = self.preprocess(img)
        return img, self.gt[idx]



class cifar10_dataloader(Dataset):
    def __init__(self, imagedir, mode):
        self.imgdir = os.path.join(imagedir, mode)
        self.imglist = sorted(os.listdir(self.imgdir))
        self.gtlist = []
        if 'train' in mode:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])

        f = open(os.path.join(imagedir, "labels.txt"), "r")
        while True:
            line = f.readline()
            if not line: break
            self.gtlist.append(line[:-1])
        f.close()

    def get_gtlist(self):
        return self.gtlist

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        gt = []
        for gt_name in self.gtlist:
            if gt_name in self.imglist[idx]:
                gt = self.gtlist.index(gt_name)
                break
        img = Image.open(os.path.join(self.imgdir, self.imglist[idx])).convert("RGB")
        img = self.preprocess(img)
        return img, gt

class MVTec_dataloader(Dataset):
    def __init__(self, image_dir, mode, in_size):
        self.gtlist = sorted(os.listdir(image_dir))
        self.mode = mode
        self.imglist = []
        self.gt = []

        if mode == "train":
            mode += "/good"
            for gt in self.gtlist:
                self.gt.extend(self.gtlist * len(os.path.join(image_dir, gt, mode)))
                self.imglist.extend(sorted(listdir_fullpath(os.path.join(image_dir, gt, mode))))
        elif self.mode == "train_one":
            self.gt = image_dir[image_dir.rfind("/"):]
            self.imglist.extend(sorted(listdir_fullpath(os.path.join(image_dir, "train/good"))))
        elif "test" in self.mode:
            self.test_list = os.listdir(os.path.join(image_dir, self.mode))
            print(self.test_list)
            for list in self.test_list:
                self.imglist.extend(sorted(listdir_fullpath(os.path.join(image_dir, self.mode, list))))
                self.gt.extend(list*len(sorted(listdir_fullpath(os.path.join(image_dir, self.mode, list)))))

        if "train" in mode:
            self.preprocess = transforms.Compose([
                transforms.Resize([in_size,in_size]),
                # transforms.Pad(8,padding_mode="symmetric"),
                # transforms.RandomCrop((in_size,in_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.5, 0.5, 0.5]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize([in_size,in_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img = self.preprocess(Image.open(self.imglist[idx]).convert("RGB"))
        gt = self.gt if self.mode == "train_one" else self.gt[idx]
        return img, gt


class cifar_anomaly(Dataset):
    def __init__(self, imagedir, mode):
        self.imgdir = imagedir
        self.imglist = sorted(os.listdir(self.imgdir))
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        if 'normal' == mode:
            self.gt = "normal"
        else:
            self.gt = "abnormal"

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.imgdir, self.imglist[idx])).convert("RGB")
        img = self.preprocess(img)
        return img, self.gt



# if __name__ == '__main__':
#     a = cifar100_dataloader(imagedir="/media/seonghun/data1/CIFAR100", mode="fine/train", anomally=None)
#     trainloader = DataLoader(a,
#         batch_size=512, shuffle=True, num_workers=2)
#
#     print(b)

