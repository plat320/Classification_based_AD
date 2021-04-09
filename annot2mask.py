import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import shutil
import copy
from pathlib import Path
from PIL import Image

def delete_dot(folder="/mnt/hdd1/mmi_tr1_hdd1/seonghun20/CUB_200/OOD"):
    fol_list = os.listdir(folder)
    for fol in fol_list:
        change_fol = list(copy.deepcopy(fol))
        idx = fol.find(".")
        change_fol[idx] = "_"
        change_fol = ''.join(change_fol)
        os.rename(os.path.join(folder, fol), os.path.join(folder, change_fol))

def check_folder_not_empty(folder):
    fol_list = os.listdir(folder)
    for fol in fol_list:
        print(len(os.listdir(os.path.join(folder, fol))))

def remove_continuously(folder, out_folder, num_instance):
    fol_list = os.listdir(folder)
    for fol in fol_list:
        instance_list = sorted(os.listdir(os.path.join(folder, fol)))
        Path(os.path.join(out_folder, fol)).mkdir(exist_ok=True, parents=True)
        for i in range(num_instance):
            shutil.copy(os.path.join(folder, fol, instance_list[i]), os.path.join(out_folder, fol, instance_list[i]))


def img2video(folder, out_name):
    frame_array = []
    files = []
    for file in os.listdir(folder):
        files.append(file) #if "ground" not in file else print(file)
    files.sort()

    for file in files:
        filename = os.path.join(folder, file)
        img = cv2.imread(filename)
        frame_array.append(img)

    height, width, layer = img.shape
    size = (width, height)

    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for frame in frame_array:
        out.write(frame)
    out.release()


def frame_capture(video_name, output_folder):
    cap = cv2.VideoCapture(video_name)
    idx = 0
    while True:
        ret, frame = cap.read()
        # if idx%4 != 0:
        #     idx+=1
        #     continue
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, video_name[:-4]+"{:04d}.png".format(idx)), frame)
        idx += 1



def file_out(folder):
    fol_list = os.listdir(folder)
    for fol in fol_list:
        file_list = os.listdir(os.path.join(folder, fol))
        original_path = os.path.join(folder, fol)
        original_file = []
        new_file = []
        for file in file_list:
            # original_file.append(os.path.join(original_path, file))
            # new_file.append(os.path.join(folder, file))

            shutil.move(os.path.join(original_path, file), os.path.join(folder, file))

def check_shape(folder):
    list = os.listdir(folder)

    for file in list:
        if Image.open(os.path.join(folder, file)).mode !="RGB":
            print(file)

        # if cv2.imread(os.path.join(folder, file)).shape[-1] == 1:
        #     print(file)


def First_frame(folder, out_folder):
    Path(out_folder).mkdir(exist_ok=True, parents=True)
    for i, vid in enumerate(os.listdir(folder)):
        cap = cv2.VideoCapture(os.path.join(folder, vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)                               # 100th best
        if cap.isOpened():
            print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))

        ret, frame = cap.read()

        # name = vid[:-4] + ".png"                # retain original name
        name = "wet_dry2_{0:04}.png".format(i)
        print(vid)
        cv2.imwrite(os.path.join(out_folder, name), frame)

def check_name(folder):
    wpath = folder + "image"
    apath = folder + "annot"
    for vid_name in os.listdir(wpath):
        # if os.path.isfile(apath+"/"+vid_name[:-3]+"png") == False:
        #     print(apath+"/"+vid_name[:-3]+"png")
        print(apath+"/"+vid_name[:-3]+"png")

def got_resize_video(folder):                                           # 576, 320
    width = 576
    height = 320
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for fol in os.listdir(folder):
        for annot in os.listdir(folder + "/" + fol):
            cap = cv2.VideoCapture(folder + "/" + fol + "/" +annot)

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            save_folder = "./video_resize/"+fol
            if os.path.isdir(save_folder) == False:
                os.mkdir(save_folder)
            out = cv2.VideoWriter(save_folder + "/" + annot, fourcc, fps, (width, height))
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                    out.write(frame)
                else: break

            out.release()


def get_num_frame(folder):
    for vid_name in os.listdir(folder):
        cap = cv2.VideoCapture(folder + "/" + vid_name)
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if  num_frame < 150:
            print("num frame = " + str(num_frame) + "video name = "+ vid_name)

def Flip_vid(folder, vidname):
    cap = cv2.VideoCapture(folder + vidname)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if os.path.isdir("./flipvid") == False:
        os.mkdir("./flipvid")
    outname = "./flipvid/"+vidname
    out = cv2.VideoWriter(outname, fourcc, fps, (height, width))
    print(cap.isOpened())
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.transpose(frame)
            img = cv2.flip(frame, 1)
            out_frame = img
            out.write(img)
        else:
            break

    cv2.destroyAllWindows()
    out.release()

def resize_img(vid_folder, mask_folder):
    for fold in os.listdir(vid_folder):
        for vid in os.listdir(vid_folder + "/" + fold):
            cap = cv2.VideoCapture(vid_folder + "/" + fold+"/"+vid)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            mask_dir = mask_folder+"/"+ fold + "/" + vid[:-3]+"png"
            ret, frame = cap.read()
            img = cv2.imread(mask_dir)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(mask_folder + "/" + fold + "/" + vid[:-3]+"png", img)

def del_edge(folder):           # edge부분 white 지우고 근접 픽셀하고 비교해서 edge 채우는 코
    if os.path.isdir("./delete_edge") == False:
        os.mkdir("./delete_edge")

    White = (255,255,255)

    for img_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,img_name))
        width, height = img.shape[:2]
        tmp_img = np.zeros((width, height, 3), np.uint8)
        print("width: "+str(width)+", height: "+str(height))

        if img[1,1,0] == 35 : tmp_img[0,0,:] == White
        if img[width-2,1,0] == 35 : tmp_img[width-1,0,:] == White
        if img[1,height-2,0] == 35 : tmp_img[0,height-1,:] == White
        if img[width-2,height-2,0] == 35 : tmp_img[width-2,height-2,:] == White

        for i in range(1, width-1):
            if img[i,1,0] == 35 : tmp_img[i,0,:] == White
            if img[i,height -2,0] == 35 : tmp_img[i,height-1,:] == White
        for i in range(1, height-1):
            if img[1,i,0] == 35 : tmp_img[0,i,:] == White
            if img[width-2,i,0] == 35 : tmp_img[width-1,i,:] == White

        for i in range(1, width-1):
            for j in range(1, height-1):
                if img[i,j,0] > 10:
                    if img[i,j,0] !=255:
                        tmp_img[i,j,:] = White

        outname = "./delete_edge/" + img_name
        cv2.imwrite(outname, tmp_img)

def make_txt(folder):
    f = open("./train.txt", 'w')
    for file_name in os.listdir(folder):
        f.write(file_name[:-4]+"\n")

    f.close()

def got_water_annot(folder, out_folder):
    Path(out_folder).mkdir(exist_ok=True)
    for annot in os.listdir(folder):
        gt = cv2.imread(os.path.join(folder, annot))
        gt = (gt != 0)
        gt = gt * 255
        cv2.imwrite(os.path.join("./water_annot/", annot), gt)

def create_white():
    a = np.ones((320, 576, 3), dtype = np.uint8)
    a = a * 255
    cv2.imwrite("./white_mask.png", a)

def change_name(folder):
    i = 0
    for i, file in enumerate(os.listdir(folder)):
        os.rename(os.path.join(folder, file), os.path.join(folder, "{0:06d}.png".format(i + 2079)))

def crop(folder, out_folder):
    Path(out_folder).mkdir(exist_ok=True, parents=True)
    Path("./tmp").mkdir(exist_ok=True)
    First_frame(folder, "./tmp")
    # for i, file in enumerate(os.listdir("./tmp")):
    #     image = cv2.imread(os.path.join("./tmp", file))              # C*H*W
    #     height, width = image.shape[:-1]
    #     dst = image[int(height/4) : int(3*height/4), int(width/4) : int(3*width/4), :]
    #     cv2.imwrite(os.path.join(out_folder, file), dst)

def extract_image(image_dir, x, y, img_size):
    total_img = cv2.imread(image_dir)
    dst = total_img[2 + y * img_size : 2 + (y+1) * img_size,
          (x * (img_size + 2) + 2) : (x + 1) * (img_size + 2),
          :]
    cv2.imwrite("./extract.png", dst)

def get_substract_image(first, second):
    sub_image = np.uint8(np.sum(np.abs(cv2.imread(first) - cv2.imread(second)), axis=2, keepdims=True)/3)
    sub_image = plt.cm.jet_r(sub_image)[:,:,0 , : 3] * 255
    plt.imsave("./sub_image.png", np.uint8(sub_image))
    # plt.imsave("./sub_image.png", cv2.cvtColor(sub_image), cmap=plt.cm.jet_r(sub_image))

def choose_one(folder, out_folder):
    Path(out_folder).mkdir(exist_ok = True)
    img_list = []
    i=0
    for fol in os.listdir(folder):
        fol_img_list = os.listdir(os.path.join(folder, fol))
        while True:
            pick = random.choice(fol_img_list)
            if pick not in img_list:
                print(pick)
                shutil.copyfile(os.path.join(folder, fol, pick), os.path.join(out_folder, str(i) + ".jpg"))
                i +=1
                break

def delete_image(folder):
    img_list = os.listdir(folder)
    for image in img_list:
        img = cv2.imread(os.path.join(folder, image))
        if img is None:
            print(image)
            os.remove(os.path.join(folder, image))
            continue
        shape = img.shape
        if shape[0] <150 and shape[1] < 150:
            os.remove(os.path.join(folder, image))
            print(shape)


def check_channel_and_delete(folder):
    fol_list = os.listdir(folder)
    for fol in fol_list:
        image_list = os.listdir(os.path.join(folder, fol))
        for image in image_list:
            tmp = Image.open(os.path.join(folder, fol, image))
            if np.array(tmp).shape[-1] != 3:
                print(os.path.join(folder, fol, image))
                os.remove(os.path.join(folder, fol, image))





if __name__ == '__main__':
    # folder = "/media/seonghun/data1/animal dataset/animal_data/train"
    # folder = "/media/seonghun/data1/Recording/road_condition/wet_dry/video"
    # out_folder = "/media/seonghun/data1/Recording/road_condition/wet_dry/capture2"
    # video = "output2.mp4"
    # output_folder = "./2"
    # Path(output_folder).mkdir(exist_ok=True)
    # frame_capture(video, output_folder)
    # folder = "./1"
    # img2video(folder, "valid2.mp4")

    # remove_continuously("/media/seonghun/data1/CALTECH256/256_objectcategories/256_ObjectCategories", "/media/seonghun/data1/CALTECH256/test", 9)
    # check_folder_not_empty("/media/seonghun/data1/CALTECH256/OOD")
    # choose_one(folder, out_folder)
    delete_dot()
    # first = "./sub/anomaly_input.png"
    # second = "./sub/anomaly_recon.png"
    # get_substract_image(first, second)
    # extract_image("./anomaly/reconst-226.png", 1, 0, 256)
    # folder = "./firstframe/20200715"
    # change_name("./masks")
    # First_frame(folder, out_folder)
    # check_channel_and_delete(folder)
    # crop(folder, out_folder)
    # del_edge(folder)
    # change_name(folder)
    # got_water_annot(folder, out_folder)
