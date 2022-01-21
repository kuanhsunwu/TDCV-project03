
"""
load_data.py
- load and preprocess all data before saving them to npy files for faster loading
"""

import os
import sys
import re
import string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from utils import *

# Path to dataset folder

base_folder = '.../dataset'

dataset_folders = ['coarse', 'fine', 'real']
NUM_CLASS = 5
class_folders = ['ape', 'benchvise', 'cam', 'cat', 'duck']

def readtxt(txtfile, delim):
    with open(txtfile) as f:
        data_list = f.readlines()
        tmpstr = ''.join(x.strip() for x in data_list)
        data_list = tmpstr.split(delim)
        data_list = [int(x.strip()) for x in data_list]
        return data_list

def getposes(pose_txt):
    delim = '#'
    with open(pose_txt) as f:
        data_list = f.readlines()
        tmp_list = [x.strip(delim).strip() for x in data_list]

    pose_quart = []
    # print(len(tmp_list))
    # print(tmp_list[-1])
    # added condition because some have an empty split at the end
    if len(tmp_list)%2 != 0:
        MAX_NUM = len(tmp_list)-1
    else:
        MAX_NUM = len(tmp_list)
    for i in range(0, MAX_NUM, 2):
        per_line = tmp_list[i+1].split()
        per_line = [float(x) for x in per_line]
        pose_quart.append(per_line)
    return pose_quart

##   get all folders
coarse_folders = []
fine_folders = []
real_folders = []

dataDir = os.path.join(base_folder, dataset_folders[0])
for folder in os.listdir(dataDir):
    coarse_folders.append(os.path.join(dataDir, folder))

dataDir = os.path.join(base_folder, dataset_folders[1])
for folder in os.listdir(dataDir):
    fine_folders.append(os.path.join(dataDir, folder))

dataDir = os.path.join(base_folder, dataset_folders[2])
for folder in os.listdir(dataDir):
    if not (folder.endswith('.txt')):
        real_folders.append(os.path.join(dataDir, folder))

#   read in img paths
#   images are named according to dataset folders e.g. real0.png
#   list separated by class
#   list = [NUM_CLASS]['FULL_PATH', img_index]
#   e.g. list[0] = ['blablabla\ape\\real1.png', 1]
coarse_imgs_list = []
fine_imgs_list = []
real_imgs_list = []

for class_f in coarse_folders:
    filelist = []
    for file in os.listdir(class_f):
        if file.endswith('.png'):
            filepath = [os.path.join(class_f, file) , [int(c) for c in re.split('([0-9]+)', file) if c.isdigit()][0]]
            filelist.append(filepath)
    if (len(filelist) != 0):
        coarse_imgs_list.append(filelist)

for class_f in fine_folders:
    filelist = []
    for file in os.listdir(class_f):
        if file.endswith('.png'):
            filepath = [os.path.join(class_f, file) , [int(c) for c in re.split('([0-9]+)', file) if c.isdigit()][0]]
            filelist.append(filepath)
    if (len(filelist) != 0):
        fine_imgs_list.append(filelist)

for class_f in real_folders:
    filelist = []
    for file in os.listdir(class_f):
        if file.endswith('.png'):
            filepath = [os.path.join(class_f, file) , [int(c) for c in re.split('([0-9]+)', file) if c.isdigit()][0]]
            filelist.append(filepath)
    if (len(filelist) != 0):
        real_imgs_list.append(filelist)

#   Sort all file paths
for i in range(0,5):
    coarse_imgs_list[i].sort(key=lambda x : x[1])
    fine_imgs_list[i].sort(key=lambda x : x[1])
    real_imgs_list[i].sort(key=lambda x : x[1])

#   get all poses
#   e.g. [-0.28184579021235323, -0.6032481990846498, 0.6534595646367771, -0.3600627142949052]
coarse_poses_list = []
for class_f in coarse_folders:
    pose_txt = class_f + '/poses.txt'
    coarse_poses_list.append(getposes(pose_txt))

real_poses_list = []
for class_f in real_folders:
    pose_txt = class_f + '/poses.txt'
    real_poses_list.append(getposes(pose_txt))

fine_poses_list = []
for class_f in fine_folders:
    pose_txt = class_f + '/poses.txt'
    fine_poses_list.append(getposes(pose_txt))

#   get training split
train_txt = os.path.join(base_folder, (dataset_folders[2] + '/training_split.txt'))
delim = ','
train_data_list = readtxt(train_txt, delim)

real_train_split_imgs_list = []
real_train_split_pose_list = []
for i in range(0, NUM_CLASS):
    filelist = []
    tmp_pose_list = []
    for img_path in real_imgs_list[i]:
        if (any((img_path[1] == s) for s in train_data_list)):
            filelist.append(img_path)
            tmp_pose_list.append(real_poses_list[i][img_path[1]])
    real_train_split_imgs_list.append(filelist)
    real_train_split_pose_list.append(tmp_pose_list)

# Full train images and poses
train_imgs_list = []
train_poses_list = []
for i in range(0, NUM_CLASS):
    train_imgs_list.append(fine_imgs_list[i] + real_train_split_imgs_list[i])
    train_poses_list.append(fine_poses_list[i] + real_train_split_pose_list[i])

# rest to test set
# get imgs_list and corresponding poses
real_test_split_imgs_list = []
real_test_split_pose_list = []
for i in range(0, NUM_CLASS):
    filelist = []
    tmp_pose_list = []
    for img_path in real_imgs_list[i]:
        if (any((img_path[1] == s) for s in train_data_list)):
            pass
        else:
            filelist.append(img_path)
            tmp_pose_list.append(real_poses_list[i][img_path[1]])
    real_test_split_imgs_list.append(filelist)
    real_test_split_pose_list.append(tmp_pose_list)

# normalized dataset
train_imgs_normalized = []
test_imgs_normalized = []
db_imgs_normalized = []

#  load imgs to list
train_imgs = []
test_imgs = []
db_imgs = []

print("Loading images takes a while...")
for i in range(0, NUM_CLASS):
    train_imgs.append([np.array(Image.open(fname[0])) for fname in train_imgs_list[i]])
    test_imgs.append([np.array(Image.open(fname[0])) for fname in real_test_split_imgs_list[i]])
    db_imgs.append([np.array(Image.open(fname[0])) for fname in coarse_imgs_list[i]])

    print("\n" + "Done class: {}".format(class_folders[i]))
    print("num_train_imgs: {}".format(len(train_imgs_list[i])))
    print("num_test_imgs: {}".format(len(real_test_split_imgs_list[i])))
    print("num_coarse_imgs: {}".format(len(coarse_imgs_list[i])))

    # perform zero mean normalization and unit variance
    train_imgs_normalized.append([ (np.array(Image.open(fname[0])) - np.array(Image.open(fname[0])).mean() ) / np.array(Image.open(fname[0])).std() for fname in train_imgs_list[i]])
    test_imgs_normalized.append([ (np.array(Image.open(fname[0])) - np.array(Image.open(fname[0])).mean() ) / np.array(Image.open(fname[0])).std() for fname in real_test_split_imgs_list[i]])
    db_imgs_normalized.append([ (np.array(Image.open(fname[0])) - np.array(Image.open(fname[0])).mean() ) / np.array(Image.open(fname[0])).std() for fname in coarse_imgs_list[i]])

    print("num_train_imgs: {}".format(len(train_imgs_normalized[i])))
    print("num_test_imgs: {}".format(len(test_imgs_normalized[i])))
    print("num_coarse_imgs: {}".format(len(db_imgs_normalized[i])))

# save all data to npy files so it's faster
np.save(os.path.join(base_folder, "train_imgs"), train_imgs)
np.save(os.path.join(base_folder, "test_imgs"), test_imgs)
np.save(os.path.join(base_folder, "db_imgs"), db_imgs)

np.save(os.path.join(base_folder, "train_imgs_normalized"), train_imgs_normalized)
np.save(os.path.join(base_folder, "test_imgs_normalized"), test_imgs_normalized)
np.save(os.path.join(base_folder, "db_imgs_normalized"), db_imgs_normalized)

np.save(os.path.join(base_folder, "train_imgs_list"), train_imgs_list)
np.save(os.path.join(base_folder, "test_imgs_list"), real_test_split_imgs_list)
np.save(os.path.join(base_folder, "db_imgs_list"), coarse_imgs_list)

np.save(os.path.join(base_folder, "train_poses"), train_poses_list)
np.save(os.path.join(base_folder, "test_poses"), real_test_split_pose_list)
np.save(os.path.join(base_folder, "db_poses"), coarse_poses_list)
