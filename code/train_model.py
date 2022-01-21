
"""
train_model.py
- train and eval model
! run load_data.py first to convert data to npy files
"""

import os
import sys
import re
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from eval import *
from batch_gen import *


base_folder = 'C:/Users/Michelle/Desktop/TUM/WS1819_RCI1/W_IN2210-Tracking and Detection in CV/practicals/Project_3/code/dataset'
LOGDIR = 'C:/Users/Michelle/Desktop/TUM/WS1819_RCI1/W_IN2210-Tracking and Detection in CV/practicals/Project_3/code/models'

batchSize = 1 # take each single image in at once
numEpochs = 3535 // batchSize # for batch repeat in line 80

trainBatch = 285
LogIter = 30
NUM_EPOCHS = int((LogIter * 1000)/trainBatch)  + ((LogIter * 1000) % trainBatch > 0)
LRATE = 1e-5

dataset_folders = ['coarse', 'fine', 'real']
NUM_CLASS = 5
class_folders = ['ape', 'benchvise', 'cam', 'cat', 'duck']


# --------------------------- LOAD DATASETS FOR TENSORFLOW ---------------------
# LOAD TRAIN DATASET
data_dataset_train = tf.placeholder(tf.float32, [None, 3, 64, 64, 3])
dataset = tf.data.Dataset.from_tensor_slices(data_dataset_train)
# # Shuffle, repeat, and batch the examples.
batched_dataset_train = dataset.repeat().shuffle(100).batch(1)
batched_dataset_train = batched_dataset_train.prefetch(1)
iterator_train = batched_dataset_train.make_initializable_iterator()
next_element_train = iterator_train.get_next()

# LOAD TEST DATASET
test_db = tf.placeholder(tf.float32, [None, 64, 64, 3])
dataset_test = tf.data.Dataset.from_tensor_slices(test_db)
batched_dataset_testing = dataset_test.batch(batchSize).repeat(numEpochs)
batched_dataset_testing = batched_dataset_testing.prefetch(1)
iterator_test = batched_dataset_testing.make_initializable_iterator()
next_element_test = iterator_test.get_next()

# LOAD DATABASE
data_db = tf.placeholder(tf.float32, [None, 64, 64, 3])
dataset_db = tf.data.Dataset.from_tensor_slices(data_db)
batched_dataset_db = dataset_db.batch(batchSize).repeat(numEpochs)
batched_dataset_db = batched_dataset_db.prefetch(1)
iterator_db = batched_dataset_db.make_initializable_iterator()
next_element_db = iterator_db.get_next()
# ------------------------------------------------------------------------------

# ------------------------ SET UP THE CNN MODEL --------------------------------
inputs_ = tf.placeholder(tf.float32, [None, 64, 64, 3], name='inputs')
m = tf.placeholder(tf.float32, shape=(), name="margin")
lrate_ = tf.placeholder(tf.float32, shape=(), name="lrate")

with tf.name_scope("LeNet"):

    # Convolutional Layer #1
    end_point = 'conv1_57x57x16'
    net = tf.layers.conv2d(
      inputs=inputs_,
      filters=16,
      kernel_size=[8, 8],
      activation=tf.nn.relu,
      name=end_point)

    # Pooling Layer #1
    end_point = 'pool1_28x28x16'
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name=end_point)

    # Convolutional Layer #2 and Pooling Layer #2
    end_point = 'conv2_24x24x7'
    net = tf.layers.conv2d(
      inputs=net,
      filters=7,
      kernel_size=[5, 5],
      activation=tf.nn.relu, name=end_point)

    # Pooling Layer #2
    end_point = 'poo2_12x12x7'
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name=end_point)

    # Dense Layer
    net = tf.reshape(net, [-1, 1008])
    end_point = 'dense_256'
    net = tf.layers.dense(inputs=net, units=256, activation=None, name=end_point)

    # Logits Layer
    end_point = 'logits_16'
    logits = tf.layers.dense(inputs=net, units=16, activation=None, name=end_point)

# Calculate Loss (for both TRAIN and EVAL modes)
batch_size = tf.shape(inputs_)[0]

diff_pos = logits[0:batch_size:3] - logits[1:batch_size:3]
diff_neg = logits[0:batch_size:3] - logits[2:batch_size:3]
l2_loss_diff_pos = tf.nn.l2_loss(diff_pos) * 2
l2_loss_diff_neg = tf.nn.l2_loss(diff_neg) * 2

loss_triplets = tf.reduce_sum(tf.maximum(0., (1.-(l2_loss_diff_neg/(l2_loss_diff_pos+m)))))

loss_pairs = tf.reduce_sum(l2_loss_diff_pos)

loss = loss_triplets + loss_pairs

opt = tf.train.AdamOptimizer(learning_rate=lrate_).minimize(loss=loss, global_step=tf.train.get_global_step())

with tf.name_scope("variables"):
    # Variable to keep track of how many times the graph has been run
    global_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name="global_step")

with tf.name_scope("update"):
    # Increments the above `global_step` Variable, should be run whenever the graph is run
    increment_step = global_step.assign_add(1)

# Creates summaries for output node
with tf.name_scope("summaries"):
    tf.summary.image(name="input_img", tensor=inputs_)
    tf.summary.scalar(family='Training loss', tensor=loss, name="total_loss")
    tf.summary.scalar(family='Training loss', tensor=loss_triplets, name="loss_triplets")
    tf.summary.scalar(family='Training loss', tensor=loss_pairs, name="loss_pairs")

with tf.name_scope("global_ops"):
    # Initialization Op
    init_op = tf.global_variables_initializer()
    # Merge all summaries into one Operation
    merged_summaries = tf.summary.merge_all()

# ------------------------------------------------------------------------------

# ------------------------load npy files ---------------------------------------

# ALL FORMAT: list = [NUM_CLASS][CONTENTS ACCORDING TO DATA]
# e.g. train_imgs[0][0] = [image_array(64x64)]
print("Loading images...")
train_imgs = np.load(os.path.join(base_folder, "train_imgs.npy"))
test_imgs_ori = np.load(os.path.join(base_folder, "test_imgs.npy"))
db_imgs_ori = np.load(os.path.join(base_folder, "db_imgs.npy"))
print("Done loading.")

# list = [NUM_CLASS]['FULL_PATH', img_index]
# e.g. list[0] = ['blablabla\ape\\real1.png', 1]
# print("Loading lists...")
# train_imgs_list = np.load(os.path.join(base_folder, "train_imgs_list.npy"))
# test_imgs_list = np.load(os.path.join(base_folder, "test_imgs_list.npy"))
# db_imgs_list = np.load(os.path.join(base_folder, "db_imgs_list.npy"))
# print("Done loading.")

# e.g. train_poses_list[0][0] = [-0.28184579021235323, -0.6032481990846498, 0.6534595646367771, -0.3600627142949052]
print("Loading poses...")
train_poses_list = np.load(os.path.join(base_folder, "train_poses.npy"))
test_poses_list = np.load(os.path.join(base_folder, "test_poses.npy"))
db_poses_list = np.load(os.path.join(base_folder, "db_poses.npy"))
print("Done loading.")

print("Loading normalized images...")
train_imgs_normalized = np.load(os.path.join(base_folder, "train_imgs_normalized.npy"))
test_imgs_normalized = np.load(os.path.join(base_folder, "test_imgs_normalized.npy"))
db_imgs_normalized = np.load(os.path.join(base_folder, "db_imgs_normalized.npy"))
print("Done loading.")

# ------------------------------ GET THE INPUT DATA FOR THE NETWORK -------------------

# DB and Test images all classes stacked together - get the right shape for feeding it into network
db_imgs = db_imgs_normalized[0]
test_imgs = test_imgs_normalized[0]
for i in range(1,NUM_CLASS):
    db_imgs = np.vstack((db_imgs, db_imgs_normalized[i]))
    test_imgs = np.vstack((test_imgs, test_imgs_normalized[i]))

gt_labels = [] # get the ground truth labels
for t in range(NUM_CLASS):
    for tt in range(test_imgs_normalized.shape[1]):
        gt_labels.append(t)


# logging
loss_log = open("Log_loss.txt", "w")

descriptorMatcher = cv2.BFMatcher() # setup Descriptormatcher in OpenCV
pred_labels = []
histo = [] # store the histogram in 4 bins (10, 20, 40 or 180 degrees difference)

sess = tf.Session() # start Tensorflow Session
sess.run(init_op)
saver = tf.train.Saver()
writer = tf.summary.FileWriter(LOGDIR + '/summary', sess.graph)
TOTAL_ITER = 0

# # generate train triplets
# triplets, mar = batch_generator(train_imgs_normalized, db_imgs_normalized, train_poses_list, db_poses_list, 512)
# triplets_f32 = [np.float32(t) for t in triplets]

loss_epoch = []
for e in range(NUM_EPOCHS):

    # New train triplets every Epoch
    if e == 0 or e%15 == 0:
        triplets, mar = batch_generator(train_imgs_normalized, db_imgs_normalized, train_poses_list, db_poses_list)
        triplets_f32 = [np.float32(t) for t in triplets]

    sess.run(iterator_train.initializer, feed_dict={data_dataset_train: triplets_f32})
    loss_mean = []
    loss_itermediate = []

    for i in range(len(triplets_f32)):
        TOTAL_ITER = TOTAL_ITER + 1
        train_input = sess.run(next_element_train)
        input_dict = {inputs_: train_input[0], m: mar, lrate_: LRATE}
        batch_loss, _, step, summary = sess.run(
            [loss, opt, increment_step, merged_summaries],
            feed_dict=input_dict)
        writer.add_summary(summary, global_step=step)
        loss_mean.append(batch_loss)
        loss_itermediate.append(batch_loss)

        if TOTAL_ITER % 10 == 0:
            # saver.save(sess, LOGDIR + '/checkpoints/model',global_step=step)
            # print("Model saved successfully")
            loss_log.write("Iter: {} Training loss: {:.4f} \n".format(TOTAL_ITER,np.mean(loss_itermediate)))
            loss_itermediate = []
        else:
            pass
        # every 1000th iteration: calculate the database descriptors and get the nearest Neighbors of the test images with the db
        if TOTAL_ITER % 1000 == 0:
            database_desc = np.array([], dtype=np.float32).reshape(0,16)
            sess.run(iterator_db.initializer, feed_dict={data_db: db_imgs})

            for i in range(len(db_imgs)): # get database descriptors
                db_input = sess.run(next_element_db)
                input_dict_db = {inputs_: db_input}
                descriptors = sess.run([logits], feed_dict=input_dict_db)
                database_desc = np.vstack((database_desc,descriptors[0]))
            # Save db embeddings
            # desc_name = "db_16_" + str(TOTAL_ITER) +".npy"
            # np.save(os.path.join(base_folder, desc_name), database_desc)

            pred_labels = [] # only get one prediction for each image - clear it here
            ang_difference = [] # get the angular difference of test vs. db images

            sess.run(iterator_test.initializer, feed_dict={test_db: test_imgs})
            test_dscrp = np.array([], dtype=np.float32).reshape(0,16)
            for i in range(len(test_imgs)):
                ## Predict Test image Descriptors ...
                test_input = sess.run(next_element_test)
                input_dict_test = {inputs_: test_input}
                features = sess.run([logits], feed_dict=input_dict_test)
                test_dscrp = np.vstack((test_dscrp,features[0]))

                ## Perform Descriptor matching on feature and Database descriptors
                matches = descriptorMatcher.match(database_desc, np.reshape(*features[0], [-1, 16]))
                matches = sorted(matches, key=lambda x: x.distance) # sort matches according to distance
                # matches[0].queryIdx = closest image index [0 ... 1334] => 1335 = 5 (classes) * 267 (images per class in db)
                predClass = int(matches[0].queryIdx/ db_imgs_normalized.shape[1]) # rounds down
                pred_labels.append(predClass)

                if predClass == gt_labels[i]: # correct prediction
                # Retrieve index of test image [0:706]
                    if i < 707:
                        properTestImg = i
                    elif i >= 707:
                        looped = int(i/ 707.) # rounds down
                        properTestImg = i - (707*looped)

                    ang_difference.append(angle_between(db_poses_list[gt_labels[i]][matches[0].queryIdx - (gt_labels[i]*db_imgs_normalized.shape[1])], test_poses_list[gt_labels[i]][properTestImg])*180/np.pi)

            # After looping through the test_images
            # SAVE TEST DESCRIPTORS
            desc_name = "test_16_" + str(TOTAL_ITER) +".npy"
            np.save(os.path.join(base_folder, desc_name), test_dscrp)

            histo.append(plot_hist(ang_difference, TOTAL_ITER))
            plot_cm(gt_labels, pred_labels, TOTAL_ITER)

    print("Epoch: {}/{}...".format(e + 1, NUM_EPOCHS),
            "Training loss: {:.4f}".format(np.mean(loss_mean)))
    loss_epoch.append(np.mean(loss_mean))

np.save(os.path.join(base_folder, "histo.npy"), histo)
plot_line(np.asarray(histo))
plot_loss(np.asarray(loss_epoch))

loss_log.close()
sess.close()
writer.flush()
writer.close()
print('Training done.')
