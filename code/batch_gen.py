
"""
batch_gen.py
- Generate triplet batches for training.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def angle_between(v1, v2):
    return 2*np.arccos(np.abs(np.dot(v1, v2)))

def batch_generator(train_norm, db_norm, train_poses, db_poses, batch_size = 285, plot=False):
    triplet = []
    maxdiff = np.pi*2
    for cnt in range(batch_size):
        closest = 1.0
        idx = 0
        randomClass = np.random.randint(0, train_norm.shape[0])
        randomImg = np.random.randint(0, train_norm.shape[1])
        anchor = train_norm[randomClass][randomImg] # randomly from training set
        anchor_pose = train_poses[randomClass][randomImg]
        for i in range(len(db_norm[1])): # loop over db images - for pusher
            # get closest quaternion angle
            dist = angle_between(anchor_pose, db_poses[randomClass][i])
            if (dist < closest and dist != 0.0): # not identical pose & smallest
                closest = dist
                idx = i
        puller = db_norm[randomClass][idx] # most similiar quaternion wise

        # All pushers are different class
        pusher = db_norm[randomClass - 1][idx]
        triplet.append((anchor, puller, pusher))

        ## Plot the Anchor, Puller pusher if wanted
        if plot:
            fig = plt.figure()
            for i in range(3):
                fig.add_subplot(1, 3, i + 1)
                img = plt.imread(triplet[i])
                plt.imshow(img)
            plt.show()

    return (triplet, maxdiff)
