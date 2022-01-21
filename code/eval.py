
"""
eval.py
plot_hist
- plot histogram for angles
plot_line
- plot line graph for angles
plot_cm
- plot confusion matrix
plot_loss
- plot line graph for loss
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

class_folders = ['ape', 'benchvise', 'cam', 'cat', 'duck']

def plot_hist(ang_difference, iter):
    curr_hist = []
    degree_10   = sum(i < 10 for i in ang_difference)
    degree_20   = sum(i < 20 for i in ang_difference)
    degree_40   = sum(i < 40 for i in ang_difference)
    degree_180  = sum(i < 180 for i in ang_difference)

    try:
        degree_10   /= 3535
        degree_20   /= 3535
        degree_40   /= 3535
        degree_180  /= 3535

        curr_hist.append(degree_10*100.)
        curr_hist.append(degree_20*100.)
        curr_hist.append(degree_40*100.)
        curr_hist.append(degree_180*100.)

        bin = ['10', '20', '40', '180']
        x_pos = [i for i, _ in enumerate(bin)]
        plt.style.use('ggplot')
        plt.bar(x_pos, curr_hist, color='green')
        plt.xlabel("Angles, $^\circ$")
        plt.ylabel("Percentage, %")
        plt.title("Tolerance angles histogram")
        plt.xticks(x_pos, ('<10$^\circ$', '<20$^\circ$', '<40$^\circ$', '<180$^\circ$'))
        plt.yticks(np.arange(0, max(curr_hist)+1, 5.))
        # plt.show()
        plt.savefig('hist_' + str(iter) + '.png')
        plt.close('all')
    except ZeroDivisionError:
        pass

    return curr_hist

def plot_line(hist):
    line_10 = np.concatenate(([0],hist[:,0]))
    line_20 = np.concatenate(([0],hist[:,1]))
    line_40 = np.concatenate(([0],hist[:,2]))
    line_180 = np.concatenate(([0],hist[:,3]))

    x=np.arange(np.shape(line_180)[0])

    fig=plt.figure()
    plt.style.use('ggplot')
    ax=fig.add_subplot(111)

    ax.plot(x,line_10,c='b',marker="^",ls='--',label='<10$^\circ$',fillstyle='none')
    ax.plot(x,line_20,c='g',marker=(8,2,0),ls='--',label='<20$^\circ$')
    ax.plot(x,line_40,c='k',marker="^", ls='-',label='<40$^\circ$')
    ax.plot(x,line_180,c='r',marker="v",ls='-',label='<180$^\circ$')

    plt.xlabel("Iteration, $k$")
    plt.ylabel("Percentage, %")
    plt.title("Tolerance angles")

    plt.yticks(np.arange(0, 100+1, 5.))
    plt.xticks(np.arange(0, np.shape(line_180)[0]+1, 1))

    plt.legend(loc=2)
    plt.draw()
    # plt.show()
    plt.savefig('angle_line_plot.png')
    plt.close('all')
    return 0

def plot_cm(gt_label, pred_label, iter):
    cm = confusion_matrix(gt_label, pred_label)
    df_cm = pd.DataFrame([((c/707.) * 100.) for c in cm], index = [i for i in class_folders], columns = [i for i in class_folders])
    df_cm = df_cm.round(2)
    print(df_cm)
    plt.figure(figsize = (10, 7))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.savefig('confusion_m_' + str(iter) + '.png', bbox_inches='tight')
    # plt.show()
    plt.close('all')

    return 0

def plot_loss(losses):
    line = np.concatenate(([0],losses))
    x=np.arange(np.shape(line)[0])

    fig=plt.figure()
    plt.style.use('ggplot')
    ax=fig.add_subplot(111)
    ax.plot(x,line,c='r',ls='-')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.xticks(np.arange(0, np.shape(line)[0]+1, 10))

    plt.legend(loc=2)
    plt.draw()
    # plt.show()
    plt.savefig('loss_line_plot.png')
    plt.close('all')
    return 0
