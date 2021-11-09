import os
import scipy.io as sio
import numpy as np
import math
 
from tkinter import _flatten


def TrainTestPixel(img, gt, trnum, cand):
    intr = [];inte = [];labeltrain = [];labeltest = [];trpos = [];tepos = []
    img = img['IP_gyh']
    gt = gt['IP_gt']
    r1, c1, s1 = img.shape
    trpos2 = [] 
    tepos2 = []
    img = img.reshape(r1*c1,s1)
    gt = gt.reshape(r1*c1,1)
    c = int(max(gt)) # max(gt.tolist())
    for i in range(1, c+1):
        pos = np.where(gt == i)
        pos = pos[0]
        index1 = np.random.permutation(pos.size)
        if(pos.size <= trnum):
            trpos1 = list(pos[index1[0:cand]])
            tepos1 = list(pos[index1[cand:index1.size]])
        else:
            trpos1 = list(pos[index1[0:trnum]])
            tepos1 = list(pos[index1[trnum:index1.size]])
        
        intr = img[trpos1, :]
        inte = img[tepos1, :]
        labeltrain = gt[trpos1, :]
        labeltest = gt[tepos1, :]
        trpos.append(trpos1)
        tepos.append(tepos1)
    trpos = list(_flatten(trpos))
    tepos = list(_flatten(tepos))
    for i in range(0, len(trpos)):
        p1 = trpos[i]
        trpos2.append([p1%r1, math.ceil(p1/r1)])
    for i in range(0, len(tepos)):
        p1 = tepos[i]
        tepos2.append([p1%r1, math.ceil(p1/r1)])

    # trpos2[np.where(trpos2 == 0)[0]] = r1
    # tepos2[np.where(tepos2 == 0)[0]] = r1

    return trpos2, tepos2



if __name__ == "__main__":
    train_num = 30
    IP_gyh = sio.loadmat("./data/IP_gyh")
    IP_gt = sio.loadmat("./data/IP_gt")
    # print(IP_gyh['IP_gyh'].shape)

    trpos,tepos = TrainTestPixel(IP_gyh, IP_gt, train_num, 15)
    sio.savemat('./data/trpos.mat', {'trposmy': trpos})
    sio.savemat('./data/tepos.mat', {'teposmy': tepos})


    os.system('python trainMDGCN.py')
