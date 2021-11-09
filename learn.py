# -*- coding: utf-8 -*-
import numpy as np
from numpy.lib.function_base import percentile
from tensorflow.python.ops.gen_math_ops import mod
from funcCNN import *
from BuildSPInst_A import *



data_name = 'IP'
num_classes = 16

learning_rate = 0.0005 
epochs = 5000
img_gyh = data_name+'_gyh'
img_gt = data_name+'_gt'


Data = load_HSI_data(data_name)
model = GetInst_A(Data['useful_sp_lab'], Data[img_gyh], Data[img_gt],
                  Data['trpos'], scale=3)  # 对sp进行处理：得到feature A,label,support DAD trpos是之前设置好的，保存在data文件夹下，是多行两列数据 x,y
sp_mean = np.array(model.sp_mean, dtype='float32') #(949, 200)
sp_label = np.array(model.sp_label, dtype='float32')

trmask = np.matlib.reshape(np.array(model.trmask, dtype='bool'), [
                           np.shape(model.trmask)[0], 1])
# print(trmask) #[[False] [ True][ True][False]]

temask = np.matlib.reshape(np.array(model.temask, dtype='bool'), [
                           np.shape(model.trmask)[0], 1])
sp_support = []


# for A_x in model.sp_A:
#     sp_A = np.array(A_x, dtype='float32')
#     sp_support.append(np.array(model.CalSupport(sp_A), dtype='float32'))

# print(model.num_classes)
# print("model.img2d.shape: ",model.img2d.shape)
# print("model.sp_num: ",model.sp_num)
# print(model.gt1d.shape)
# print(sp_mean.shape)
# print(sp_label)
# print(trmask.shape)

# print(model.trpos)
# print(model.trpos.shape)
# def Eu_dist(vec, mat):
#         rows = np.shape(mat)[0]
#         mat1 = np.matlib.repmat(vec, rows, 1)
#         dist1 = np.exp(-0.2*np.sum(np.power(mat1-mat, 2), axis = 1))
#         return dist1        
# def AddConnect(A):
#         A1 = A.copy()#(949, 949)
#         num_rows = np.shape(A)[0]
#         for row_idx in range(num_rows):
#             pos1 = np.argwhere(A[row_idx, :]!=0) # (949, 949)中某一行Eu_dist不为0的地方
#             for num_nei1 in range(np.size(pos1)): # 对每一个不为0的地方处理，就是两个像素块有联系的地方
#                 nei_ori = A[pos1[num_nei1, 0], :].copy() 
#                 pos2 = np.argwhere(nei_ori!=0)[:, 0] 
                
#                 nei1 = sp_mean[pos2, :]
#                 dist1 = Eu_dist(sp_mean[row_idx, :], nei1)
#                 A1[row_idx, pos2] = dist1
#             A1[row_idx, row_idx] = 0
#         return A1
# a = np.array([[1,1,1],[1,1,1],[1,1,1]])

# a1 = AddConnect(a)
# print(a1)

# print(model.sp_nei)



