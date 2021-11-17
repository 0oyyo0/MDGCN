import numpy as np
from LoadData import *
import numpy.matlib


class GetInst_A(object):
    def __init__(self, useful_sp_lab, img3d, gt, trpos, scale = 3):
        self.useful_sp_lab = useful_sp_lab
        self.img3d = img3d
        [self.r, self.c, self.l] = np.shape(img3d) # (145,145,200)
        self.num_classes = int(np.max(gt)) # 16
        self.img2d = np.reshape(img3d,[self.r*self.c, self.l]) # (21025, 200)
        self.sp_num = np.array(np.max(self.useful_sp_lab), dtype='int') # 949
        gt = np.array(gt, dtype='int')
        self.gt1d = np.reshape(gt, [self.r*self.c]) # (21025,) 一列
        self.gt_tr = np.array(np.zeros([self.r*self.c]), dtype='int') #？ zero?
        self.gt_te = self.gt1d
        trpos = np.array(trpos, dtype='int')
        self.trpos = (trpos[:,0]-1)*self.c+trpos[:,1]-1 # matlab从1开始编号的, 行乘以列，因为img3d变成
        ###
        self.sp_mean = np.zeros([self.sp_num, self.l]) # CalSpMean中计算 (949, 200) 
        self.sp_center_px = np.zeros([self.sp_num, self.l]) 
        self.sp_label = np.zeros([self.sp_num]) #sp的label怎么给的？(949, 16)
        self.trmask = np.zeros([self.sp_num])
        self.temask = np.ones([self.sp_num])
        self.sp_nei = []
        self.sp_label_vec = []
        self.sp_A = [] 
        self.support = []
        self.CalSpMean()    
        self.CalSpNei()
        self.CalSpA(scale)
        
    def CalSpMean(self): #得到超像素块的特征(由原来像素点的平均值得到),以及该像素块的标签
        self.gt_tr[self.trpos] = self.gt1d[self.trpos]
        mark_mat = np.zeros([self.r*self.c]) # (21025,)
        mark_mat[self.trpos] = -1 #self.trpos (450,) mark_mat[self.trpos] (450,)
        for sp_idx in range(1, self.sp_num+1):
            region_pos_2d = np.argwhere(self.useful_sp_lab == sp_idx)
            region_pos_1d = region_pos_2d[:, 0]*self.c + region_pos_2d[:, 1] #
            px_num = np.shape(region_pos_2d)[0] #行数
            if np.sum(mark_mat[region_pos_1d])<0: #?
                self.trmask[sp_idx-1] = 1
                self.temask[sp_idx-1] = 0
            region_fea = self.img2d[region_pos_1d, :]
            if self.trmask[sp_idx-1] == 1:
                region_labels = self.gt_tr[region_pos_1d]
            else:
                region_labels = self.gt_te[region_pos_1d]
            # print("region_labels", region_labels)
            # print("np.bincount(region_labels), 0)",np.bincount(region_labels))
            # print("np.delete(np.bincount(region_labels), 0)",np.delete(np.bincount(region_labels), 0))
            # print("np.argmax(np.delete(np.bincount(region_labels), 0))+1", np.argmax(np.delete(np.bincount(region_labels), 0))+1)
            # region_labels [14 14 14 14 14 14 14 14 14 14 14 14]
            # np.bincount(region_labels), 0) [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 12]
            # np.delete(np.bincount(region_labels), 0) [ 0  0  0  0  0  0  0  0  0  0  0  0  0 12]
            # np.argmax(np.delete(np.bincount(region_labels), 0))+1 14
            # region_labels [14 14 14 14 14  0  0  0  0  0  0]
            # np.bincount(region_labels), 0) [6 0 0 0 0 0 0 0 0 0 0 0 0 0 5]
            # np.delete(np.bincount(region_labels), 0) [0 0 0 0 0 0 0 0 0 0 0 0 0 5]
            # np.argmax(np.delete(np.bincount(region_labels), 0))+1 14
            self.sp_label[sp_idx-1] = np.argmax(np.delete(np.bincount(region_labels), 0))+1 #sp的label选择的是原来图像像素点处的标签的最多的值，比如此块sp处对应到原来图上有两个类别，选择点数最多的那个类别
            region_pos_idx = np.argwhere(region_labels == self.sp_label[sp_idx-1])
            pos1 = region_pos_1d[region_pos_idx]
            self.sp_rps = np.mean(self.img2d[pos1, :], axis = 0)
            vj = np.sum(np.power(np.matlib.repmat(self.sp_rps, px_num, 1)-region_fea, 2), axis=1)
            vj= np.exp(-0.2*vj)
            self.sp_mean[sp_idx-1, :] = np.sum(np.reshape(vj, [np.size(vj), 1])*region_fea, axis=0)/np.sum(vj)
        sp_label_mat = np.zeros([self.sp_num, self.num_classes])
        for row_idx in range(np.shape(self.sp_label)[0]):
            col_idx = int(self.sp_label[row_idx])-1
            sp_label_mat[row_idx, col_idx] = 1
        self.sp_label_vec = self.sp_label
        self.sp_label = sp_label_mat #转化为one-hot (949, 16)
        
    def CalSpNei(self): # 得到邻居超像素块
        for sp_idx in range(1, self.sp_num+1):
            nei_list = []
            region_pos_2d = np.argwhere(self.useful_sp_lab == sp_idx)#行、列
            r1 = np.min(region_pos_2d[:, 0])
            r2 = np.max(region_pos_2d[:, 0])
            c1 = np.min(region_pos_2d[:, 1])
            c2 = np.max(region_pos_2d[:, 1]) #得到一个矩形框,框住这个超像素块，这个超像素块不一定是矩形的
            for r in range(r1, r2+1):
                pos1 = np.argwhere(region_pos_2d[:, 0] == r)[:, 0] #行数在矩阵中的行 [ 9 10 11 12]
                min_col = np.min(region_pos_2d[:, 1][pos1]) # pos1中当前行的目标块的最小列
                max_col = np.max(region_pos_2d[:, 1][pos1])
                nc1 = min_col-1
                nc2 = max_col+1
                if nc1>=0:
                    nei_list.append(self.useful_sp_lab[r, nc1])
                if nc2<=self.c-1:
                    nei_list.append(self.useful_sp_lab[r, nc2]) # 每一行的超像素块的左右邻居
            for c in range(c1, c2+1):
                pos1 = np.argwhere(region_pos_2d[:, 1] == c)[:, 0]
                min_row = np.min(region_pos_2d[:, 0][pos1])
                max_row = np.max(region_pos_2d[:, 0][pos1])  
                nr1 = min_row-1
                nr2 = max_row+1
                if nr1>=0:
                    nei_list.append(self.useful_sp_lab[nr1, c])
                if nr2<=self.r-1:
                    nei_list.append(self.useful_sp_lab[nr2, c]) # 每一列的超像素块的上下邻居
            nei_list = list(set(nei_list))
            nei_list = [int(list_item) for list_item in nei_list]
            # print(nei_list)
            if 0 in nei_list:
                nei_list.remove(0)
            self.sp_nei.append(nei_list)#包围超像素块的邻居[[2, 26], [27, 1, 26, 3], [2, 4, 28], [29, 3, 5], ...]第一个像素块的邻居块有2，26

    def CalSpA(self, scale = 1): # 根据跳数得到邻接矩阵A，权重由Eu_dist得到
        sp_A_s1 = np.zeros([self.sp_num, self.sp_num])
        for sp_idx in range(1, self.sp_num+1):
            sp_idx0 = sp_idx-1 # 行
            cen_sp = self.sp_mean[sp_idx0] # (200,)
            nei_idx = self.sp_nei[sp_idx0] # list 第1个像素块的邻居[2, 26]
            nei_idx0 = np.array([list_item-1 for list_item in nei_idx], dtype=int) # 列[1, 25]
            cen_nei = self.sp_mean[nei_idx0, :]# 200列，行数不确定 找到第1和第25像素块的特征
            dist1 = self.Eu_dist(cen_sp, cen_nei)
            sp_A_s1[sp_idx0, nei_idx0] = dist1 # sp_A_s1[0,[1,25]] = dist1 sp_A_s1[0][1]=sp_A_s1[0][25]
            

            
        self.sp_A.append(sp_A_s1) #第一个multi-scale
        for scale_idx in range(scale-1): # scale 0 1  得到第二个第三个multi-scale
            self.sp_A.append(self.AddConnection(self.sp_A[-1])) # 第二个由第一个得到?第三个由第二个得到？
            # sp_A[0] = AddConnection(sp_A_s1)
            # sp_A[1] = AddConnection(sp_A[0])
            # sp_A[2] = AddConnection(sp_A[1])
        for scale_idx in range(scale):  
            self.sp_A[scale_idx] = self.SymmetrizationMat(self.sp_A[scale_idx])
            

    def AddConnection(self, A):
        A1 = A.copy()#(949, 949)
        num_rows = np.shape(A)[0] #949
        for row_idx in range(num_rows):
            pos1 = np.argwhere(A[row_idx, :]!=0) # (949, 949)A中某一行Eu_dist不为0的地方
            for num_nei1 in range(np.size(pos1)): # 对每一个不为0的地方处理，就是两个像素块有联系的地方
                nei_ori = A[pos1[num_nei1, 0], :].copy() # pos1中第0、1、2、3。。。个位置 A中这些位置的行 就是像素块的第二层邻居
                pos2 = np.argwhere(nei_ori!=0)[:, 0] #A中这些位置的行的不为0的列
                
                nei1 = self.sp_mean[pos2, :]#这些行的特征
                dist1 = self.Eu_dist(self.sp_mean[row_idx, :], nei1)
                A1[row_idx, pos2] = dist1
            A1[row_idx, row_idx] = 0
        return A1
             
             
             
    def Eu_dist(self, vec, mat):
        rows = np.shape(mat)[0]
        mat1 = np.matlib.repmat(vec, rows, 1) # 由传入的矩阵平移复制产生新矩阵 原矩阵作为一个整体 复制的行数、列数
        dist1 = np.exp(-0.2*np.sum(np.power(mat1-mat, 2), axis = 1))#np.power 前面部分的2次方 np.sum(, axis = 1) 每一行求和
        return dist1   

    def SymmetrizationMat(self, mat):
        [r, c] = np.shape(mat)
        if r!=c:
            print('Input is not square matrix')
            return
        for rows in range(r):
            for cols in range(rows, c):
                e1 = mat[rows, cols]
                e2 = mat[cols, rows]
                if e1+e2!=0 and e1*e2 == 0:
                    mat[rows, cols] = e1+e2
                    mat[cols, rows] = e1+e2
        return mat #有向图变无向图

    def CalSupport(self, A): #DAD
        num1 = np.shape(A)[0]
        A_ = A + 15*np.eye(num1)
        D_ = np.sum(A_, 1)
        D_05 = np.diag(D_**(-0.5))
        support = np.matmul(np.matmul(D_05, A_), D_05)
        return support
