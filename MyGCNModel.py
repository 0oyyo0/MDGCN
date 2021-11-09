# -*- coding: utf-8 -*-
import tensorflow as tf
from GCNLayer import *
import numpy as np
from funcCNN import *


def masked_softmax_cross_entropy(preds, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(loss))


def masked_accuracy(preds, labels, mask):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(accuracy_all))

class MDGCNModel(object):
    def __init__(self, features, labels, learning_rate, num_classes, mask, support, scale_num, h): #注意h，隐藏层的神经元个数
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # support就是几个scale的DAD [1]、[2]、[3]
        self.classlayers = []
        self.labels = labels
        self.inputs = features
        self.scale_num = scale_num
        self.loss = 0
        self.support = support
        self.concat_vec = []
        self.outputs = None
        self.num_classes = num_classes
        self.hidden1 = h 
        self.mask = mask
        self.hidden = [] #为了得到hidden层添加的
        self.build()
        
    def _build(self):
        for scale_idx in range(self.scale_num):
            activations = []
            activations.append(self.inputs) # feature矩阵 (949, 200)
            self.classlayers.append(GraphConvolution(act = tf.nn.softplus,
                                      input_dim = np.shape(self.inputs)[1],
                                      output_dim = self.hidden1,
                                      support = self.support[scale_idx],#(949, 949)
                                      bias = True,
                                      isnorm = False 
                                      ))  
            layer = self.classlayers[-1]  # 最新的GCN层     
            hidden = layer(activations[-1]) #(949, 25)
            self.hidden = hidden #为了得到hidden层添加的
            activations.append(hidden)


                     
            support_dynamic = tf.exp(-0.02*tf.matmul(hidden, tf.transpose(hidden))) #(949, 949)
            support_dynamic = 0.1*support_dynamic*self.Get01Mat(self.support[scale_idx]) + self.support[scale_idx] #support_dynamic = 0.1*self.Get01Mat(self.support[scale_idx])*support_dynamic + self.support[scale_idx]效果好于support_dynamic = 0.1*support_dynamic*self.Get01Mat(self.support[scale_idx]) + self.support[scale_idx]
            support_dynamic_1 = tf.matmul(tf.matmul(self.support[scale_idx], support_dynamic), tf.transpose(self.support[scale_idx])) + 0*tf.eye(np.shape(self.support[scale_idx])[0])
            self.classlayers.append(GraphConvolution(act = lambda x:x,
                                      input_dim = self.hidden1,
                                      output_dim = self.num_classes, 
                                      support = support_dynamic_1,
                                      bias = True
                                      ))   
            layer = self.classlayers[-1]
            hidden = layer(activations[-1])#(949, 16)
            activations.append(hidden)
            # 总的来说，第一层的输出hidden作为第二层的feature输入，第一层的feature就是最原始的图
            # 并且第二层的参数DAD又由第一层的输出hidden得到


            if scale_idx == 0:
                self.concat_vec = activations[-1]
            else:
                self.concat_vec += activations[-1]

            
    def build(self):
        self._build()
        self.outputs = self.concat_vec
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)        
        
    def _loss(self):
        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.labels, self.mask)

        
        
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels, self.mask)
    
    # 不为0的地方全为1
    def Get01Mat(self, mat1):
        [r, c] = np.shape(mat1)
        mat_01 = np.zeros([r, c])
        pos1 = np.argwhere(mat1!=0)
        mat_01[pos1[:,0], pos1[:,1]] = 1
        return np.array(mat_01, dtype='float32')


class GCNModel(object):
    def __init__(self, features, labels, learning_rate, num_classes, mask, support, scale_num, h):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.classlayers = []
        self.labels = labels
        self.inputs = features
        self.scale_num = scale_num
        self.loss = 0
        self.support = support
        self.concat_vec = []
        self.outputs = None
        self.num_classes = num_classes
        self.hidden1 = h 
        self.mask = mask
        self.build()
        
    def _build(self):
        for scale_idx in range(self.scale_num):
            activations = []
            activations.append(self.inputs) # feature矩阵 (949, 200)
            self.classlayers.append(GraphConvolution(act = tf.nn.softplus,
                                      input_dim = np.shape(self.inputs)[1],
                                      output_dim = self.hidden1,
                                      support = self.support[scale_idx], #(949, 949)
                                      bias = True
                                      ))   
            layer = self.classlayers[-1]  # 最新的GCN层     
            hidden = layer(activations[-1]) #(949, 20)

            self.classlayers.append(GraphConvolution(act = lambda x:x,
                                      input_dim = self.hidden1,
                                      output_dim = self.num_classes,
                                      support = self.support[scale_idx],
                                      bias = True
                                      ))   
            layer = self.classlayers[-1]
            out_classes = layer(hidden)

            if scale_idx == 0:
                self.concat_vec = out_classes
            else:
                self.concat_vec += out_classes

            
    def build(self):
        self._build()
        self.outputs = self.concat_vec
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)        
        
    def _loss(self):
        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.labels, self.mask)

        
        
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels, self.mask)
    
    # 不为0的地方全为1
    def Get01Mat(self, mat1):
        [r, c] = np.shape(mat1)
        mat_01 = np.zeros([r, c])
        pos1 = np.argwhere(mat1!=0)
        mat_01[pos1[:,0], pos1[:,1]] = 1
        return np.array(mat_01, dtype='float32')


class MyMDGCNModel(object):
    def __init__(self, features, labels, learning_rate, num_classes, mask, scale_num, h, support_dynamic_my): #注意h，隐藏层的神经元个数
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # support就是几个scale的DAD [1]、[2]、[3]
        self.classlayers = []
        self.labels = labels
        self.inputs = features
        self.scale_num = scale_num
        self.loss = 0
        self.concat_vec = []
        self.outputs = None
        self.num_classes = num_classes
        self.hidden1 = h 
        self.mask = mask
        self.support_dynamic_my = support_dynamic_my
        self.build()
        
    def _build(self):
        # support_dynamic_my = self.support_dynamic_my
        activations = []
        activations.append(self.inputs) # feature矩阵 (949, 200)
        self.classlayers.append(GraphConvolution(act = tf.nn.softplus,
                                    input_dim = np.shape(self.inputs)[1],
                                    output_dim = self.hidden1,
                                    # support = list(np.squeeze(np.array(self.support_dynamic_my[-1], dtype='float32'))),#(949, 949) #需要将[1, 949, 949], but wanted [1]
                                    support = list(self.support_dynamic_my[-1]),

                                    bias = True,
                                    isnorm = False 
                                    ))  
        layer = self.classlayers[-1]  # 最新的GCN层     
        hidden = layer(activations[-1]) #(949, 25)
        activations.append(hidden) 
                    
        support_dynamic = tf.exp(-0.02*tf.matmul(hidden, tf.transpose(hidden))) #(949, 949)
        # support_dynamic = 0.1*support_dynamic*self.Get01Mat(self.support) + self.support 
        # support_dynamic_1 = tf.matmul(tf.matmul(self.support, support_dynamic), tf.transpose(self.support)) 
        self.support_dynamic_my.append(support_dynamic)
        self.classlayers.append(GraphConvolution(act = lambda x:x,
                                    input_dim = self.hidden1,
                                    output_dim = self.num_classes, 
                                    # support = list(np.squeeze(np.array(self.support_dynamic_my[-1], dtype='float32'))),
                                    support = list(self.support_dynamic_my[-1]),
                                    bias = True
                                    ))   
        layer = self.classlayers[-1]
        hidden = layer(activations[-1])#(949, 16)
        activations.append(hidden)
        # 总的来说，第一层的输出hidden作为第二层的feature输入，第一层的feature就是最原始的图
        # 并且第二层的参数DAD又由第一层的输出hidden得到

        support_dynamic = tf.exp(-0.02*tf.matmul(hidden, tf.transpose(hidden))) #(949, 949)
        # support_dynamic = 0.1*support_dynamic*self.Get01Mat(self.support) + self.support
        # support_dynamic_1 = tf.matmul(tf.matmul(self.support, support_dynamic), tf.transpose(self.support)) 
        self.support_dynamic_my.append(support_dynamic)


        self.concat_vec = activations[-1]

            
    def build(self):
        self._build()
        self.outputs = self.concat_vec
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)        
        
    def _loss(self):
        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.labels, self.mask)

        
        
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels, self.mask)
    
    # 不为0的地方全为1
    def Get01Mat(self, mat1):
        [r, c] = np.shape(mat1)
        mat_01 = np.zeros([r, c])
        pos1 = np.argwhere(mat1!=0)
        mat_01[pos1[:,0], pos1[:,1]] = 1
        return np.array(mat_01, dtype='float32')