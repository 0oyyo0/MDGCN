# -*- coding: utf-8 -*-
import numpy as np
from funcCNN import *
from MyGCNModel import GCNModel
from BuildSPInst_A import *
import tensorflow as tf
import time


data_name = 'IP'
num_classes = 16

learning_rate = 0.0005
epochs = 5000
img_gyh = data_name+'_gyh'
img_gt = data_name+'_gt'


Data = load_HSI_data(data_name)
model = GetInst_A(Data['useful_sp_lab'], Data[img_gyh], Data[img_gt],
                  Data['trpos'], scale=1)
sp_mean = np.array(model.sp_mean, dtype='float32')
sp_label = np.array(model.sp_label, dtype='float32')

trmask = np.matlib.reshape(np.array(model.trmask, dtype='bool'), [
                           np.shape(model.trmask)[0], 1])

temask = np.matlib.reshape(np.array(model.temask, dtype='bool'), [
                           np.shape(model.trmask)[0], 1])
sp_support = []


for A_x in model.sp_A:
    sp_A = np.array(A_x, dtype='float32')
    sp_support.append(np.array(model.CalSupport(sp_A), dtype='float32'))

with tf.Session() as sess:
    mask = tf.placeholder("int32", [None, 1])
    labels = tf.placeholder("float", [None, num_classes])

    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)  

    model = GCNModel(features=sp_mean, labels=sp_label, learning_rate=learning_rate,
                     num_classes=num_classes, mask=mask, support=sp_support, scale_num=len(model.sp_A), h=25)

    sess.run(tf.global_variables_initializer())

    # Train
    for epoch in range(epochs):
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict={labels: sp_label,
                        mask: trmask})
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]))
    print("Optimization Finished!")

    # Testing
    t_test = time.time()
    outs_val = sess.run([model.loss, model.accuracy],
                        feed_dict={labels: sp_label, mask: temask})
    test_cost, test_acc, test_duration = outs_val[0], outs_val[1], (time.time(
    ) - t_test)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    # Pixel-wise accuracy
    outputs = sess.run(model.outputs)  

    pixel_wise_pred = np.argmax(outputs, axis=1)

    # Generating results
    pred_mat = AssignLabels(Data['useful_sp_lab'], np.argmax(
        sp_label, axis=1), pixel_wise_pred, trmask, temask)
    # pred_mat = pixel_wise_pred

    # print("pred_mat", pred_mat)
    # print("shape of pred_mat", pred_mat.shape)

    scio.savemat('./data/pred_mat.mat', {'pred_mat': pred_mat})
    stat_res = GetExcelData(Data[img_gt], pred_mat, Data['trpos'])
    scio.savemat('./data/stat_res.mat', {'stat_res': stat_res})
    print(stat_res)
    print("finished!")
