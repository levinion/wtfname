import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from keras import Model

np.set_printoptions(threshold=np.inf)



def resnetblock(filters, strides, residual_path, inputs):
    # block结构建立

    c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
    b1 = BatchNormalization()

    a1 = Activation('relu')
    c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
    b2 = BatchNormalization()
    if residual_path:
        down_c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        down_b1 = BatchNormalization()

    a2 = Activation('relu')

    #    数据流处理

    x = c1(inputs)
    x = b1(x)
    x = a1(x)
    x = c2(x)
    y = b2(x)

    if residual_path:
        residual = down_c1(inputs)
        residual = down_b1(residual)
    else:
        residual = 0

    out = a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
    return out


def resnet18(inputs):
    # 参数设置
    initial_filters = 64
    block_list = [2, 2, 2, 2]

    num_blocks = len(block_list)  # 共有几个block
    block_list = block_list
    out_filters = initial_filters
    c1 = Conv2D(out_filters, (3, 3), strides=1, padding='same', use_bias=False)
    b1 = BatchNormalization()
    a1 = Activation('relu')
    blocks = tf.keras.models.Sequential()

    for block_id in range(len(block_list)):  # 第几个resnet block
        for layer_id in range(block_list[block_id]):  # 第几个卷积层

            if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                block = resnetblock(filters=out_filters, strides=2, residual_path=True)
            else:
                block = resnetblock(filters=out_filters, residual_path=False)
            blocks.add(block)  # 将构建好的block加入resnet
        out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
    p1 = tf.keras.layers.GlobalAveragePooling2D()
    f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    x = c1(inputs)
    x = b1(x)
    x = a1(x)
    x = blocks(x)
    x = p1(x)
    y = f1(x)
    return y
