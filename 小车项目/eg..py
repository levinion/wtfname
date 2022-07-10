import os
import subprocess

import tensorflow as tf

# from dellcar.parts.mycallback import MyCallback, save_final

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Convolution2D  # , BatchNormalization
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Cropping2D, Lambda, MaxPooling2D
from keras.layers import BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler


# from tensorflow.python.keras.optimizers import Adam
# from keras.backend.tensorflow_backend import set_session
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
resolution = (x_train, y_train)

class KerasCategorical(Model):

    def _restore_from_tensors(self, restored_tensors):
        pass

    def _serialize_to_tensors(self):
        pass

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        pass

    def __init__(self):
        super(KerasCategorical, self).__init__()
        self.model = default_categorical(resolution, use_smooth=False)


def default_categorical(resolution, use_smooth):
    """
    默认的网络结构

    Parameters
    ----------
    resolution: tuple
        要训练的图片的分辨率
    use_smooth: bool
        是否将横移功能加入训练

    Returns
    -------

    """
    relu = tf.keras.layers.ReLU()
    # 输入层

    img_in = Input(shape=(resolution[0], resolution[1], 3),
                   name='img_in')
    x = img_in
    initial_filters = 64
    block_list = [2, 2, 2, 2]

    # num_blocks = len(block_list)  # 共有几个block
    block_list = block_list
    out_filters = initial_filters
    x = Convolution2D(out_filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = relu(x)
    # blocks = tf.keras.models.Sequential()

    for block_id in range(len(block_list)):  # 第几个resnet block
        for layer_id in range(block_list[block_id]):  # 第几个卷积层

            if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                x = resnetblock(filters=out_filters, strides=2, residual_path=True, inputs=x)
            else:
                x = resnetblock(filters=out_filters, residual_path=False, inputs=x)
            # blocks.add(x)  # 将构建好的block加入resnet
        out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    # y = f1(x)
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    # 速度连续输出
    # 减少到1个数字，只选择最有可能的数
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)
    out_puts = [angle_out, throttle_out]
    if use_smooth:
        # 横移的分类输出
        # 将每个输入与每个输出连接起来，输出15个隐藏单元。使用Softmax给出百分比。15个类别，根据百分比0.0 - 1.0找到最佳类别
        sth_out = Dense(15, activation='softmax', name='sth_out')(x)
        out_puts.append(sth_out)
    # 定义输入输出
    model = Model(inputs=[img_in], outputs=out_puts)
    # 告知训练时用的优化器、损失函数和准确率评测标准
    loss = {'angle_out': 'categorical_crossentropy',
            'throttle_out': 'mean_absolute_error'}
    loss_weights = {'angle_out': 0.9, 'throttle_out': .01}
    if use_smooth:
        loss['sth_out'] = 'categorical_crossentropy'
        loss_weights['sth_out'] = 0.9
    model.compile(optimizer='adam',
                  loss=loss,
                  loss_weights=loss_weights)

    return model


def resnetblock(filters, inputs, strides=1, residual_path=False):
    # block结构建立
    relu = tf.keras.layers.ReLU()

    x = Convolution2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)

    x = relu(x)
    x = Convolution2D(filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if residual_path:
        down_c1_x = Convolution2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)(x)
        down_b1_x = BatchNormalization()(down_c1_x)

    x = relu(x)

    if residual_path:
        out = down_b1_x + x
    else:
        out = x
    # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
    return out




model = default_categorical(resolution, use_smooth=False)

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()
