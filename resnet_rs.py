import os,sys
import subprocess
import tensorflow as tf
from tensorflow.python import keras

from dellcar.parts.mycallback import MyCallback, save_final

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import BatchNormalization,SeparableConv2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Cropping2D, Lambda, MaxPooling2D
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.python.keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session

os.system("pip install git+https://github.com/sebastian-sz/resnet-rs-keras@main")

from resnet_rs import ResNetRS50

class KerasPilot:
    """
    这是一个训练模型的代码类，主要提供load模型和训练模型的代码
    """
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        gpu_left = False
        used_str = "nvidia-smi | awk '{print $9}' | grep MiB"
        used_process = subprocess.Popen(used_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        used_output, _ = used_process.communicate()
        gpuUsedMemoryList = str(used_output)[2:-3].replace("MiB", "").split("\\n")
        total_str = "nvidia-smi | awk '{print $11}' | grep MiB"
        total_process = subprocess.Popen(total_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        total_output, _ = total_process.communicate()
        gpuTotalMemoryList = str(total_output)[2:-3].replace("MiB", "").split("\\n")
        for i in range(len(gpuUsedMemoryList)):
            if float(gpuUsedMemoryList[i]) / float(gpuTotalMemoryList[i]) <= 0.9:
                os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % i
                gpu_left = True
                print("leftMemory: ", gpuUsedMemoryList[i], "TotalMemory: ", gpuTotalMemoryList[i])
                break
            else:
                print("No GPU left")
        if not gpu_left:
            sys.exit(1)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.25
        set_session(tf.Session(config=config))

    def load(self, model_path):
        """
        加载模型

        Parameters
        ----------
        model_path: str
            要加载的模型的路径

        Returns
        -------

        """
        self.model = load_model(model_path)

    def train(self, train_gen, val_gen,
              saved_model_path, lr=.001, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        训练模型

        Parameters
        ----------
        train_gen: list
            存放训练数据的生成器
        val_gen: list
            存放验证数据的生成器
        saved_model_path: str
            模型保存路径
        lr: float
            学习率
        epochs: int
            对数据完整训练的的次数
        steps: int
            一次完整训练的步数，受batch_size影响
        train_split: float
            训练集的比例，取值范围（0， 1）
        verbose: int
            日志显示
                | verbose = 0 为不在标准输出流输出日志信息
                | verbose = 1 为输出进度条记录
                | verbose = 2 为每个epoch输出一行记录
        min_delta: float
            增大或减小的阈值，只有大于这个部分才算作有效提升。
        patience: int
            能够接受多少个epoch内都没有improvement。
        use_early_stop: bool
            用于提前停止训练的callbacks。具体地，可以达到当训练集上的loss不在减小（即减小的程度小于某个阈值）的时候停止继续训练。

        Returns
        -------

        """
        # 每个epoch后保存模型的检查点
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # 如果验证的结果停止改善，则停止训练
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')

        # 设置学习率
        def schedule(epoch, learning_rate=lr):
            return learning_rate

        lr_scheduler = LearningRateScheduler(schedule)
        my_callback = MyCallback(model_path=saved_model_path)

        callbacks_list = [save_best, lr_scheduler, my_callback]

        if use_early_stop:
            callbacks_list.append(early_stop)

        # 为了避免数据过大造成的内存泄漏，我们用fit_generator生成器的方式进行训练
        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            # workers=8,
            # use_multiprocessing=True,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=int(steps * (1.0 - train_split) // train_split))

        save_final(model_path=saved_model_path, history=hist)
        return hist


class KerasCategorical(KerasPilot):
    """
    真正被调用的类，继承了KerasPilot类

    Parameters
    ----------
    resolution: tuple
        图片的分辨率
    use_smooth: bool
        是否训练横移数据
    args
    kwargs
    """

    def __init__(self, resolution=(480, 640), use_smooth=False, *args, **kwargs):
        """
        初始化类，传入图片的分辨率

        Parameters
        ----------
        resolution: tuple
            图片的分辨率
        use_smooth: bool
            是否训练横移数据
        args
        kwargs
        """
        super(KerasCategorical, self).__init__(*args, **kwargs)
        self.model = default_categorical(resolution, use_smooth)


def optimal_categorical(resolution, use_smooth):
    """
    优化版本的网络结构

    Parameters
    ----------
    resolution: tuple
        要训练的图片的分辨率
    use_smooth: bool
        是否将横移功能加入训练

    Returns
    -------

    """
    # 激活函数
    relu = tf.keras.layers.ReLU()
    # 输入层
    img_in = Input(shape=(resolution[0], resolution[1], 3),
                   name='img_in')
    x = img_in
    # 24个features，5 * 5卷积核，2wx2h步长
    x = SeparableConv2D(24, (3, 3), strides=(2, 2))(x)
    # 对于每个神经元做归一化处理
    x = BatchNormalization(axis=1)(x)
    x = relu(x)
    # 随机关闭20%的神经元（防止过拟合）
    x = Dropout(.2)(x)
    # 32个features，3 * 3卷积核，2wx2h步长
    x = SeparableConv2D(32, (3, 3), strides=(2, 2))(x)
    # 对于每个神经元做归一化处理
    x = BatchNormalization(axis=1)(x)
    x = relu(x)
    # 随机关闭20%的神经元（防止过拟合）
    x = Dropout(.2)(x)
    # 64个features，5 * 5卷积核，2wx2h步长，激活函数用relu
    x = SeparableConv2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    # 池化层，池化窗口 2 * 2 步长 2 * 2， 向上取整
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # 对于每个神经元做归一化处理
    x = BatchNormalization(axis=1)(x)
    x = relu(x)
    # 随机关闭20%的神经元（防止过拟合）
    x = Dropout(.2)(x)
    # 64个features，5 * 5卷积核，2wx2h步长，激活函数用relu
    x = SeparableConv2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    # 池化层，池化窗口 2 * 2 步长 2 * 2， 向上取整
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # 对于每个神经元做归一化处理
    x = BatchNormalization(axis=1)(x)
    x = relu(x)
    # 随机关闭20%的神经元（防止过拟合）
    x = Dropout(.2)(x)
    # 全连接层
    x = Flatten(name='flattened')(x)
    # 将数据分为100个特征，使用正则化防止过拟合
    x = Dense(100, activation='relu', kernel_regularizer='l2')(x)
    # 随机关闭20%的神经元（防止过拟合）
    x = Dropout(.2)(x)
    # 将数据分为50个特征，使用正则化防止过拟合
    x = Dense(50, activation='relu', kernel_regularizer='l2')(x)
    # 随机关闭20%的神经元（防止过拟合）
    x = Dropout(.2)(x)
    # 转向的分类输出
    # 将每个输入与每个输出连接起来，输出15个隐藏单元。使用Softmax给出百分比。15个类别，根据百分比0.0 - 1.0找到最佳类别
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    # 速度连续输出
    # 减少到1个数字，只选择最有可能的数
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)
    # 定义输入输出
    out_puts = [angle_out, throttle_out]
    if use_smooth:
        # 横移的分类输出
        # 将每个输入与每个输出连接起来，输出15个隐藏单元。使用Softmax给出百分比。15个类别，根据百分比0.0 - 1.0找到最佳类别
        sth_out = Dense(15, activation='softmax', name='sth_out')(x)
        out_puts.append(sth_out)
    model = Model(inputs=[img_in], outputs=out_puts)
    # 加入Adam优化器
    optimizer = Adam(lr=0.0005)
    # 告知训练时用的优化器、损失函数和准确率评测标准
    loss = {'angle_out': 'categorical_crossentropy',
            'throttle_out': 'mean_absolute_error'}
    loss_weights = {'angle_out': 0.9, 'throttle_out': .01}
    if use_smooth:
        loss['sth_out'] = 'categorical_crossentropy'
        loss_weights['sth_out'] = 0.9
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights)

    return model


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
    # 输入层
    backbone = ResNetRS50(
        weights='imagenet', 
        input_shape=(224, 224, 3),
        include_top=False
    )
    backbone.trainable=False

    img_in=Input(shape=(resolution[0], resolution[1], 3),name='img_in')
    x=img_in
    x=backbone(x)
    x=keras.layers.GlobalAveragePooling2D()(x)

    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    throttle_out = Dense(15, activation='relu', name='throttle_out')(x)
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
