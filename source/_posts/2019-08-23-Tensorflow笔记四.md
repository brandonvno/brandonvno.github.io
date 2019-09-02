---
title: Tensorflow笔记四
date: 2019-08-23 16:02:25
tags: 
- 人工智能
- 深度学习
- Tensorflow
categories:
- 人工智能
- 机器学习
- 深度学习
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


## 概述
 mooc 北京大学曹健老师课程：tensorflow笔记 第七节  卷积神经网络 要点记录
 版本：python(3.6.6)， tensorflow(1.3.0)

---

## 卷积神经网络相关概念和原理：

- 全连接NN：每个神经元与前后相邻层的每一个神经元都有连接关系，输入是特征，输出为预测结果。
 参数个数：\\(\\sum(前层 x 后层 + 后层)\\)

 大多数时候，我们需要对彩色图片，也就是红绿蓝三通道图片进行处理，相比手写数字识别示例中的单通道有更多参数需要优化，而待优化的参数过多会导致模型过拟合。

 所以实际应用中会先对原始图像进行特征提取，再把提取到的特征喂给全连接网络。

- 卷积(convolutional)
 卷积就是一种有效提取图像特征的方法。
 一般会用一个正方形的卷积核，遍历图片上的每个点。图片区域内，相对应的每一个像素值，乘以卷积核内相对应点的权重，求和，再加上偏置。
 输出图片边长 = (输入图片边长 - 卷积核长 +1 ) / 步长
 如上例：边长= (5-3 +1)/1 = 3
 单次计算过程如下图所示：
 ![](/uploads/tensorflow_notes/image13.png )

 滑动计算过程如下图所示：
 ![](/uploads/tensorflow_notes/image14.gif )

 有时候可以在输入图片外围一圈进行全零填充(padding)， 使得输出图片边长跟输入图片边长相等。
 ![](/uploads/tensorflow_notes/image15.png )

 tensorflow中计算卷积(三通道示例)：
 ![](/uploads/tensorflow_notes/image16.png)


- 池化(Pooling)
 使用卷积处理的特征数量有时候仍然巨大，因此需要进一步减少特征数量。
 池化用于减少特征数量，池化分为最大值池化和均值池化。
 最大值池化可提取图片纹理，均值池化可保留背景特征。
 下图是最大值池化和均值池化的图示：
 ![](/uploads/tensorflow_notes/image17.png)

 tensorflow中计算池化：
 ![](/uploads/tensorflow_notes/image18.png)

- 舍弃(Dropout)
 有时候为了进一步简化计算，在神经网络**训练**过程中，将一部分神经元按照一定概率从神经网络中舍弃。使用时被舍弃的神经元恢复链接。

 tensorflow中设置dropout
 `tf.nn.dropout(上层输出，暂时舍弃的概率)`
 训练过程中，指定概率的神经元被随机置零，置零的神经元不参与当前轮的参数优化
 ```
 if train:
    输出 = tf.nn.dropout(上层输出， 暂时舍弃的概率)
 ```
 实际应用中常常在前向前向传播构建神经网络时使用dropout来减小过拟合，来加快模型的训练速度。dropout一般在全连接网络中使用

- 卷积神经网络：借助卷积核(kernel)提取特征后，送入全连接网络。
 CNN模型主要模块：
 1.卷积(Convolutional)
 2.激活(Activation)
 3.池化(Pooling)
 4.全连接(FC)
 其中1-3是高层次抽象特征，精简特征点的过程


 CNN模型的发展史：
 Lenet-5 -> AlexNet -> VGGNet -> GoogleNet -> ResNet -> ...

---

## Lenet-5模型实现手写数字识别
原始模型：
![](/uploads/tensorflow_notes/image19.png)

对应mnist数据集后修改的模型：
![](/uploads/tensorflow_notes/image20.png)

使用lenet-5模型实现手写数字识别代码示例：

mnist_lenet5_forward.py
```
#coding:utf-8
import tensorflow as tf
IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE  = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512   #NN第一层网络有512个神经元
OUTPUT_NODE = 10    #NN第二层网络有10个神经元，对应10分类输出

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x, w):
    #x是输入描述，一维张量的四个参数分别是batch_size， 行分辨率，列分辨率，通道数
    #w是卷积核描述，一维张量四个参数分别是行分辨率，列分辨率，通道数，核个数
    #strides是核滑动步长，一维张量第二个和第三个参数代表行和列步长，其他两个参数固定
    #padding零填充，'SAME'代表使用0填充
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    #x是输入描述，同上
    #ksize是池化核描述，第1和第4个参数固定为1，第2第3个参数代表大小
    #strides是滑动步长描述，同上
    #padding零填充，同上
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    pool_shape = pool2.get_shape().as_list()
    #计算4维张量每个第2,3,4维元素总和，即长度宽度深度乘积,得到所有特征点的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #reshape为2维张量，第一维不变(为batch)，第二维元素数改为nodes
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])


    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train: fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y
```

mnist_lenet5_backward.py
```
#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward as fw
import os
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

def backward(mnist):
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        fw.IMAGE_SIZE,
        fw.IMAGE_SIZE,
        fw.NUM_CHANNELS
        ])
    y_ = tf.placeholder(tf.float32, [None, fw.OUTPUT_NODE])
    y = fw.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable = False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True
        )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name = 'train')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                fw.IMAGE_SIZE,
                fw.IMAGE_SIZE,
                fw.NUM_CHANNELS
                ))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:reshaped_xs, y_:ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g" %(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot = True)
    backward(mnist)

if __name__ == '__main__':
    main()
```

mnist_lenet5_test.py
```
#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward as fw
import mnist_lenet5_backward as bw
import numpy as np

TEST_INTERVAL_SECS = 5

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            mnist.test.num_examples,
            fw.IMAGE_SIZE,
            fw.IMAGE_SIZE,
            fw.NUM_CHANNELS
            ])
        y_ = tf.placeholder(tf.float32, [None, fw.OUTPUT_NODE])
        y = fw.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(bw.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(bw.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_x = np.reshape(mnist.test.images,(
                        mnist.test.num_examples,
                        fw.IMAGE_SIZE,
                        fw.IMAGE_SIZE,
                        fw.NUM_CHANNELS
                        ))
                    accuracy_score = sess.run(accuracy, feed_dict={x:reshaped_x, y_:mnist.test.labels})
                    print("after %s training step(s), test accuracy = %g" %(global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)
def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()
```

如果运行过程中出现了内存分配的异常，应适当减小BATCH

---
