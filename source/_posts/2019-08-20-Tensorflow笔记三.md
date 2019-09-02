---
title: Tensorflow笔记三
date: 2019-08-20 15:32:34
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
 mooc 北京大学曹健老师课程：tensorflow笔记 第五、六节  全连接网络基础和实践 要点记录
 版本：python(3.6.6)， tensorflow(1.3.0)

---
## 前置准备内容

### MNIST 数据集：
提供6w张 28 * 28像素点的0-9手写数字图片和标签，用于训练
提供1w张 28 * 28像素点的0-9手写数字图片和标签，用于测试
![](/uploads/tensorflow_notes/image12.png)

图片标签以一维数组形式给出，每个元素表示对应分类出现的概率。
比如6：[0,0,0,0,0,0,1,0,0,0]

导入数据集模块
`from tensorflow.examples.tutorials.mnist import imput_data`

加载数据集（数据集不存在则下载，分为train数据集和test数据集），指定数据存取路径，指定以读热码的形式存取：
`mnist = input_data.red_data_sets('./data/', one_hot = True)`
注：如果运行报错，说明是源不可达，换一个源即可，具体方法：
打开mnist.py文件，参考路径如下：
miniconda3/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py
定位到第33行左右，找到SOURCE_URL，将其值改为：
`'http://yann.lecun.com/exdb/mnist/'`
然后重新运行以上命令即可


获取子集样本数
`mnist.train.num_examples`
`mnist.validation.num_examples`
`mnist.test.num_examples`

获取标签和数据
`mnist.train.labels[0]`
`mnist.train.images[0]`

随机抽取BATCH_SIZE个数据，喂入神经网络并接收结果
```
BATCH_SIZE = 200
xs, ys = mnist.train.next_batch(BATCH_SIZE)
print("xs shape:", xs.shape)
print("ys shape:", ys.shape)
```
### 几个会用到的函数：
`tf.get_collection("")`
从集合中取全部变量，生成一个列表

`tf.add_n([])`
列表内对应元素相加

`tf.cast(x, dtype)`
把x转化为dtype类型

`tf.argmax(x, axis)`
返回最大值所在的索引号，如tf.argmax([1,0,0], 1) 返回0

`os.path.join("home", "name")`
返回home/name

字符串.split()
按指定拆分符对字符串切片，分为分割后的列表
如：`'./model/minst_model-1001'.split('/')[-1].split('-')[-1]` 返回1001

`with tf.Graph().as_default() as g:`
其内定义的节点在计算图g中

### 模型的保存和加载

保存模型：
```
saver = tf.train.Saver()
with tf.Session() as sess:
    for i in range(STEPS):
        if i % 轮数 == 0:
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)
```


实例化可还原滑动平均值的saver：
```
ema = tf.train.ExponentialMovingAverage(滑动平均基数)
ema_restore = ema.variables_to_restore()
saver = tf.train.Saver(ema_restore)
```
说明：这样所有参数通过saver对象在会话中被加载时都会被初始化为滑动平均值.


加载模型：
```
with tf.Session() as sess:
    # 加载ckpt，如果saver()设置了滑动平均值，在这里将每个参数设置为其对应的滑动平均值
    ckpt = tf.train.get_chekpoint_state(存储路径)
    # 判断ckpt是否成功加载并有模型
    if ckpt and ckpt.model_checkpoint_path:
        #恢复模型到当前会话
        saver.restore(sess, ckpt.model_checkpoint_path)
```

准确率计算方法：
```
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
说明：这里的y，是一个BATCH * n的二维数组，BATCH是一轮训练喂入的数据个数，n是训练的轮数。 argmax()第二个参数为1，是指只在第一个维度寻找最大值的索引号。取得的结果是在所有n个长度为BATCH的一维数组中，最大的一个数组的索引号。
accuracy计算得所有数据准确率的平均值，就是NN在这组数据上的准确率。

---

## MNIST获得训练过程中的准确率，代码示例：
mnist_forward.py
```
#coding:utf-8
import tensorflow as tf

# 每张图分辨率是28×28,共784个像素点
INPUT_NODE = 784
#输出是0-9这十个数字的概率
OUTPUT_NODE = 10
#隐藏层节点个数
LAYER1_NODE = 500

def get_weight(shape, regularizer):
    #利用正态分布产生随机值，标准差和均值需要自己设置
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    #正则化
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) +b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
```


mnist_backward.py
```
#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward as fw
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

def backward(mnist):
    x = tf.placeholder(tf.float32, [None, fw.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, fw.OUTPUT_NODE])
    y = fw.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True    
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x:xs, y_:ys})
            if i % 1000 == 0:
                print("After %d training steps, loss on training batch is%g."%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()
```


mnist_test.py
```
#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward as fw
import mnist_backward as bw
TEST_INTERVAL_SECS = 5

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, fw.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, fw.OUTPUT_NODE])
        y = fw.forward(x, None)

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
                    accuracy_score = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
                    print("After %s training step, test accuracy = %g"%(global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                time.sleep(TEST_INTERVAL_SECS)
def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()
```
运行mnist_backward.py，然后同时运行mnist_test.py
可以查看loss值的变化和准确率的变化


在mnist_backward.py46行处，初始化所有变量之后增加断点续训代码，即可实现断点续训：
```
#新增代码：从保存的数据中加载训练结果，实现断点续练
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
```

---

## 实践
要让程序实现特定应用，我们需要解决以下两个问题：
如何实现对输入的图片做出正确的预测？
如何制作数据集，实现特定应用？

- 如何实现对输入图片的正确预测

 在上一个例程中，我们用三个文件实现了前向传播、反向传播和测试准确率三个功能。
 下面这段代码在上面例程三个.py文件的基础上，实现输入图片，输出预测值。

 mnist_app.py
 ```
 #coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward as bw
import mnist_forward as fw

def restore_model(testPicArr):
    #默认创建一个图，在该图中进行以下操作
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, fw.INPUT_NODE])
        y = fw.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(bw.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(bw.MODEL_SAVE_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1

def pre_pic(picName):
    img = Image.open(picName)
    # 用消除锯齿的方法重新给图片设定大小
    reIm = img.resize((28,28), Image.ANTIALIAS)
    # 把原始图片转化成灰度图，并以矩阵的形式存到变量里
    im_arr = np.array(reIm.convert('L'))
    #模型要求黑底白字且每个像素点在0-1取值，输入是白底黑字取值0-255，所以需要进行反色和降值处理
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255- im_arr[i][j]
            if(im_arr[i][j]<threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready

def application():
    testNum = input("input the num of test pictures:")
    for i in range(int(testNum)):
        testPic = input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("The prediction number is: ", preValue)

def main():
    application()


if __name__ == '__main__':
    main()

 ```

 测试图片下载地址：https://github.com/cj0012/AI-Practice-Tensorflow-Notes/blob/master/num.zip
 注意：由于此网络只有两层，比较简单，自己制作图片识别率可能没有那么高。

- 如何制作数据集，实现特定应用？
 制作数据集可以使用二进制文件：tfrecords，可以先将图片和标签制作成该格式的文件，然后使用tfrecords进行数据读取，会提高内存利用率

 用tf.train.Example的协议存储训练数据。训练数据的特征用键值对的形式表示。
 如：'img_raw':值  'label':值 值是Byteslist/FloatList/Int64List

 用SerializeToString()把数据序列化成字符串存储

 生成tfrecords文件代码：
 ```
#新建一个writer
writer = tf.python_io.TFRecordWriter(tfRecordName)
#把每张图片和标签封装到example中
for 循环遍历每张图片和标签：
    example = tr.train.Example(features = tf.train.Features(feature = {
        'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw])),
        'label':tf.train.Feature(int64_list=tf.train.Int64List(value = labels))
        }))
    #把example序列化
    writer.write(example.SerializeToString())
writer.close()
 ```
 解析tfrecords 文件
 ```
#新建一个文件队列
filename_queue = tf.train.string_input_producer([tfRecord_path])
#新建一个reader
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_singel_example(serialized_example, features = {
    'img_raw':tf.FixedLenFeature([], tf.string),
    'label':tf.FixedLenFeature([10], tf.int64)
    })
img = tf.decode_raw(features['img_raw', tf.uint8])
img.set_shape([784])
img = tf.cast(img, tf.float32)*(1.0/255)
label = tf.cast(features['label'], tf.float32)
 ```

 下面将用代码展示如何实现数据集生成和读取。
 下面的代码仍然解决手写数字识别问题。其中
 mnist_forward.py和mnist_app.py与之前的代码完全相同
 mnist_backward.py和mnist_test.py需要更改图片和标签获取的方式
 新增了数据集生成读取文件: mnist_generateds.py




 mnist_forward.py
 ```
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape))  
    return b
    
def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
 ```




 mnist_backward.py
 ```
#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward as fw
import os
import mnist_generateds as gd

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"
train_num_examples = 60000

def backward():
    x = tf.placeholder(tf.float32, [None, fw.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, fw.OUTPUT_NODE])
    y = fw.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    #1 更改代码
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True    
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver()
    #2 新增代码：
    img_batch, label_batch = gd.get_tfrecord(BATCH_SIZE, isTrain=True)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #3.新增代码：从保存的数据中加载训练结果，实现断点续练
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        #4.新增代码：开启线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)

        for i in range(STEPS):
            #5.改动代码：
            xs, ys = sess.run([img_batch, label_batch])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x:xs, y_:ys})
            if i % 1000 == 0:
                print("After %d training steps, loss on training batch is%g."%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        #6新增代码：关闭线程协调器
        coord.request_stop()
        coord.join(threads)

def main():
    backward()

if __name__ == '__main__':
    main()
 ```




 mnist_test.py
 ```
#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward as fw
import mnist_backward as bw
import mnist_generateds as gd

TEST_INTERVAL_SECS = 5
TEST_NUM = 10000

def test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, fw.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, fw.OUTPUT_NODE])
        y = fw.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(bw.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #新增代码1：
        img_batch, label_batch = gd.get_tfrecord(TEST_NUM, True)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(bw.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    #2.新增代码：开启线程协调器
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

                    #3.改动代码：
                    xs, ys = sess.run([img_batch, label_batch])

                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                    print("After %s training step, test accuracy = %g"%(global_step, accuracy_score))

                    #4新增代码：关闭线程协调器
                    coord.request_stop()
                    coord.join(threads)
                else:
                    print("No checkpoint file found")
                time.sleep(TEST_INTERVAL_SECS)
def main():
    test()

if __name__ == '__main__':
    main()

 ```




 mnist_app.py
 ```
#coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward as bw
import mnist_forward as fw

def restore_model(testPicArr):
    #默认创建一个图，在该图中进行以下操作
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, fw.INPUT_NODE])
        y = fw.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(bw.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(bw.MODEL_SAVE_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1

def pre_pic(picName):
    img = Image.open(picName)
    # 用消除锯齿的方法重新给图片设定大小
    reIm = img.resize((28,28), Image.ANTIALIAS)
    # 把原始图片转化成灰度图，并以矩阵的形式存到变量里
    im_arr = np.array(reIm.convert('L'))
    #模型要求黑底白字且每个像素点在0-1取值，输入是白底黑字取值0-255，所以需要进行反色和降值处理
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255- im_arr[i][j]
            if(im_arr[i][j]<threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready

def application():
    testNum = input("input the num of test pictures:")
    for i in range(int(testNum)):
        testPic = input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("The prediction number is: ", preValue)

def main():
    application()


if __name__ == '__main__':
    main()
 ```




 mnist_generateds.py
 ```
#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path = './mnist_data_jpg/mnist_train_jpg_60000'
label_train_path = './mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train = './data/mnist_train.tfrecords'
image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test = './data/mnist_test.tfrecords'
data_path = './data'
resize_height = 28
resize_width = 28

def write_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    #用来计数
    num_pic = 0
    #以读方式打开label文件，lebel文件是个txt文件，每行由图片名和标签组成，中间用空格隔开
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        #用空格分隔每行的内容
        value = content.split()
        #取第一个元素，即为图片名，组成路径
        img_path = image_path + value[0]
        #打开图片并转化成二进制数组
        img = Image.open(img_path)
        img_raw = img.tobytes()
        #初始化labels所有元素为0
        labels = [0]*10
        #把图片对应的标签位赋值为1
        labels[int(value[1])] = 1

        #创建example
        example = tf.train.Example(features = tf.train.Features(feature = {
            'img_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = labels))
            }))
        writer.write(example.SerializeToString())
        num_pic += 1
        print("the number of picture", num_pic)
    writer.close()
    print("write tfrecord successfully")


def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("The directory was created successfully")
    else:
        print("Directory already exists")
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle = True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        'label' : tf.FixedLenFeature([10], tf.int64),
        'img_raw' : tf.FixedLenFeature([], tf.string)
        })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1.0/255)
    label = tf.cast(features['label'], tf.float32)
    return img, label

def get_tfrecord(num, isTrain = True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label],
        batch_size = num,
        num_threads = 2, 
        capacity = 1000,
        min_after_dequeue = 700)
    return img_batch, label_batch

def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()
 ```

数据集下载：
https://github.com/cj0012/AI-Practice-Tensorflow-Notes/blob/master/fc4.zip
https://github.com/cj0012/AI-Practice-Tensorflow-Notes/blob/master/fc4.z01
https://github.com/cj0012/AI-Practice-Tensorflow-Notes/blob/master/fc4.z02
https://github.com/cj0012/AI-Practice-Tensorflow-Notes/blob/master/fc4.z03


---



---
