---
title: Tensorflow笔记五
date: 2019-08-27 20:40:19
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
 mooc 北京大学曹健老师课程：tensorflow笔记 第八节 卷积神经网络 要点记录
 版本：python(3.6.6)， tensorflow(1.3.0)

---

## VGG net实现图片识别(千分类)
### 相关资料
依据论文：《VERY DEEP CONVOLUTIONAL NETWORKS FOR
LARGE-SCALE IMAGE RECOGNITION》

论文下载地址：
https://nos.netease.com/edu-lesson-pdfsrc/A259F94F8F75436E482FE1C12B6C177F-1525019670967?Signature=s48UpJM5l4ytnZXNrY9CT8zW6nFN1EfE3C53P61XViA%3D&Expires=1566898884&NOSAccessKeyId=7ba71f968e4340f1ab476ecb300190fa&download=VGGNET%E8%AE%BA%E6%96%87.pdf

代码相关文档下载地址：
https://nos.netease.com/edu-lesson-pdfsrc/991E299B808DBA0CC42F2323E3A589DE-1525019763996?Signature=iI0%2FwD%2F1MPq0FB%2Be6HNm6B3MJga45rgmjvAFF7EPkm0%3D&Expires=1566898929&NOSAccessKeyId=7ba71f968e4340f1ab476ecb300190fa&download=%E5%8A%A9%E6%95%99%E7%9A%84Tensorflow%E7%AC%94%E8%AE%B08.pdf

### 一些方法的说明

tensorflow文档：
https://tensorflow.google.cn/
https://www.tensorflow.org/

os.getcwd(): 返回当前目录

os.path.join(a,b,c,...)：拼出整个路径
如：vgg16_path = os.path.join(os.getcwd(), "vgg16.npy")可获得npy文件的完整路径

np.save("name.npy", 数组)：将某数组写入"name.npy"文件(未压缩的二进制形式，文件默认的扩展名是.npy)。

variable = np.load("name.npy", encoding = "").item():将name.npy文件读出给varibale变量。
encoding可以填"latinl", "ASCII", "bytes"，不填默认"ASCII"

.item():遍历(键值对)。

np.argsort(列表)： 对列表从小到大排序，返回列表索引值

tf.shape(a)和a.get_shape()比较：
    相同点：都可以得到tensor a的尺寸
    不同点：tf.shape()中a的数据类型可以是tensor，list，array；而a.get_shape()中a的数据类型只能是tensor，且返回的是一个元组(tuple)

tf.nn.bias_add(乘加和，bias): 把bias加到乘加和上。

tf.reshape(tensor,[n行，m列]):改变tensor的形状
tf.reshape(tensor,[-1, m列])：-1表示行跟随m列自动计算

tf.split(切谁, 怎么切, 在哪个维度切):
dimension: 输入张量的哪一个维度，如果是0就表示对第0维进行切割
num_split: 切割的数量，切割完每一份是一个列表
如：
```
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
tf.shape(split0) ==> [5, 4]
tf.shape(split1) ==> [5, 15]
tf.shape(split2) ==> [5, 11]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
tf.shape(split0) ==> [5, 10]
```
red, greed, blue = tf.split(输入, 3, 3)
TF卷积的输入shape为：[batch, 长，宽，深]
个人理解：第0维度为batch个图片，第1维度为图片的每行，第2维度为图片的每列，第3维度为图片的每个rgb像素

tf.concat(concat_dim, values):
沿着某一个维度连接tensor
如：
```
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11,
12]]
tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
```
如果想沿着 r tensor 一新轴连结打包, , 那么可以：
`tf.concat(axis, [tf.expand_dims(t, axis) for t in tensors])`
等同于 
`tf.pack(tensors, axis=axis)`


fig = plt.figure("名字")
实例化图对象

ax = fig.add_subplot(m n k)
将画布分隔成m行n列，图像画在从左到右，从上到下的第k块

img =io.imread(图片路径)
读入图片

ax = fig.add_subplot(x,y,z)
x,y,z分别是包含几行，包含几列，当前是第几个

ax.bar(bar的个数,bar的值,每个bar的名字, bar宽度, bar颜色)
可以画出柱状图

ax.set_ylabel(字符串)
设置y轴的名字，如果字符串是中文，有时候需要在字符串前加u

ax.set_title(字符串)
设置子图名字

ax.text(文字x坐标，文字y坐标， 文字内容， ha = 'center', va = 'bottom', fontsize = 7)
ha水平对齐方式：参数可以为center、right、left
va垂直对齐方式：参数可以为center、top、bottom

ax = imshow(图)
画出子图

---
## 源代码
- app.py
 ```
#coding:utf-8
import numpy as np
#Linux服务器没有GUI的情况下使用matplotlib绘图，必须置于pyplot之前
import matplotlib
#matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt

import vgg16
import utils
from Nclasses import labels

img_path = input('Input the path and image name:')
#调用load_image()函数，对待测试的图像做一些预处理操作
img_ready = utils.load_image(img_path)

#定义一个figure画图窗口，并制定窗口的名称，也可以设置窗口的大小
fig = plt.figure(u"Top-5 预测结果")
with tf.Session() as sess:
    #定义一个维度为[1,224,224,3]，类型为float32的tensor占位符
    x = tf.placeholder(tf.float32, [1,224,224,3])
    #实例化对象
    vgg = vgg16.Vgg16()
    #调用类的成员方法forward()，并传入待测试图像，即前向传播过程
    vgg.forward(x)
    #将一个batch的数据喂入网络，得到网络的预测输出
    probability = sess.run(vgg.prob, feed_dict = {x:img_ready})


    top5 = np.argsort(probability[0])[-1:-6:-1]
    print("top5:", top5)

    #定义两个list--对应的概率值和实际标签(zebra)
    values = []
    bar_label = []

    #枚举上面取出的5个索引值
    for n,i in enumerate(top5):
        print("n:", n)
        print("i:", i)
        values.append(probability[0][i])
        bar_label.append(labels[i])
        #打印属于某个类别的概率
        print(i, ":", labels[i], "---", utils.percent(probability[0][i]))
    
    #将画布划分为1行1列，并把下图放入其中
    ax = fig.add_subplot(111)
    #bar()函数绘制柱状图，参数range(len(values))是柱子的下标，values表示柱高的列表(也就是五个预测概率值)
    #tick_label是每个柱子上显示的标签（实际对应的标签）， width是柱子的宽度，fc是柱子的颜色
    ax.bar(range(len(values)), values, tick_label = bar_label, width = 0.5, fc = 'g')
    #设置横轴标签
    ax.set_ylabel(u'probability')
    #添加标题
    ax.set_title(u'Top-5')

    for a,b in zip(range(len(values)), values):
        #在每个柱子的顶端添加对应的预测概率值，a,b表示坐标，b+0.0005表示要把文本信息放置在高于每个柱子顶端0.0005的位置。
        #center是表示文本位于顶端水平方向上的中间位置，bottom是将文本水平放置在柱子顶端垂直放心上的地段位置，fontsize是字号
        ax.text(a, b+0.0005, utils.percent(b), ha = 'center', va = 'bottom', fontsize = 7)

    #保存图片
    plt.savefig('./result.jpg')
    #弹窗展示图像，在服务器上需要将下面这行代码注释掉
    plt.show()
 ```

- vgg16.py
 ```
#!usr/bin/python
#coding:utf-8
import inspect
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

#样本RGB的平均值
VGG_MEAN = [103.939, 155.779, 123.68]

class Vgg16():
    def __init__(self, vgg16_path = None):
        if vgg16_path is None:
            #os.getcwd() 返回当前工作目录
            vgg16_path = os.path.join(os.getcwd(), "vgg16.npy")
            print(vgg16_path)
            #遍历键值对，导入模型参数
            self.data_dict = np.load(vgg16_path, encoding = 'latin1').item()
        #遍历data_dict中的每个键
        for x in self.data_dict:
            print(x)


    def conv_layer(self, x, name):
        #定义卷积运算

        #根据命名空间找到对应的卷积层的网络参数
        with tf.variable_scope(name):
            #读取该层的卷积核
            w = self.get_conv_filter(name)
            #卷积计算
            conv = tf.nn.conv2d(x,w,[1,1,1,1], padding = "SAME")
            conv_biases = self.get_bias(name)
            #加偏置，激活
            result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
            return result


    def get_conv_filter(self, name):
        #根据命名空间name从参数字典中读取对应的卷积核
        return tf.constant(self.data_dict[name][0], name = 'filter')

    def get_bias(self, name):
        #根据命名空间name从参数字典中渠道对应的偏置项
        return tf.constant(self.data_dict[name][1], name = 'biases')

    def max_pool_2x2(self, x, name):
        #定义最大池化操作
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME', name = name)

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name = "weights")

    def fc_layer(self, x, name):
        #定义全连接的前向传播计算
        with tf.variable_scope(name):
            #根据命名空间name做全连接计算
            shape = x.get_shape().as_list()
            print("fc_layer shape", shape)
            dim = 1
            #计算第1,2,3维度下的元素总和
            for i in shape[1:]:
                dim *= i
            #改变特征图的形状，也就是将得到的多维特征做拉伸操作，只在进入第六层全连接层做该操作
            x = tf.reshape(x, [-1, dim])

            #读出参数
            w = self.get_fc_weight(name)
            b = self.get_bias(name)
            #对该层输入做加权求和，再加上偏置
            result  = tf.nn.bias_add(tf.matmul(x,w), b)
            return result


    def forward(self, images):
        #plt.figure("process pictures")
        print("bulid model started")
        #获取前向传播的开始时间
        start_time = time.time()
        #逐像素乘以255.0(依据原论文所叙述的初始化步骤)
        rgb_scaled = images * 255.0

        #从GRB转换色彩通道到BGR，也可使用cv中的GRBtoBGR
        red, green, blue = tf.split(rgb_scaled, 3, 3)

        #断言，用来判断每个操作后的维度变化是否和预期一致
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]

        #逐样本减去每个通道的像素平均值，这种操作可以移除图像的平均亮度值，该方法常用在灰度图像上
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]],
            3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]



        #接下来构建VGG的16层网络(包含5段卷积，3层全连接)， 并逐层根据命名空间读取网络参数
        #第一段卷积，含有两个卷积层，后面接最大池化层，用来缩小图片尺寸
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")

        #传入命名空间的name，用来获取该层的卷积核和偏置，并做卷积运算，最后返回经过激活函数后的值
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")

        #根据传入的pooling名字对该层做相应的池化操作
        self.pool1 = self.max_pool_2x2(self.conv1_2, "pool1")


        #下面的前向传播过程与第一段同理
        # 第二段卷积，同样包含两个卷积层，一个最大池化层
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool_2x2(self.conv2_2, "pool2")

        #第三段卷积，包含三个卷积层，一个最大池化层
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool_2x2(self.conv3_3, "pool3")

        #第四段卷积，包含三个卷积层，一个最大池化层
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool_2x2(self.conv4_3, "pool4")

        #第五段卷积，包含三个卷积层，一个最大池化层
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool_2x2(self.conv5_3, "pool5")

        #第六层全连接
        #根据命名空间name做加权求和运算
        self.fc6 = self.fc_layer(self.pool5, "fc6")
        #4096是该层输出后的长度
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        #经过relu激活函数
        self.relu6 = tf.nn.relu(self.fc6)

        #第七层全连接，同上
        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        #第八层全连接
        self.fc8 = self.fc_layer(self.relu7, "fc8")
        #经过最后一层的全连接后，再做softmax分类，得到属于各类别的概率

        self.prob = tf.nn.softmax(self.fc8, name = "prob")

        #得到前向传播的结束时间
        end_time = time.time()
        print("time consuming:%f"%(end_time - start_time))

        #清空本次读取到的模型参数字典
        self.data_dict = None

 ```

- utils.py
 ```
#!/usr/bin/python
#coding:utf-8

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl


#正常显示中文标签
mpl.rcParams['font.sans-serif']=['SimHei']
#正常显示正负号
mpl.rcParams['axes.unicode_minus'] = False

def load_image(path):
    fig = plt.figure("Center and Resize")
    #根据传入的路径读入图片
    img = io.imread(path)
    #将像素归一化到[0,1]
    img = img / 255.0

    #131表示：将该画布分为一行三列，将第一个画布赋给ax0
    ax0 = fig.add_subplot(131)
    #添加子标签
    ax0.set_xlabel(u'Original Picture')
    #在此部分画布中展示img图片
    ax0.imshow(img)


    #找到图像的最短边
    short_edge = min(img.shape[:2])
    #把图像的x和h分别减去最短边，并求平均
    y = (img.shape[0] - short_edge)//2
    x = (img.shape[1] - short_edge)//2
    #取出切分出的中心图像
    crop_img = img[y:y+short_edge, x:x+short_edge]
    print(crop_img)

    #把画布分为1行3列，并将第2个画布赋给ax1
    ax1 = fig.add_subplot(132)
    #添加子标签
    ax1.set_xlabel(u'Center Picture')
    ax1.imshow(crop_img)

    #resize 成固定的imag_size
    re_img = transform.resize(crop_img, (224,224))

    ax2 = fig.add_subplot(133)
    ax2.set_xlabel(u'Resize Picture')
    ax2.imshow(re_img)
    img_ready = re_img.reshape((1,224,224,3))
    return img_ready

def percent(value):
    return '%.2f%%'%(value*100)
 ```

---

## 资源
链接：https://pan.baidu.com/s/1WWNoY-ahajm2qkcCeNNgqg
密码：52b2

---