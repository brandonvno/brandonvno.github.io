---
title: Tensorflow笔记一
date: 2019-08-13 18:47:21
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
 mooc 北京大学曹健老师课程：tensorflow笔记 第三节 Tensorflow框架 要点记录

---

## 几个概念

### 基于TensorFlow的NN：
 用张量表示数据，用计算图搭建神经网络，用回话执行计算图，优化线上的权重（参数），得到模型。

### 张量(tensor)：多维数组(列表)
 张量的维数称为**阶**
 0阶张量称为标量(scalar)，如123
 1阶张量称为向量(vector)
 2阶张量称为矩阵(matrix)
 张量可以表示0阶到n阶数组

### 计算图(Graph)：
 搭建神经网络的计算过程，只搭建，不计算

### 会话(Session)：
 执行计算图中的节点运算
 用python实现计算的语法：
 ```
 with tensorflow.Session() as sess:
    print(sess.run(y))
 ```

### 参数：是指神经元线上的权重，用标量表示，随机赋给初值
 随机赋值举例：
 `w = tf.Variable(tf.randon_normal([2,3], stddev = 2, mean = 0, seed = 1))`
 random_normal表示正态分布，stddev是标准差，mean是平均值， seed是随机种子。
 标准差，平均值，随机种子可不写，随机种子不写每次随机的值都不一样。

 另外几个常用方法：
 `tf.truncated_normal()`
 去掉过大偏离点的正态分布（随机出来的值与平均值差距超过两个标准差，则舍弃）
 `tf.random_uniform()`
 平均分布
 `tf.zeros()`
 生成全0数组，如tf.zeros([3,2], int32)
 `tf.ones()` 
 生成全1数组，如tf.ones([3,2], int32)
 `tf.fill()`
 生成定值数组，如tf.fill([3,2], 6)
 `tf.constant()`
 生成指定数组
 
---

## 神经网络实现过程
 1. 准备数据集，提取特征，作为输入喂给神经网络（Neural Network，NN）
 2. 搭建NN结构，从输入到输出（先搭建计算图，再用会话执行）
  （NN前向传播算法 -> 计算输出）
 3. 大量特征数据喂给NN，迭代优化NN参数
  （NN反向传播算法 -> 优化参数训练模型）
 4. 使用训练好的模型预测和分类

---

## 前向传播
 示例：
 !["示例"](/uploads/tensorflow_notes/image1.png "前向传播讲解示例")
 推导：
 !["推导"](/uploads/tensorflow_notes/image2.png "前向传播示例推导")
 W矩阵，前面有m个节点，后面有n个节点，则为mXn阶矩阵。
 输入不计入神经网络层，a层是第一个计算层，也是神经网络的第一层，这里是1X3阶矩阵

 说明：
 - 变量初始化、计算图节点运算都要使用会话（with结构）实现
  `with tf.Session() as sess`
 - 变量初始化
  `init_op = tf.global_variables_initializer()`
  `sess.run(init_op)`
 - 计算图节点运算: sess.run()中写入带运算节点
 `sess.run(y)`
 - 给神经网络喂数据：用tf.placeholder占位，在sess.run()中使用feed_dict喂数据
 喂一组数据：
 `x = tf.placeholder(tf.float32, shape=(1,2))`
 `sess.run(y, feed_dict={x:[[0.5, 0.6]]}`

 喂多组数据：
 ```
 x = tf.placeholder(tf.float32, shape=(None, 2))
 sess.run(y, feed_dict={x:[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]})
 ```
 shape第二个参数是模型的特征数，比如此模型有重量和体积两个特征，故为2。

代码示例1：
 ```
 #coding:utf-8
#两层简单神经网络
import tensorflow as tf

#定义输入和参数
x = tf.constant([[0.7, 0.5]])
w1 = tf.Variable(tf.random_normal([2,3], stddev = 1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev = 1, seed = 1))

#定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("result y in this file is:\n", sess.run(y))
 ```

代码示例2：
```
#coding:utf-8
#两层简单神经网络

import tensorflow as tf

#定义输入和参数
#用placeholder实现输入定义
x = tf.placeholder(tf.float32, shape=(1,2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed = 1))

#定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y in this file is:\n", sess.run(y, feed_dict = {x:[[0.7, 0.5]]}))

```

代码示例3：
```
#coding:utf-8
#两层简单神经网络(喂多组数据)

import tensorflow as tf

#定义输入和参数
#用placeholder定义输入(sess.run()喂多组数据)
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

#定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#调用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer();
    sess.run(init_op)
    print("y in this file is:\n", sess.run(y, feed_dict = {x:[[0.7, 0.5],[0.2, 0.3],[0.3, 0.4], [0.4, 0.5]]}))
    print("w1:\n", sess.run(w1))
    print("w1:\n", sess.run(w2))

```

---

## 反向传播
 反向传播是一个不断训练模型参数，在所有参数上使用梯度下降，使NN模型在训练数据上的损失函数最小

 损失函数(loss)：预测值(y)与已知答案(y_)的差距。 均方误差MSE是计算损失函数的一种方法

 均方误差MSE：MSE(y_, y) = \\(\frac{\sum_{i=1}^{n}(y-y\\_)^{2}}{n}\\)

 均方误差计算损失函数代码： 
 `loss = tf.reduce_mean(tf.square(y-y_))`
 其中y和y\_都是张量

 反向传播的训练方法：以减小loss值为优化目标
 三种训练方法：
 `train_step = tf.train.GiadientDescentOptimizer(learning_rate).minimize(loss)`
 `train_step = tr.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)`
 `train_step = tr.train.AdamOptimizer(learning_rate),minimize(loss)`
 learning_rate是指学习率，它决定每次更新的幅度，一开始可以选一个较小值，比如0.001

 反向传播代码示例：
 ```
#coding:utf_8
#反向传播过程实例
#导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455

#基于seed生成随机数
rng = np.random.RandomState(seed)
#随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
X = rng.rand(32, 2)
#从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如该和不小于1 给Y赋值0 作为输入数据集的标签
Y = [[int(x0 + x1 <1)] for (x0, x1) in X]
print("X:\n", X)
print("Y:\n", Y)

#定义神经网络的输入
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

#定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

#定义神经网络前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数
loss = tf.reduce_mean(tf.square(y-y_))
#三种反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #输出当前（未经训练）的参数取值
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")

    #训练模型
    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) %32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i%500 == 0:
            total_loss = sess.run(loss, feed_dict={x:X, y_:Y})
            print("After %d training steps, loss on all data is %g"%(i, total_loss))
    #输出训练后的参数取值
    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))

 ```

---

## 搭建神经网络的八股：准备、前向传播、反向传播、迭代

- 准备
 import 相关模块、 定义常量、 生成数据集

- 前向传播
 定义输入：特征输入x，标准答案y_
 定义参数：（一般是随机）定义第一层网络参数w1和第二程网络参数w2
 定义输出：定义计算图(Graph)，即定义第一层网络a和结果y的计算过程

- 反向传播
 定义损失函数：loss
 定义反向传播方法：train_step

- 迭代：生成会话(Session)，训练STEPS轮
```
with tf.Session() as sess
    init_op = tf.global_variables_initializer()
    sess_run(init_op)
    
    STEPS = 3000
    for i in range(STEPS):
        start = 
        end = 
        sess.run(train_step, feed_dict=)
```

---