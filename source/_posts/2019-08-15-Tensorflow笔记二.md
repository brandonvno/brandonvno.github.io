---
title: Tensorflow笔记二
date: 2019-08-15 11:01:03
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
 mooc 北京大学曹健老师课程：tensorflow笔记 第四节 神经网络优化 要点记录

---

## 模型、激活函数、NN复杂度
 常用模型
!["模型说明"](/uploads/tensorflow_notes/image3.png "模型说明")

 三个常用的激活函数
!["激活函数"](/uploads/tensorflow_notes/image4.png "激活函数")

 NN复杂度多用NN层数和NN参数的个数表示
 层数 = 隐藏层的个数+1个输出层（下图层数为2）
 总参数 = 总W+总b（下图总参数为：3*4+4 + 4*2+2 = 26）
!["NN复杂度辅图"](/uploads/tensorflow_notes/image5.png "NN复杂度辅图")

---

## 损失函数(loss)：预测值(y)与已知答案(y_)的差距
 NN优化目标：loss最小。

 三种loss计算：均方误差mse(mean squared error)、 自定义、 交叉熵ce(Cross Entropy)
 
### 均方误差MSE：MSE(y_, y) = \\(\frac{\sum_{i=1}^{n}(y-y\\_)^{2}}{n}\\)

 均方误差计算损失函数代码： 
 `loss = tf.reduce_mean(tf.square(y-y_))`

 下面举了一个栗子来说明损失函数：
 预测酸奶日销量y。x1、x2是影响日销量的因素。
 建模前，应预先采集的数据有：每日x1、x2和销量y_（即已知答案，最佳情况：产量=销量）
 拟造数据集X，Y_：y_= x1 + x2  噪声：-0.05 ~ +0.05  拟合可以预测销量的函数。
 示例代码
    ```
    #coding:utf-8
    #假设预测多了与预测少了结果一样
    #导入模块，生成数据集
    import tensorflow as tf
    import numpy as np
    BATCH_SIZE = 8
    SEED = 23455

    rdm = np.random.RandomState(SEED)
    X = rdm.rand(32, 2)
    Y_ = [[x1+x2+(rdm.rand()/10-0.05)] for (x1, x2) in X]

    #定义神经网络的输入，参数和输出，定义前向传播过程。
    x = tf.placeholder(tf.float32, shape = (None, 2))
    y_ = tf.placeholder(tf.float32, shape = (None, 1))

    w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
    y = tf.matmul(x, w1)

    #定义损失函数及反向传播方法。
    #定义损失函数为MSE 反向传播方法为梯度下降
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

    #生成会话，训练STEPS轮
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        STEPS = 20000
        for i in range(STEPS):
            start = (i*BATCH_SIZE)%32
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict = {x:X[start:end], y_:Y_[start:end]})
            if i%500 == 0:
                print("After %d training steps, w1 is:"%(i))
                print(sess.run(w1), "\n")
        print("final w1 is:\n", sess.run(w1))
    ```
运行以上代码最终结果为：[[0.98019385], [1.0159807 ]]

### 自定义损失函数：
接着上一个例子说，如果预测商品销量的时候，预测多了，损失成本，预测少了损失利润
如果利润≠成本，则mse产生的loss无法将利益最大化。这时候我们可以使用自定义损失函数

自定义损失函数  loss(y_, y) = \\(\sum_{n}f(y,y\\_)\\)
  ![](/uploads/tensorflow_notes/image6.png "")

自定义损失函数示例
```
#coding:utf-8
#酸奶成本1元，利润9元
#预测少了损失大，所以应该避免预测少。
#导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST = 1
PROFIT = 9

rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y_ = [[x1+x2+(rdm.rand()/10-0.05)] for (x1, x2) in X]

#定义神经网络的输入，参数和输出，定义前向传播过程。
x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))

w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x, w1)

#定义损失函数及反向传播方法。
#定义损失函数为MSE 反向传播方法为梯度下降
loss_mse = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*COST, (y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

#生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict = {x:X[start:end], y_:Y_[start:end]})
        if i%500 == 0:
            print("After %d training steps, w1 is:"%(i))
            print(sess.run(w1), "\n")
    print("final w1 is:\n", sess.run(w1))
```
以上代码只添加了成本和利润两个参数，修改了损失函数。
代码运行最终结果为：[[1.020171 ], [1.0425103]] ，可见w1的两个参数都比原来大了，神经网络在尽量多的预测。

将成本和利润的值互换之后，运行结果为：[[0.9661967 ], [0.97694933]]， 可见神经网络在尽量少的预测。

### 交叉熵ce(Cross Entropy)：表征两个概率分布之间的距离

 H(y_, y) = \\(-\sum\\) y_*log y

 举例说明：二分类 已知答案y_= (1, 0), 预测y1 = (0.6, 0.4)  y2 = (0.8, 0.2) 哪个更接近标准答案？
 H1((1,0), (0.6, 0.4)) = -(1*log0.6 + 0*log0.4) ≈ -(-0.222+0) = 0.222
 H2((1,0), (0.8, 0.2)) = -(1*log0.8 + 0*log0.2) ≈ -(-0.097+0) = 0.097
 所以y2预测更准
 `cem = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-12, 1.0)))`

 n分类的n个输出（y1,y2,...yn）通过使用softmax()函数，便满足了概率分布要求：\\(\forall P(X=x)\in [0,1] 且\sum_{x}P(X=x)=1\\)
 softmax(yi) = \\(\frac{e^{y_{i}}}{\sum_{j=1}^{n}e^{y_{j}}}\\)
 
 tensorflow中代码实现使输出经过softmax函数处理后得到满足概率分布要求的结果，再与标准答案求交叉熵：
 ```
 ce = tf.nn.spare_softmax_cross_entropy_with_logits(logits=y, labels = tf.argmax(y_, 1))
 cem = tf.reduce_mean(ce)
 ```

---

## 学习率(learning_rate)：每次参数更新的幅度


---
