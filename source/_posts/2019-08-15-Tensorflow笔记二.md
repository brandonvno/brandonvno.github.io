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
 版本：python(3.6.6)， tensorflow(1.3.0)

---

## 模型、激活函数、NN复杂度
 常用模型
!["模型说明"](/uploads/tensorflow_notes/image3.png "模型说明")

 三个常用的激活函数
!["激活函数"](/uploads/tensorflow_notes/image4.png "激活函数")

 NN复杂度多用NN层数和NN参数的个数表示
 层数 = 隐藏层的个数+1个输出层（下图层数为2）
 总参数 = 总W+总b（下图总参数为：3 * 4 + 4 + 4 *2 + 2 = 26）
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
 H1((1,0), (0.6, 0.4)) = -(1 * log0.6 + 0 * log0.4) ≈ -(-0.222+0) = 0.222
 H2((1,0), (0.8, 0.2)) = -(1 * log0.8 + 0 * log0.2) ≈ -(-0.097+0) = 0.097
 所以y2预测更准
 `cem = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-12, 1.0)))`

 n分类的n个输出（y1,y2,...yn）通过使用softmax()函数，便满足了概率分布要求：\\(\forall P(X=x)\in [0,1] 且\sum_{x}P(X=x)=1\\)
 softmax(yi) = \\(\frac{e^{y_{i}}}{\sum_{j=1}^{n}e^{y_{j}}}\\)
 
 tensorflow中代码实现使输出经过softmax函数处理后得到满足概率分布要求的结果，再与标准答案求交叉熵：
 ```
 ce = tf.nn.spare_softmax_cross_entropy_with_logits(logits=y, labels = tf.argmax(y_, 1))
 cem = tf.reduce_mean(ce)
 ```
 spare_softmax_cross_entropy_with_logits()的一些说明：
 logits参数是经softmax()处理前的数据，即神经网络训练得到的数据y。
 argmax(y_, 1)是获得y_的第2个维度的张量中最大值的索引
 labels参数是y对应的原始数据x的标签的整数列表，类似[2,3,6,4]，个人瞎邒推测spare...函数将整数列表转换成了one-hot即类似[0,0,1,0]这样的张量(其实也就是y_的one-hot)。
 spare_softmax_cross_entropy_with_logits()首先用softmax()处理logits，然后计算labels和处理后的logits的交叉熵。

---

## 学习率(learning_rate)：每次参数更新的幅度

![](/uploads/tensorflow_notes/image7.png "")

- 学习率对传播过程影响示例代码
```
 #conding:utf-8
#设定损失函数 loss = (w+1)^2  ,令w初值为5，反向传播就是求最优w，即求最小loss对应的w值
import tensorflow as tf
#定义带优化参数w，初始值赋为5
w = tf.Variable(tf.constant(5, dtype = tf.float32))
#定义损失函数loss
loss =  tf.square(w+1)
#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps: w is %f, loss is %f."%(i, w_val, loss_val))
```
 更改学习率0.2为1，观察学习情况；更改学习率为0.001，再观察情况。可以直观的看到学习率的影响。

- 学习率设置多少合适？
由上例代码可知，学习率大了（比如设置为1）可能会导致震荡不收敛，学习率小了（比如设置0.001）收敛速度慢，所以可以考虑动态学习率。

- 指数衰减学习率
learning_rate = LEARNING_RATE_BASE * LEARNING_RATE_DECAY ^ (global_step/LEARNING_RATE_STEP)

 LEARNING_RATE_BASE 指学习率初始值
 LEARNING_RATE_DECAY 指学习率衰减率(一般取0-1，开区间)
 global_step 指运行的总轮数
 LEARNING_RATE_STEP 指多少轮更新一次学习率，计算方式为：总样本数/BATCH_SIZE

 函数代码：
 ```
 global_step = tf.Variable(0, trainable = false)  #记录当前运行轮数的计数器，trainable为False即标注为此参数不可训练
 learning_rate = tf.train.exponential_decay(
     LEARNING_RATE_BASE,
     global_step,
     LEARNING_RATE_STEP,
     LEARNING_RATE_DECAY,
     staircase = True
     )
     #staircase 取True时, global_step/LEARNING_RATE_STEP取整数，学习率阶梯型衰减，取False时，学习率下降沿一条平滑曲线
 ```

 指数衰减学习率代码示例：

 ```
 #coding:utf-8
#设损失函数loss = (w+1)^2, 令w初始值是常数10,，反向传播就是求最优w，即求最小loss对应的w值
#使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得更有收敛度。

import tensorflow as tf

LEARNING_RATE_BASE = 0.1  #最初学习率
LEARNING_RATE_DECAY = 0.99 #学习率衰减率
LEARNING_RATE_STEP = 1 #喂入多少轮BATCH_SIZE后，更新一次学习率， 一般设为：总样本数/BATCH_SIZE

#运行了几轮BATCH_SIZE的计数器，初始值为0, 设为不被训练
global_step = tf.Variable(0, trainable= False)

#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP,
 LEARNING_RATE_DECAY, staircase= True)

#定义待优化参数w，初始值为0
w = tf.Variable(10, dtype=tf.float32)

#定义损失函数loss
loss = tf.square(w+1)

#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
 global_step=global_step)

#生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps: global_step is %f, w is %f, learning_rate is %f, loss is %f" 
        %(i, global_step_val, w_val, learning_rate_val, loss_val))


 ```

---

## 滑动平均（影子值）：记录了每个参数一段时间内过往值的平均，增加了模型的泛化性。
针对所有参数进行优化，包括所有的w和b（像是给参数加了影子，参数变化，影子缓慢跟随）
计算方法： 影子 = 衰减率 * 影子 + (1- 衰减率) * 参数       
影子初值 = 参数初值
衰减率 = min{MOVING_AVERAGE_DECAY, (1+轮数)/(10+轮数)}
MOVING_AVERAGE_DECAY是滑动平均衰减率，是一个超参数

滑动平均计算过程举例：
![](/uploads/tensorflow_notes/image8.png "")

滑动平均计算常用代码：
定义滑动平均参数：
`ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)`

求所有待优化的参数的滑动平均值
`ema_op = ema.apply(tf.trainable_variables())`
ema.apply() 可以求指定参数的滑动平均值
tf.trainable_variables()可以将所有待优化的参数汇总成一个列表

常用以下代码将训练过程和计算滑动平均值绑定成一个训练节点：
 ```
 with tf.control_dependencies([train_step, ema_op]):
     train_op = tf.no_op(name = 'train')
 ```
 control_dependencies作用是当你运行train_op时会先运行train_step和ema_op，即设置train_op的依赖。
 对tf.no_op的解释引用stackflow上的一句话：
 >As the documentation says, tf.no_op() does nothing. However, when you create a tf.no_op() inside a with tf.control_dependencies([x, y, z]): block, the op will gain control dependencies on ops x, y, and z. Therefore it can be used to group together a set of side effecting ops, and give you a single op to pass to sess.run() in order to run all of them in a single step.



查看某参数的滑动平均值：
`ema.average(参数名)`


示例代码：
```
#coding:utf-8
import tensorflow as tf

#1. 定义变量及滑动平均类
w1 = tf.Variable(0, dtype = tf.float32)
#定义一个32位浮点变量， 初始值喂0.0, 这个代码就是不断更新w1参数，优化w1参数，滑动平均做了w1的影子
w1 = tf.Variable(0, dtype=tf.float32)
#定义num_updates (NN的迭代轮数), 初始值为0, 不可被优化（训练）
global_step = tf.Variable(0, trainable = False)
#实例化滑动平均类，给衰减率为0.99, 当前轮数global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#ema.apply()括号里的内容是更新列表，每次运行sess.run(ema_op)时， 对更新列表中的元素求滑动平均值
#在实际应用中会使用tf.trainable_varibales()自动将所有待训练的参数汇总喂列表
#ema_op = ema.apply([w1])
ema_op = ema.apply(tf.trainable_variables())

#2. 查看不同迭代中变量取值的变化。
with tf.Session() as sess:
    #初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #用ema.average(w1)获取w1滑动平均值（要运行多个节点，作为列表中的元素列出，写在sess.run中）
    #打印出当前参数w1和w1的滑动平均值
    print(sess.run([w1, ema.average(w1)]))

    #参数w1的值赋为1
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    #更新step和w1的值，模拟出100轮迭代后，参数w1变为10
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)

    #每次sess.run会更新一次w1的滑动平均值
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
```

---

## 正则化缓解过拟合
当模型在训练数据集上的正确率非常高，而很难对新数据集做出正确的相应时，可能是出现了过拟合现象。使用正则化可以有效缓解过拟合。
正则化在损失函数中引入模型复杂度指标，利用给w加权值，弱化了训练数据的噪声(一般不正则化b)

正则化公式：
loss = loss(y与y_)① + REGULARIZER② * loss(w)③
①指模型中所有参数的损失函数，如：交叉熵、均方误差等。
②指用超参数REGULARIZER给出参数w在总loss中的比例，即正则化的权重
③是需要正则化的参数
loss(w)有两种求法：

L1正则化：
\\(loss_{L1}(w) = \sum_{i}|w_{i}|\\)
`loss(w) = tf.contrib.layers.l1_regularizer(REGULARIZER(w))`

L2正则化：
\\((loss_{L2}(w) = \sum_{i}|w_{i}^{2}|\\)
`loss(w) = tf.contrib.layers.l2_regularizer(REGULARIZER(w))`       

将得到的loss(w)加到losses集合中：
`tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))`
获得最终损失函数：
`loss = cem + tf.add_n(tf.get_collection('losses'))`

正则化示例问题描述：
画一条线将红色点和蓝色点隔离开来
![""](/uploads/tensorflow_notes/image9.png "")

正则化示例中用到的模块matplotlib介绍：
`import matplotlib as plt`

`plt.scatter(x坐标，y坐标，c=“颜色”)`
`plt.show()`
上面两个语句用来设置散点，并显示出来：

下面的语句用来初始化网格，喂入神经网络并得到结果：

`xx, yy = np.mgrid[起:止:步长, 起:止:步长]`
隔步长取起止位置之间的所有坐标。

`grid = np.c_[xx.ravel(), yy.ravel()]`
先用ravel()将所有横纵坐标拉直，即将所有坐标组成一维张量，然后用np.c_()将两个一维张量纵向(列方向，c为column缩写)组成矩阵，即n行2列的矩阵，即n个坐标。

`probs = sess.run(y, feed_dict = {x:gird})`
将grid喂入神经网络，计算得各个点的标志(0或1)，并存到probs中

`probs = probs.reshape(xx.shape)`
将probs设置为xx的shape，即n行1列的2维张量

`plt.contour(x轴坐标值，y轴坐标值， 该点的高度， levels = [等高线的高度])`
`plt.show()`
使用上面语句把所有坐标点都设置好高度（0或者1），这样将level设置成两个高度中间值（0.5），就可以将线画出来


正则化示例代码：
```
#coding:utf-8
#导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 30
seed = 2

#基于seed产生随机数
rdm = np.random.RandomState(seed)
#随机数返回300行2列的矩阵，表示300组坐标点（x0, x1）， 作为输入数据集
X = rdm.randn(300, 2)
#从X这个300行2列的矩阵中取出1行，判断如果两个坐标的平方和小于2,给Y赋值1, 其余赋值0，作为输入数据集的标签
Y_ = [int(x0*x0 + x1*x1 <2) for (x0, x1) in X]
#遍历Y中的每个元素，1赋值'red'， 其余赋值'blue'， 这样可视化显示时人可以直观区分
Y_c = [['red' if y else 'blue'] for y in Y_]
#对数据集X和标签Y进行shape整理，第一个元素为-1，表示n行，随第二个参数计算得到， 第二个元素表示多少列，把X整理为n行2列，把Y整理为n行1列
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)
print(X)
print(Y_)
print(Y_c)

#用plt.scatter画出数据集X各行中第0列元素和第一列元素的点，即各行的(x0, x1)，用各行Y_c对应的值表示颜色(c是color的缩写)
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.show()

#定义神经网络的输入，参数和输出，定义前向传播过程
# 生成权重，输入：w的shape和正则化权重
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape = shape))
    return b
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2,11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_weight([11,1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1,w2) + b2

#定义损失函数
loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

#定义反向传播方法：不含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i%2000 == 0:
            loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
            print("After %d steps, loss is %f"%(i, loss_mse_v))
    #xx在-3到3之间以步长为0.01, yy在-3到3之间以步长0.01,生成二维网格坐标点
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    #将xx，yy拉直，并合并成一个2列的矩阵，得到一个网格坐标点的集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    #将网格坐标点喂入神经网络，probs为输出
    probs = sess.run(y, feed_dict={x:grid})
    print(probs.size)
    #将probs的shape调整成xx的样子
    probs = probs.reshape(xx.shape)
    print("w1:\n", sess.run(w1))
    print("b1:\n", sess.run(b1))
    print("w2:\n", sess.run(w2))
    print("b2:\n", sess.run(b2))
    
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()


#定义反向传播方法：含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i% 2000 == 0:
            loss_v = sess.run(loss_total, feed_dict={x:X, y_:Y_})
            print("After %d steps, loss is: %f"%(i, loss_v))
    
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x:grid})
    probs = probs.reshape(xx.shape)
    print("w1:\n", sess.run(w1))
    print("b1:\n", sess.run(b1))
    print("w2:\n", sess.run(w2))
    print("b2:\n", sess.run(b2))

plt.scatter(X[:,0], X[:,1], c = np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels = [.5])
plt.show()
```
运行即可看到不使用正则化的结果和使用正则化的结果：
未使用正则化：
![](/uploads/tensorflow_notes/image10.png "")
使用正则化：
![](/uploads/tensorflow_notes/image11.png "")

可以明显看到，使用正则化能够明显减弱噪点的影响。

---

## 搭建模块化的神经网络八股

### 前向传播
前向传播的过程就是搭建网络，设计网络结构的过程。通常用forward.py文件定义前向传播过程。在文件中通常定义三个函数：
```
def forward(x, regularizer):
 w = 
 b = 
 y = 
 return y
```

```
def get_weight(shape, regularizer):
 w = tf.Variable()
 tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
 return w
```

```
def get_bias(shape):
 b = tf.Variable()
 return b
```

### 反向传播
反向传播的过程就是训练网络，优化网络参数的过程。通常用backward.py文件定义反向传播过程。

```
def backward():
    x = tf.placeholder(  )
    y_ = tf.placeholder(  )
    y = forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable = false)
    loss = 

    #loss()函数有三种选择：
    #均方误差: 
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    #自定义：暂无代码
    #交叉熵：
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    
    #加入正则化
    loss = 三种损失函数之一 + tf.add_n(tf.get_collection('losses'))

    #使用指数衰减学习率：
    learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    数据集总样本数/BATCH_SIZE,
    LEARNING_RATE_DECAY,
    staircase=True
    )


    #定义反向传播训练过程：
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

    #计算滑动平均：
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name = 'train')

    #with结构初始化所有参数并开始训练
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            sess.run(train_step, feed_dict={x:, y_:})
            if i%轮数 == 0:
                print

if __name__ == '__main__'
    backward()
```
### 用模块化思想实现正则化示例的代码：
generateds.py
```
#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

seed = 2
def generateds():
    #基于seed产生随机数
    rdm = np.random.RandomState(seed)
    #随机数(标准正态分布)返回300行2列的矩阵，表示300组坐标点(x0, x1)作为输入数据集
    X = rdm.randn(300, 2)
    #从X这个300行2列的矩阵中取出一行，判断如该这两个坐标的平方和小于2,给Y_赋值1, 其余赋值0
    #作为输入数据集的标签(正确答案)
    Y_ = [(int)(x0*x0 + x1*x1 < 2) for (x0, x1) in X]
    #遍历Y中的每个元素，1赋值'red'其余赋值'blue'，这样可视化显示时人可以直观区分
    Y_c = [['red' if y else 'blue'] for y in Y_]
    #对数据集X和标签Y进行形状整理，第一个元素喂-1表示跟随第二列计算，第二个元素表示多少列。X为n行2列，Y_为n行1列
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)

    plt.scatter(X[:,0], X[:,1], c = np.squeeze(Y_c))
    plt.show()

    return X, Y_, Y_c
```

forward.py
```
#coding:utf-8
#导入模块，生成模拟数据集
import tensorflow as tf
# 定义神经网络的输入，参数和输出，定义前向传播过程

def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b

def forward(x, regularizer):
    w1 = get_weight([2,11], regularizer)
    b1 = get_bias([11])
    #计算层需要经过激活函数处理
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([11,1], regularizer)
    b2 = get_bias([1])
    #结果层不需要经过激活函数处理
    y = tf.matmul(y1, w2) + b2
    return y
```

backward.py
```
'''
#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import opt4_8_generateds as ge
import opt4_8_forward as fw

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DACY = 0.999
REGULARIZER = 0.01

def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    
    X, Y_, Y_c = ge.generateds()

    y = fw.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, 
        global_step,
        300/BATCH_SIZE,
        LEARNING_RATE_DACY,
        staircase= True
    )

    #定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = tf.add_n(tf.get_collection('losses')) + loss_mse
    
    #定义包含正则化的反向传播方法
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i*BATCH_SIZE)%300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict={x:X, y_:Y_})
                print("After %d steps, loss is %f"%(i, loss_v))
        
        xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x:grid})
        probs = probs.reshape(xx.shape)
    plt.scatter(X[:,0], X[:,1], c = np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels = [.5])
    plt.show()

if __name__ == "__main__":
    backward()
    
```

---