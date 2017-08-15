---
layout:     post
title:      Tensorflow实战系列
subtitle:   （一）用tensorflow实现Logistic Regression
date:       2017-07-27
author:     Stephen
header-img: img/post-bg-debug.png
catalog: true
tags:
    - 深度学习
    - Tensorflow
    - Python
    - Logistic Regression
---

# Tensorflow 实现 Logistic Regression 模型  

---

> 下面开始用 Tensorflow 实现 Logistic Regression 模型  
> 这里数据使用的是 MNIST 的图片分类数据

---

## 1.导入 tensorflow 包并下载 MNIST 数据

```python
import tensorflow as tf

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

这里用到的 tensorflow 版本为 1.0.0，其他版本没有测试。

## 2.设定模型训练的参数

```python
# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
#display_step = 1
```
参数说明：

 - learning_rate 学习率
 - training_epochs 训练轮数(模型见到全部数据的次数)
 - batch_size 数据不是一次性喂给模型，而是分为一个一个batch丢给模型，用每个batch来训练模型
 

## 3.定义模型的输入、输出  

```python
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
```

这里用 placeholder 来装载变量，格式为 float32

输入：x (MNIST 的图像shape 为 28*28)  
输出：y (0-9 digits)

## 4.定义模型的 weights(权重矩阵) 和 bias(偏置)

```python
# Set model weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```
因为只用到了单层的神经网络，所以权重矩阵和偏置都只有一个

## 5.搭建模型

```python
# Construct model
pred = tf.nn.softmax(tf.matmul(x,W)+b)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
```

逻辑回归可以用公式：  y=wx+b  来表示。
损失函数用交叉熵损失，训练采用梯度下降。

---

## 6.模型训练和结果输出

```python
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print ("Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print ("Optimizition Finished!")
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y:mnist.test.labels[:3000]}))
```

经过25轮的训练之后，模型的准确率可以达到0.889333
> * Optimizition Finished!
> * Accuracy: 0.889333