---
layout:     post
title:      Tensorflow实战系列
subtitle:   （二）用tensorflow实现K Nearest Neighbors
date:       2017-07-28
author:     Stephen
header-img: img/post-bg-coffee.jpeg
tags:
    - 深度学习
    - Tensorflow
    - Python
    - K-Nearest-Neighbors
    - KNN
---

# Tensorflow 实现 K Nearest Neighbors 模型  

---

> 下面开始用 Tensorflow 实现 K Nearest Neighbors 模型  
> 这里数据使用的是 MNIST 的图片分类数据

---

## 1.导入 tensorflow 包并下载 MNIST 数据

```python
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

这里用到的 tensorflow 版本为 1.0.0，其他版本没有测试。

## 2.模型数据准备

```python
# In this example, we limit mnist data
Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)
```
数据说明：

 - 这里用5000条数据做训练
 - 200条数据做测试
 

## 3.定义模型的输入 

```python
# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder('float',[784])
```

这里用 placeholder 来装载变量，格式为 float32

> * xtr: 所有的训练数据集，维度为 _ * 784 (MNIST 的图像shape 为 28*28)  
> * xte: 测试数据，维度 1 * 784， 对于每一个测试数据都要和所有的训练数据进行距离计算

## 4.定义距离度量方式

```python
# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
```
这里用曼哈顿距离

## 5.定义预测函数

```python
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0

# Initializing the variables
init = tf.global_variables_initializer()
```

获得和每个测试数据相近的邻居数据。

---

## 6.模型运行和结果输出

```python
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i,:]})
        # Get nearest neighbor class label and compare it to ite true label
        print ("Test", i, "Prediction:", np.argmax(Ytr[nn_index]),\
               "True Class:", np.argmax(Yte[i]))
        
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1 / len(Xte)
            
    print ("Done!")
    print ("Accuracy:", accuracy)
```

KNN在200条测试数据集上的准确率可以达到0.89
> * Accuracy: 0.890000000000000789333