---
layout: post
title:  "多变量逻辑回归与竞争模型"
date:   2018-04-13 21:34:28 +0800
categories: [strategy]
---

* TOC
{:toc}

[TOC]

## 问题
在品牌广告中，平台通常需要提供大量的剧目内容，例如电影、电视剧、综艺等。招商广告主在广告投放时希望指定若干，在同一个时期，可能会有多个剧目同时播放，在


[Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)
[Multinomial probit](https://en.wikipedia.org/wiki/Multinomial_probit)


样例demo
```Python
#https://www.cnblogs.com/chenkuo/p/8087055.html
import tensorflow as tf
import numpy as np
from numpy.random import RandomState
batch_size = 8

#假设是3维度，或者30，由样本特征维度决定
xa = tf.placeholder(tf.float32, shape=(None, 3), name="xa-input")
wa = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
tempa=tf.exp(tf.matmul(xa,wa))

xb = tf.placeholder(tf.float32, shape=(None, 3), name="xb-input")
wb = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
tempb=tf.exp(tf.matmul(xb,wb))

ya=tempa/(1+tempa+tempb)
yb=tempb/(1+tempa+tempb)

ya_ = tf.placeholder(tf.float32, shape=(None, 1), name="ya-input")
yb_ = tf.placeholder(tf.float32, shape=(None, 1), name="yb-input")

loss=tf.reduce_sum(-ya_*tf.log(ya)-yb_*tf.log(yb))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#通过随机数生成一个模拟数据集
rdma = RandomState(10)
rdmb = RandomState(20)
dataset_size = 128
Xa = rdma.rand(dataset_size, 3)
Xb = rdmb.rand(dataset_size, 3)

Ya_temp = [[x1 + 5*x2+ 2*x3 + rdma.rand()/10.0-0.05] for (x1, x2,x3) in Xa]
Yb_temp = [[2*x1 + 1*x2+ 3*x3 + rdmb.rand()/10.0-0.05] for (x1, x2,x3) in Xb]

#tensor can not be used in feed_dict
Ya=np.divide(np.exp(Ya_temp),1+np.exp(Ya_temp)+np.exp(Yb_temp))
Yb=np.divide(np.exp(Yb_temp),1+np.exp(Ya_temp)+np.exp(Yb_temp))

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	steps = 50000
	for i in range(steps):
		start = (i * batch_size) % dataset_size
		end = min(start + batch_size, dataset_size)
		sess.run(train_step, feed_dict={xa:Xa[start:end], ya_:Ya[start:end],xb:Xb[start:end], yb_:Yb[start:end]})
		if i%1000==0:
			res=(sess.run(loss, feed_dict={xa:Xa[start:end], ya_:Ya[start:end],xb:Xb[start:end], yb_:Yb[start:end]}))
			print(i,res)

	print(sess.run([wa,wb]))

'''
48000 3.7348
49000 2.38889
[array([[ 1.32607698],
       [ 5.57283545],
       [ 2.3874886 ]], dtype=float32), array([[ 2.38537025],
       [ 1.29027772],
       [ 3.43826056]], dtype=float32)]
'''
```

## 总结
