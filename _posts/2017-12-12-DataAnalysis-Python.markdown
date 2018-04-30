---
layout: post
title:  "数据分析工具之二:Python常用库"
date:   2017-12-12 21:12:06 +0800
categories: [Tools]
---

* TOC
{:toc}

[TOC]
## 前言
Python是一门常用的编程语言，易于入门，方便灵活。尤其当掌握了一些常用的库，更是用的飞起。这里主要介绍介绍Pandas。

2018年3月1日，微信公众号《机器学习研究会》推送的一篇《Pandas习题集》，其github链接提供了大量的**微操**（*Tiny but effective and strong operation*）,原文：
Since pandas is a large library with many different specialist features and functions, these excercises focus mainly on the fundamentals of manipulating data (indexing, grouping, aggregating, cleaning), making use of the core DataFrame and Series objects. [Many of the excerises](https://github.com/ajcr/100-pandas-puzzles) here are straightforward in that the solutions require no more than a few lines of code (in pandas or NumPy - don't go using pure Python!). Choosing the right methods and following best practices is the underlying goal.

另外，还有[《Numpy习题集》](https://github.com/rougier/numpy-100)。


**持续更新**

## 文件交互
- read_csv
```python
df=pd.read_csv(train_file,sep=",",index_col=target_list,nrows=5)
```
- to_csv

## 常用算子
- query
```python
select_df=total_df.query("video>10000 and title_index>0")
```
- groupby
- merge
- date_range
- concat
- reset_index
- sort_values
```python
algo_filter_df=compare_df.groupby(target_list)[['algorithm','result']].apply(lambda x:pd.DataFrame.sort_values(x,by='result')[['algorithm']].head(2)).reset_index()
```
- lambda
```python
df.apply(lambda x:True if x.sum()>10 else 'false')
df['mean']=df.mean(axis=1)
df.query("mean>10").drop(['mean'],axis=1)
```

- fillna
- drop del
drop 删除列返回新的df，不影响原有数据，更安全

- loc、iloc、ix
```python
df.head(2).iloc[:,0:1]
```

- isin
```python
>>> df
  countries
0        US
1        UK
2   Germany
3     China
>>> countries
['UK', 'China']
>>> df.countries.isin(countries)
0    False
1     True
2    False
3     True
Name: countries, dtype: bool
>>> df[df.countries.isin(countries)]
  countries
1        UK
3     China
>>> df[~df.countries.isin(countries)]
  countries
0        US
2   Germany


或者：
criterion = lambda row: row['countries'] not in countries
not_in = df[df.apply(criterion, axis=1)]
```
具体请见[how-to-implement-in-and-not-in-for-pandas-dataframe](https://stackoverflow.com/questions/19960077/how-to-implement-in-and-not-in-for-pandas-dataframe)

## 覆盖式定义
```Python
df['C'] = df['C'].apply(np.int64)
```
[重新赋值修改部分](https://stackoverflow.com/questions/21291259/convert-floats-to-ints-in-pandas/21291622)
## matplotlib绘图
- figure
- hist
- xlabel
- ylabel
- legend
- show
- 坐标轴点数选取
横坐标如果太密集，可以[选取部分值出来](https://mail.python.org/pipermail/python-list/2006-March/389991.html)：
```Python
from matplotlib.ticker import MaxNLocator
from pylab import figure, show, nx

fig = figure()
ax = fig.add_subplot(111)
ax.plot(nx.mlab.rand(1000))
ax.xaxis.set_major_locator(MaxNLocator(4))
show()
```
- 坐标轴时间轴格式
可以调整时间轴的[格式](https://www.tuicool.com/articles/jmQzUzy)和旋转角度：
```Python
 ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
```
[mdate](https://matplotlib.org/devdocs/gallery/api/date.html)




## 其他常用点
- glob
```Python
current_predict_paths = glob.glob('data/%s/%s/model/%s_output_*'%(key,target,train_day))
```
- os.makedirs
- os.sys.platform.startswith("win")
- 正则表达式
```python
例如找出剧目vid列表，其中每个剧目都是由小写字母和数字构成：
import re
pattern=re.compile(r'[a-z0-9]+')
vid_list=pattern.findall(vid_string)
```
- 参数解析
```python
import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("-d", "--date", help="日期", default="2017-08-01")
args=parser.parse_args(sys.argv[1:])
today=args.date
```
- 配置文件
```python
import ConfigParser
config=ConfigParser.ConfigParser()
config.read("conf/env.ini")
path=self.config.get("Section_A", "variable_a")
其中国配置文件env.ini内容类似如下：
[Section_A]
variable_1=10
variable_2=10
[Section_B]
variable_1=10
...
```

- 界面显示字符宽度
```python
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.height',500)
```
- 虚拟化界面
```python
在linux+selenium中模拟浏览器爬虫会有遇到：
from xvfbwrapper import Xvfb
xvfb = Xvfb(width=1280,height=720)
xvfb.start()
xvfb.stop()
```

- 变量名字符串
[how-to-get-a-variable-name-as-a-string](https://stackoverflow.com/questions/2553354/how-to-get-a-variable-name-as-a-string)
```python
>>> some= 1
>>> list= 2
>>> of= 3
>>> vars= 4
>>> dict( (name,eval(name)) for name in ['some','list','of','vars'] )
{'list': 2, 'some': 1, 'vars': 4, 'of': 3}
```

- 字符编码
```python
import chardet
detect_result = chardet.detect(target.encode('utf8'))
```

- 自定义包的引用路径
```python
查看的两种方法：sys.path  或者 os.sys.path
修改的两种方法：export PYTHONPATH=$PYTHONPATH：$NEW_PATH  或者在~/.bashrc中修改
在代码前面sys.path.append()不够优雅
```
## 常用框架
### 框架之一：Django
### 框架之二：Selenium
### 框架之三：Sklearn
```python
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
```
### 框架之四：Tensorflow
1.Graph
    Tensorflow支持通过tf.Graph函数来生成性的计算图。不同计算图上的张量和运算都不会共享。
2.tensor
    所有数据都可以通过张量的形式来表示。张量可以被简单理解为多维数组。
- 零阶张量表示标量（Scalar），也就是一个数；
- 第一阶张量表示向量（Vector），也就是一个一维数组；
- 第$n$阶张量表示一个$n$维数组。

张量中并没有真正保存数字，它保存的是如何得到这些数字的计算过程，是对结果的一个引用。
   TensorFlow的这一点与Numpy中的数组不同，计算结果不是一个具体的数字，而是一个张量的结构。主要保存：
   - 名字（name）
   - 维度（shape）
   - 类型（type）
 可以通过result.get_shape得到结果的维度信息。可以通过tf.Session().run(result)语句得到计算结果。
3.dtype
类型不匹配不能加减，要制定，如dtype=tf.float32,主要支持14种不同的类型：
- 实数
    - tf.float32
    - tf.float64
- 整数
    - tf.int8
    - tf.int16
    - tf.int32
    - tf.int64
    - tf.uint8
- 布尔型
    - tf.bool
- 复数
    - tf.complex64
    - tf.complex128

4.Session
一方面tf需要通过Graph、Tensor、运算符等来组织数据和运算，另一方面需要通过会话来执行定义好的运算。简单创建会话的方式如下:
```Python
with tf.Session() as sess:
    sess.run(...)
```
会话可以通过tf.ConfigProto()来实现。

-游乐场[http://playground.tensorflow.org](http://playground.tensorflow.org)是一个通过网页浏览器就可以训练的简单神经网络并实现了可视化训练过程的工具。
这样可以自动管理资源创建和释放工作，不会因为忘记sess.close()而造成资源泄漏。

5.算子
- 乘法：matmul
- 变量分布初始化：
weights=tf.variable(tf.random_normal([2,3],stddev=2))
随机数生成函数：
  - tf.random_normal
  - tf.truncated_normal
  - tf.random_uniform
  - tf.random_gamma
常数生成函数：
  - tf.zeros
  - tf.ones
  - tf.fill
  - tf.constant


其他：
1.TensorFlow提供了placeholder机制用于提供输入数据，placeholder相当于定义了一个位置，这个位置中的数据在程序运行时再指定。这样在程序中就不需要生成大量敞亮来提供输入数据，而只需要通过将数据通过placeholder传入TensorFlow，便于每轮迭代计算。
另外变量的类型不可变，维度可以推导。
2.feed_dict是一个字典，给出每个用到的placeholder的取值。
3.常见的三种优化方法：
- tf.train.GradientDescentOptimizer
- tf.train.AdamOptimizer
- tf.train.MomentumOptimizer
选定的反向传播算法将会对所有的Graphkeys.TRAINABLE_VARIABLES集合中的变量进行优化，使得在当前的batch下损失函数最小。

4.交叉熵
[交叉熵](https://www.zhihu.com/question/41252833)刻画的是两个分布之间的距离。表征的是通过概率分布$q$来表达概率分布$p$的困难程度。因为正确答案是希望得到的结果，所以当交叉熵作为神经网络的损失函数时，$p$代表的是正确答案，$q$代表的是预测值。两个概率分布之间的距离越小，说明预测的结果和真实的结果差距越小。
连续函数：
$$H(p,q)=E_p[-logq]=H(p)+D_{KL}(p||q)$$
两项中$H(p)$是$p$的信息熵，后者是想对熵。

离散函数：
$$H(p,q)=-\sum_{x}p(x)logq(x)$$

要注意交叉熵的乘法是＊，不是矩阵乘法tf.matmul。
5.深度学习的非线性
[维基百科](https://en.wikipedia.org/wiki/Deep_learning)定义：一类通过多层非线性变换对高复杂性数据建模算法的合集。其中深层神经网络是实现“多层非线性变换”最常用的一种方法。
线性模型的局限性：
只通过线性变换，则任意层的全连接神经网络和单层神经网络模型的表达能力没有任何的差异。
TensorFlow提供了7种不同的非线性激活函数，常用的非线性激活函数如下三个：
- ReLU
$$f(x)=max(x,0)$$
对应的是tf.nn.relu。
- sigmoid
$$f(x)=\dfrac{1}{1+e^{-x}}$$
对应的是tf.sigmoid。
- tanh
$$f(x)=\dfrac{1-e^{-2x}}{1+e^{-2x}}$$
对应的是tf.tanh。
另外，偏置项bias也是神经网络中常用的一种结构。

6.反向传播backpropagation与梯度下降gradient descent
- 学习率
learning rate控制参数更新的速度，如果幅度过大，那么可能导致参数在极优值的两侧来回移动。TensorFlow提供了一种更加灵活的学习率衰减方法：
指数衰减法：tf.train.exponential_decay

- 过拟合
当一个模型过度复杂之后，它可以很好地“记忆”每一个训练数据中的随机噪声的部分而忘记了要去”学习”训练数据中通用的趋势。
可以采用正则化regularization.正则化的思想就是在损失函数中加入刻画模型复杂程度的指标。在稀疏性能和求导方面，$L_1$和$L_2$有差异。一般可以同时使用：
$$R(w)=\sum_{i}\alpha|w_i|+(1-\alpha)w_i^2$$.
两种正则化的api如下：
 - tf.contrib.layers.l1_regularizer(lambda=步长)(weights)
 - tf.contrib.layers.l2_regularizer(lambda=步长)(weights)

关键词总结：
```Python
非线性
多层
优化目标
反向传播
梯度下降
正则化
```

7.Mnist数字识别问题
[$MNIST]$(http://yann.lecun.com/exdb/mnist)是一个非常有名的手写数字识别数据集，它是$NIST$数据集的一个子集。它包含了60000张图片作为训练数据，10000张图片作为测试数据。在$MNIST$数据集中的每一张图片都代表了0~9中的一个数字。图片的大小都是28*28，且数字都会出现在图片的正中间。
- train     训练
- validation验证
- test      测试


从60000张训练数据中抽取出5000张作为验证集，其余55000作为训练集，可以采用交叉验证($cross  validation$)的方式来验证模型效果。但因为神经网络训练时间本身更加就比较长，采用交叉验证会花费大量时间。在海量数据的情况下，一般更多采用验证数据集的形式来评测模型的效果。
需要注意的是，虽然一个神经网络模型的效果最终是通过测试数据来评判的，但是我们不能直接通过模型在测试数据上的效果来选择参数。使用测试数据来选择参数可能会造成神经网络过度拟合测试数据，从而失去对未知数据的预判能力。
一般，可以在不同的迭代轮数的情况下，计算模型在验证数据和测试数据上的正确率。




## 总结
"工欲善其事，必先利其器"。

## 参考文献
```python
```
- 《Pandas习题集》 https://github.com/ajcr/100-pandas-puzzles
