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
另外，可以计算模型在验证集和测试集的准确率曲线，分析二者是否正相关。
代码主要分为三个部分：
- mnist_inference.py
定义了前向传播的过程以及神经网络中的参数。
- mnist_train.py
定义了神经网络的训练过程。
- mnist_eval.py
定义了测试过程。

8.变量管理
TensorFlow提供了通过变量名称来创建或者获取一个变量的机制，通过这个机制，在不同的函数中可以直接通过变量的名字来使用变量，而不需要将变量通过参数的形式到处传递。主要是通过tf.get_variable和tf.variable_scope函数来实现。
tf.variable_scope函数生成的上下文管理器也会创建一个TensorFlow中的命名空间，在命名空间内创建的变量名称都会带上这个命名空间名作为前缀。

9.模型持久化
- tf.train.Saver
  - model.ckpt.meta
    保存了计算图的结构
  - model.ckpt
    保存了程序中每一个变量的取值
  - ckeckpoint
    保存了一个目录下所有的模型文件列表

TensorFlow可以通过字典将模型保存时的变量名和需要加载的变量联系起来。通过元图(MetaGraph)来记录计算图中节点的信息以及运行计算图中节点所需要的元数据。元图是由MetaGraphDef Protocol Buffer定义的。

10.卷积神经网络CNN
$MNIST$手写字体识别数据集是一个相对简单的数据集，在其他更加复杂的图像识别数据及上，卷积神经网络有更加突出的表现。$Cifar$数据集就是一个影响力很大的图像分类数据集。$Cifar$数据集分为了$Cifar-10$和$Cifar-100$两个问题，它们都是图像词典项目(visual dictionary)中800万张图片中的一个子集。
$Cifar$数据集的图片为32*32的彩色图片，这些图片是由$Alex\ Krizhevsky$教授、$Vinod\ Nair$博士和$Geoffrey\ Hinton$教授整理的。
区别在于：$MNIST$是手写字体识别，每一张图片只包含一个数字；$Cifar$是不同类别的识别，每一张图片都只包含一个种类的物体。
然而无论是$MNIST$还是$Cifar$数据集，相比真是环境下的图像识别问题，还是太简单，主要在于：
- 分辨率
现实生活图片分辨率要高于32*32，并且分别率不固定
- 类别多
现实世界物体种类不再局限于10类或者100类，并且一张图片不会只有一个种类的物体。

为了更加贴近真实环境下的图像识别问题，由斯坦福大学的李飞飞教授带头整理的$ImageNet$很大程度可以解决这个问题。
$ImageNet$是一个基于$WordNet$的大型图像数据库，在$ImageNet$中，将近1500万图片被关联到了$WordNet$的大约20000个名词同义词集上。目前每一个与WordNet相关的$ImageNet$同义词集都代表了现实世界的中的一个实体，可以被认为是分类问题中的一个类别。
$ImageNet$中的图片是从互联网爬虫得到，并且通过亚马逊的人工标注服务将图片分类到对应的同义词集合上。
$ImageNet$以前每年都会举办图像识别相关的竞赛($ImageNet\ Large\ Scale\ Visual\ Recognition\ Challenge,ILSVRC$),在$ILSVRC2012$中有来自1000个类别的120万张图片，其中每张图片属于且只属于一个类别。图片大小各异。
2012,AlexNet
2013,ZF
2014,VGG,GoogleNet
2015,Inception-v3,ResNet
2016,GoogleNet-v4
全连接层网络结构：每两层之间的所有节点都是有边相连的。
**这是与卷积神经网络、循环神经网络***最大的区别。  

神经网络两层之间的连接方式很重要，在全连接神经网络中网络参数太多。假设第一个隐层有500个节点，如果都是全连接神经网络，则有如下参数规模：
- $MNIST$
28*28*500+500=392500
- $CIFAR$
32*32*3*500+500=150万

参数这么大，难以计算和训练，并且关键在于下一层的某个节点是否有必要和上面一层所有节点相连咧？
因此，针对图像问题，需要设计更合理的神经网络来有效减少神经网络中的参数个数。卷积神经网络正是基于这个目的。
- 输入层
输入RGB三通道维度图片
- 卷积层
针对每个小块深入分析得到抽象特征
- 池化层
不改变矩阵深度，会改变矩阵大小。
- 全连接层
经过几层卷积层和池化层，抽象得到信息含量更高的特征，是一个自动图像特征抽取的过程。之后，依然使用全连接来完成分类任务。
- softmax层
可以得到当前样例属于不同种类的概率分布情况。

#### 两种重要结构
- 卷积层
filter过滤器或者kernel内核。
zero padding补0
移动步长
$$out=(in-filter+1)/stride$$
对长和宽都生效。在卷积神经网络中，每一个卷积层中使用的过滤器中的参数都是一样的。从直观上看，共享过滤器的参数可以使得图像上的内容不受位置的影响。并且可以大大减少神经网络的参数、参数量与输入图片的大小无关，只和过滤器的尺寸、深度以及当前节点的矩阵深度有关，易于扩展。
```Python
filter_weight=tf.get_variable(
            'weights',[5,5,3,16],
            initializer=tf.truncated_normal_initializer(stddev=0.1)  
            )
biases=tf.get_variable(
            'biases',[16],
            initializer=tf.constant_initializer(0.1)
            )
conv=tf.nn.conv2d(input,filter_weight,stides=[1,1,1,1],padding='SAME')
bias=tf.nn.bias_add(conv,biases)
actived_conv=tf.nn.relu(bias)
```

- 池化层
池化层可以非常有效地缩小矩阵的尺寸，从而减少最后全连接层中的参数，并且加快计算速度、防止过拟合。本质也是一个过滤器，不过常见的有如下两种：
  - 最大池化层
    max pooling，最大值操作，应用较多
  - 平均池化层
    average pooling，平均值操作，应用较少。
既然也是滤波器，也需要设置尺寸、是否全0补充以及移动步长。实践中，一般不使用池化来改变图像举证的深度。
page 161
```Python
pool=tf.nn.max_pool(
        actived_conv,
        ksize=[1,3,3,1],
        strides=[1,2,2,1],
        padding='SAME'
)
```
其中各个参数解释如下：
- ksize：kernel size，滤波器尺寸，第一个和最后一个是1，意味着池化层的滤波器是不可以跨不同输入样例或者节点矩阵深度的，最常见的尺寸为[1，2，2，1]或者[1，3，3，1]。
- strides：步长，第一个和最后一个是1，一直这样池化层不能减少节点矩阵的深度或者输入样例的个数。
- padding表示填充补0，取值有VALID和SAME，VALID表示不使用全0填充，SAME表示使用全0填充。
#### 两种经典网络
##### LeNet-5
这是Yann Lecun教授在1998年《Gradient-based learning applied to document recognition》,它是第一个成功用户数字识别问题的*卷积神经网络*。这是一个7层的网络，在MNIST数据集上，可以达到99.2%的正确率。
- 第一层：卷积层
- 第二层：池化层
- 第三层：卷积层
- 第四层：池化层
- 第五层：全连接层
- 第六层：全连接层
- 第七层：全连接层

一般而言，用于图片分类问题的卷积神经网络架构：
输入层->(卷积层+->池化层？)+->全连接层+
其中，+号表示一层或者多层，？号表示有或者没有。池化层虽然可以起到减少参数防止过拟合的问题，但是部分论文发现可以通过直接调整卷积层步长也可以完成。*因此有些卷积神经网络中没有池化层*。
AlexNet、ZFNet以及VGGNet基本满足上述正则表达式结构。

##### Inception-v3
Inception结构是一种和LeNet-5结构完全不同的卷积神经网络结构。LeNet-5结构中，不同卷积层通过串联的方式连接在一起。而Inception-v3结构中将不同的卷积层通过并联的方式结合在一起。
虽然滤波器的大小不同，但如果所欲的滤波器都是用全0填充且步长为1，那么前向传播得到的结果矩阵的长和宽都与输入矩阵一致。这样经过不同过滤器处理的结果矩阵可以拼接得到一个*更深*的矩阵。
在《rethinking the Inception Architecture for Computer Vision》中，Inception—v3模型共有46层，由11个Inception模块构成。共有96个卷积层。


#### 迁移学习
可以使用少量训练数据在短时间内训练处还不错的神经网络模型。
根据论文DeCAF：A Deep Convolutional Activation Feature for Generic Visual Recognition中的结论，可以保留训练好的Inception-v3模型中所有卷积层的参数，只是替换最后一层全连接层。在最后这一层全连接层之前的网络层称之为瓶颈层（bottleeneck）。
游戏场景的识别。

11.图像数据处理
- TFRecord输入数据格式
TFRecord文件中的数据都是通过tf.train.Example Protocol Buffer的格式存储的。
- 图像预处理
图片在存储时并非直接记录RGB色彩模式数据，而是压缩之后的结果。所以要将一张图像还原成一个三维矩阵，需要解码的过程。TensorFlow提供了对jpeg和png格式图像的编码/解码函数，如下：
```Python
image_raw_data=tf.gfile.FastGFile("/path/to/picture","r").read()
with tf.Session() as sess:
  #read to [1-256]
  img_data=tf.image.decode_jpeg(image_raw_data)
  #write to float encoded
  encoded_jpeg=t.image.encode_jpeg(img_data)
  with tf.gfile.GFile("/path/to/output","wb") as f:
    f.write(encoded_jpeg.eval())
```

- 图像大小调整
TensorFlow提供了四种不同的方法，封装在tf.image.resize_images()
```Python
resized=tf.image.resize_images(img_data,300,300,method=0)
```
| method | 算法 |
| :-: | :-: |
|0|双线性插值法Bilinear interpolation|
|1|最近邻居法Nearest Neighbour interpolation|
|2|双三次插值法Bicubic interpolation|
|3|面积插值法Area interpolation|

- 图像翻转
```Python
flipped=tf.image.flip_up_down(img_data)#上下翻转
flipped=tf.image.flip_left_right(img_data)#左右翻转
flipped=tf.image.transpose_image(img_data)#对角线翻转
```
- 图像色彩调整
```Python
tf.image.adjust_hue
tf.image.adjust_saturation
tf.image.adjust_brightness
tf.image.adjust_contrast
```
- 图像标注框
```Python
tf.image.draw_bounding_boxes
```
- 多线程输入数据处理框架
  - 队列与多线程
  tf.FIFOQueue
  tf.Coordinator
  - 输入文件队列
  tf.train.string_input_producer
  - 组合训练数据
  将多个输入样例组织成一个batch可以提高模型训练的效率。所以在得到单个样例的预处理结果后，还需要将他们组织成batch，然后再提供给神经网络的输入层。
  tf.train.batch
  tf.train.shuffle_batch
  tf.train.shuffle_batch_join

12.循环神经网络
循环神经网络（recurrent neural network,RNN）
长短时记忆网络（long short-term memory,LSTM）
主要应用在自然语言处理和时序分析领域。
[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
![LSTM电路图](/img/RNN-unrolled.png)
一个最简单的循环体结构的循环神经网络，在这个循环体中只使用了一个类似全连接层的神经网络结构，如下图：
 ![LSTM电路图](/img/SimpleRNN.png)
循环神经网络中的状态是通过一个向量来表示的，这个向量的大小称为循环神经网络隐藏层的大小，假设为$h$。上图中循环神经网络的输入分为两个部分，一部分为上一个时刻的状态，另一个部分为当前时刻的输入样本。
对于时间序列分析数据来说（例如不同时刻商品销量），每一个时刻的输入样例可以是当前时刻的数值（例如销量值）；对于语言模型而言，输入样例可以是当前单词对应的单词向量（word embedding）。
循环体的参数个数为：
$$(h+x)\cdot h+h$$

循环神经网络的关键是使用历史的信息来帮助当前的决策。例如使用之前的单词来加强对当前单词的理解，这样可以利用传统神经网络不具备的信息，也带来了更大的技术挑战--长期依赖（long term dependencies）。
与单一tanh循环体结构不同，LSTM是一种拥有三种“门”结构的特殊网络结构。LSTM靠一些门的结构让信息有选择性的影响循环神经网络中每个时刻的状态。输入门和遗忘门是LSTM结构的核心。

```Python
lstm=rnn_cell.BasicLSTMCell(lstm_hidden_size)
state=lstm.zero_state(batch_size,tf.float32)
loss=0.0
for i in range(num_steps):
  if i>0:
    tf.get_variable_scope().reuse_variables()
  lstm_output,state=lstm(current_input,state)
  final_output=fully_connected(lstm_output)
  loss+=calc_loss(final_output,expected_output)
```

- 循环神经网络的变种
  - 双向循环神经网络
  《Bidirectional recurrent neural network》,Bidirectional RNN.在预测一个语句中的缺失的单词不仅需要根据前文来判断，也需要根据后面的内容，这时双向循环网络就可以发挥它的作用。双向神经网络是由两个循环神经网络上下叠加在一起组成的。输出由这两个循环神经网络的状态共同决定。
  在每一个时刻$t$，输入同时给两个循环神经网络，而输出则是由这两个单向循环神经网络共同决定。

  - 深层循环神经网络
  deepRNN，为了增强模型的表达能力，可以将每一个时刻上的循环体重复多次。和卷积神经网络类似，每一层的循环体中参数是一致的，而不同层中的参数可以不同。为了更好地支持深层循环，TTensorFlow中提供了MultiRNNCell类来实现深层循环神经网络的前向传播过程。
  ```Python
  lstm=rnn_cell.BasicLSTMCell(lstm_size)
  stacked_lstm=rnn_cell.MultiRNNCell([lstm]*number_of_layers)
  state=stacked_lstm.zero_state(batch_size,tf.float32)
  for i in range(len(num_steps)):
      if i>0:
        tf.get_variable_scope().reuse_variables()
        stacked_lstm_output,state=stacked_lstm(current_input,state)
        final_output=fully_connected(stacked_lstm_output)
        loss+=calc_loss(final_output,expected_output)
  ```

- dropout
通过dropout，可以让神经网络更加健壮。卷积神经网络只在最后的全连接层使用dropout，循环神经网络一般只在不同层循环体结构之间使用dropout，而不在同一层的循环体结构之间使用。
也就是说从时刻$t-1$传递到时刻$t$时，循环神经网络不会进行状态的dropout；而在同一个时刻$t$中，不同层循环体之间会使用dropout。
```Python
lstm=rnn_cell.BasicLSTMCell(lstm_size)
dropout_lstm=tf.nn.rnn_cell.DropoutWrapper(lstm,out_keep_prob=0.5)
stacked_lstm=rnn_cell.MultiRNNCell([dropout_lstm]*number_of_layers)
```

- 案例使用：
  - 自然语言建模

  - 时间序列建模



13.TensorBoard可视化



14.TensorFlow计算加速















## 总结
"工欲善其事，必先利其器"。

## 参考文献
```python
```
- 《Pandas习题集》 https://github.com/ajcr/100-pandas-puzzles
