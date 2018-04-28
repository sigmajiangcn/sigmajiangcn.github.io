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


















## 总结
"工欲善其事，必先利其器"。

## 参考文献
```python
```
- 《Pandas习题集》 https://github.com/ajcr/100-pandas-puzzles
