---
layout: post
title:  "深度学习之一:Caffe环境搭建"
date:   2017-03-12 21:12:06 +0800
categories: [DeepLearning]
---

* TOC
{:toc}

[TOC]
## 前言
深度学习突破了以往的统计学习方法，虽然目前理论还不明朗，但是着实在很多领域取得了一些阶段性的成果，例如Google的机器翻译、百度的自动驾驶、腾讯的智慧医疗、科大讯飞的语音识别等，引领者新的浪潮。深度学习的工具方面，有Caffe、Tensorflow、Theano、Keras等。网络上流传一张图，如下：
![caffe](/img/caffe_install.jpg)

经过查证，[Lecun](https://www.imooc.com/article/1370)应该没有转推过上图，不过从图中看，Caffe安装的价格最高的，可以看出即使深度学习已经成为一个“地摊”术语，Caffe安装这第一步还是比较重要的。最近结合Caffe，应用深度学习研究广告素材的优化，最开始在环境搭建上面走了一些弯路，为了避免后续再次入坑，简略记录下。

## 为什么会有问题
主要是各个用户机器环境、软件版本(Centos、Ubuntu及其衍生版)都有差异，与官网不一定一致。Caffe安装使用会涉及一些依赖，例如Opencv、Python等，另外编译安装时会涉及一些动态链接库的知识等。

## 有哪些依赖库
主要依赖为Opencv(图像处理)、Boost（例如可以提供python接口)、Protobuf（定义网络结构层）等。

```shell
sudo yum install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel gflags-devel glog-devel lmdb-devel
```

## 安装Anaconda配置Python
通过Anaconda可以方便配置各种需要的环境（例如python 3）需要的依赖。

## 下载Caffe
在github下载源代码，地址为：
https://github.com/BVLC/caffe
## 编译配置
编译caffe常见有两种方式，分别是直接修改makefile和cmake。这里介绍第一种：
```shell
cd caffe
cp Makefile.config.example Makefile.config
vim Makefile.config
```
Makefile.config里面有依赖库的路径，及各种编译配置，如果是没有GPU的情况下，可以参照我下面帮你改的配置文件内容:
```shell
## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# cuDNN acceleration switch (uncomment to build with cuDNN).
# USE_CUDNN := 1

# CPU-only switch (uncomment to build without GPU support).
 CPU_ONLY := 1  如果硬件限制，只需要编译cpu版本

# uncomment to disable IO dependencies and corresponding data layers
# USE_OPENCV := 0
# USE_LEVELDB := 0
# USE_LMDB := 0

# uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
#	You should not set this flag if you will be reading LMDBs with any
#	possibility of simultaneous read and write
# ALLOW_LMDB_NOLOCK := 1

# Uncomment if you're using OpenCV 3
# OPENCV_VERSION := 3

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
# CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
CUDA_DIR := /usr/local/cuda  如果有cuda可以加速计算，这里实际没有用
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 through *_61 lines for compatibility.
# For CUDA < 8.0, comment the *_60 and *_61 lines for compatibility.
# For CUDA >= 9.0, comment the *_20 and *_21 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
		-gencode arch=compute_20,code=sm_21 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
#BLAS := atlas
BLAS := open  blas作为矩阵计算的方式，常见有三种，openblas是其中一种开源免费的库，可以通过yum安装
# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

# Homebrew puts openblas in a directory that is not on the standard search path
# BLAS_INCLUDE := $(shell brew --prefix openblas)/include
# BLAS_LIB := $(shell brew --prefix openblas)/lib

BLAS_INCLUDE := /usr/include/openblas
# This is required only if you will compile the matlab interface.
# MATLAB directory should contain the mex binary in /bin.
# MATLAB_DIR := /usr/local
# MATLAB_DIR := /Applications/MATLAB_R2012b.app

# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
#PYTHON_INCLUDE := /usr/include/python2.7 \
#		/usr/lib/python2.7/dist-packages/numpy/core/include
#PYTHON_INCLUDE := /usr/local/bin/pyt/include/python2.7/ \
		/usr/lib64/python2.7/site-packages/numpy/core/include
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
ANACONDA_HOME := $(HOME)/anaconda2   注意HOME的路径 与sudo有关
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		 $(ANACONDA_HOME)/include/python2.7 \
		 $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include

# Uncomment to use Python 3 (default is Python 2)
# PYTHON_LIBRARIES := boost_python3 python3.5m
# PYTHON_INCLUDE := /usr/include/python3.5m \
#                 /usr/lib/python3.5/dist-packages/numpy/core/include

# We need to be able to find libpythonX.X.so or .dylib.
#PYTHON_LIB := /usr/lib
#PYTHON_LIB := /usr/local/lib
PYTHON_LIB := $(ANACONDA_HOME)/lib

# Homebrew installs numpy in a non standard path (keg only)
# PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
# PYTHON_LIB += $(shell brew --prefix numpy)/lib

# Uncomment to support layers written in Python (will link against Python libs)
 WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib

# If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
# INCLUDE_DIRS += $(shell brew --prefix)/include
# LIBRARY_DIRS += $(shell brew --prefix)/lib

# NCCL acceleration switch (uncomment to build with NCCL)
# https://github.com/NVIDIA/nccl (last tested version: v1.2.3-1+cuda8.0)
# USE_NCCL := 1

# Uncomment to use `pkg-config` to specify OpenCV library paths.
# (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
# USE_PKG_CONFIG := 1

# N.B. both build and distribute dirs are cleared on `make clean`
BUILD_DIR := build
DISTRIBUTE_DIR := distribute

# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
# DEBUG := 1

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0

# enable pretty build (comment to see full commands)
Q ?= @

```
可以看出每个路径配置的方法都有多种方法，可以实际“坑”去google搜索解决。这里有一个有意思的是，sudo make和make编译和运行结果是不一样的，主要是由于HOME目录差异。

## 编译caffe
开启并行编译：
```shell
make -j4
```
可以测试下结果：
```shell
make test
make runtest
```
## 编译pycaffe
```shell
make pycaffe -j4
```
可以测试下python接口是否安装好：
```python
>>> import caffe
```
如果没有报错则说明caffe可能安装好了。
## 应用caffe开发
caffe是否安装完成，是否能够支持引用开发，还需要进一步设置，例如
```shell
export CAFFE_ROOT=...
export LD_LIBRARY_PATH=...
```
这里需要注意LD_LIBRARY_PATH中不要添加anaconda的路径，否则会[libtiff报错](https://groups.google.com/forum/#!msg/caffe-users/wKYe45FKSqE/HcFMlGS-M8gJ).

遇到一个coredump问题，gdb查出
```shell
Temporary breakpoint 1, main (argc=1, argv=0x7fffffffe3d8) at src/server_main.cpp:372
372	src/server_main.cpp: No such file or directory.
```
重新编译静态链接库解决。

## 相关知识点
- LD_LIBRARY_PATH
- PKG_CONFIG_PATH
- 动态链接库
- makefile
- gdb
- rpm
- tesseract


## 总结
"工欲善其事，必先利其器"。caffe是深度学习在图像领域广泛使用的框架，其model zoo有大量的预训练好的模型提供使用。大部分图像相关的应用部分将用到caffe。如何进一步挖掘caffe中的模型实现方法，高效完成个性化的需求是一个重要的方向。

## 参考文献
- [Caffe 安装错误记录及解决办法](http://coldmooon.github.io/2015/07/09/caffe/)
- [linux(CentOS)下的caffe编译安装简易手册](https://www.zybuluo.com/hanxiaoyang/note/364680)
- [caffe install](https://gist.github.com/arundasan91/b432cb011d1c45b65222d0fac5f9232c)
