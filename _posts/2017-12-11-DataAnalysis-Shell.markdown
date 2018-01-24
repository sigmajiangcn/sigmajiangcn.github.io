---
layout: post
title:  "数据分析工具之一:Shell常用命令"
date:   2017-12-11 21:12:06 +0800
categories: [Tools]
---

* TOC
{:toc}

[TOC]
## 前言
Shell是一门常用的脚本语言，易于入门，方便灵活。在数据处理时，尤其是处理日志时非常有帮助。常用的主要如Grep、awk、sed三大神器。

**持续更新**

## 用grep查找日志问题
```shell
grep -R -n -A 5 -B 6 "error" *.log
```

## 单个文件字符串替换
```shell
sed -i "s/[()]//g"   替换
sed  -i "1i\%s" %s   插入
```

## 两个文件合并
```shell
awk -F"," 'NR==FNR {a[$1]=$2} NR>=FNR{if(NF>5 && $6 in a && $5==a[$6]){b[$6]=$0}} END{for(i in b){print b[i]}}' a b  >c
```

## 顺序执行
```shell
ps -ef |grep smart_server |awk '{print $2}' |xargs kill -9
```

## 总结
"工欲善其事，必先利其器"。

## 参考文献