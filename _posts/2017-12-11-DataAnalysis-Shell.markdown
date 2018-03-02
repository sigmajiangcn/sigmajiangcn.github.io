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
## 文件导入mysql
```shell
mysql -h $IP -u $USER -p $PASSWORD -D $DATABASE -e"use $ONE_DATABASE;load data local  infile '$LOCAL_FILE_NAME' replace  into table $TO_WHICH_TABLE fields terminated by ',';"
```

## 当前路径
```shell
FOLDER=$(cd `dirname $0`;pwd)
echo $FOLDER
```

## 时间参数
```shell
if [[ $1 == "" ]]; then
  DAY=$( date +%Y%m%d -d "yesterday" )
else
  DAY=`date +%Y%m%d -d "$1"`
fi
```

## 时间遍历
```shell
input_start="2016-10-21"
input_end="2018-02-28"

day="$input_start"
while [ "$day" != $input_end ];
do
  today=$(date -d"$day" +"%Y%m%d")
  OTHER COMMAND
  day=$(date -I -d "$day + 1 day")
  sleep 1
done
exit
```

## 文件传输
```shell
autoscp(){
    expect -c "
    set timeout 10;
    spawn scp $1 $2
    expect  "*assword*" {send $3\r}
    expect "*100%"
    expect eof
"
}

autoscp $REMOTE $LOCAL $PASSWORD
```

## 检查集群文件
```shell
hadoop fs -test -e $HADOOP_FILE_PATH
if [ $? -eq 0 ];then
    ACTION
else
    OTHER_ACTION
fi
```




## 总结
"工欲善其事，必先利其器"。

## 参考文献
