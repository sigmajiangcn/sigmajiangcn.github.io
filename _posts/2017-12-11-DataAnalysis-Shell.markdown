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
mysql -h $IP -u $USER -p $PASSWORD -D $DATABASE -e"use $ONE_DATABASE;
load data local  infile '$LOCAL_FILE_NAME' replace  into table $TO_WHICH_TABLE fields terminated by ',';"
要注意的是 如果这里的表格最后一列有time_stamp,而文本数据没有的话，可以如下设置：

CREATE TABLE `video_play` (
  `fdate` date NOT NULL,
  `fcoverid` varchar(40) NOT NULL COMMENT 'cover id',
  `fvideoid` varchar(40) NOT NULL COMMENT 'video id',
  `ftotal` int(100) NOT NULL,
  `fopt_time` timestamp  NOT NULL  ON UPDATE CURRENT_TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '数据库更新时间',
  PRIMARY KEY (`fdate`,`fcoverid`,`fvideoid`,`ftotal`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='。。。';
load data local  infile 'testin' ignore  into table video_play  fields terminated by ',' (fdate,fcoverid,fvideoid,ftotal,@fopt_time) set fopt_time=NOW();
或者将最后的改为
 fopt_time=CURRENT_TIMESTAMP
要注意这里 replace与ignore的用法
```
可以参见
- [timestamp-field-error-with-load-data-infile-with-python-on-linux](https://stackoverflow.com/questions/38269211/timestamp-field-error-with-load-data-infile-with-python-on-linux)
- [mysql-how-do-i-insert-now-date-when-doing-a-load-data-infile](https://stackoverflow.com/questions/9591170/mysql-how-do-i-insert-now-date-when-doing-a-load-data-infile)

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
## 一个月以前
```shell
DATE_SUB(CURDATE(), INTERVAL 1 DAY)
DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
datediff
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

## 批量关闭进程
```shell
sleep 10
ps -ef |grep display |awk '{print $2}' |xargs kill -9
```
## 常用控制语句
### If
```shell
#!/bin/bash

a=10
b=20
if [ $a -eq $b ]
then
   echo "a 等于 b"
elif [ $a -gt $b ]
then
   echo "a 大于 b"
elif [ $a -lt $b ]
then
   echo "a 小于 b"
else
   echo "没有符合的条件"
fi
```
### While
```shell
while condition
do
    command
done
例如：
echo '按下 <CTRL-D> 退出'
echo -n '输入你最喜欢的书名: '
while read book
do
    echo "是的！$book 是一部好书"
done
```

### For
```shell
#!/bin/bash
for var in 1 2 3 4 5
do
    echo "value is ${var}"
done
或者为for var in `seq 1 5`
```

### Case
```shell
#!/bin/bash

while : #开启无限循环
do
    echo -n "输入 1 到 5 之间的数字:"
    read aNum
    case $aNum in
        1|2|3|4|5) echo "你输入的数字为 $aNum!"
        ;;
        *) echo "你输入的数字不是 1 到 5 之间的! 游戏结束"
            break # break 表示退出循环; 如果使用continue关键字，则结束本次循环，继续执行循环后面的内容
        ;;
    esac
done
```
## 总结
"工欲善其事，必先利其器"。

## 参考文献
```shell

```
