---
layout: post
title:  "1_TwoSum"
date:   2018-04-13 21:34:28 +0800
categories: [lettcode]
---

* TOC
{:toc}

[TOC]

## 问题
[寻找数组中满足指定和的两个数字](https://leetcode.com/problems/two-sum/description/)

Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.

Examples：
```python
Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
```


## 思路分析
在一个数组中，寻找两个数字，以致于其和等于指定的结果。直接来看，仿佛需要两次遍历，判断数组中的数字两两之和是否满足条件，这样复杂度过高，为了简化，可以借用一个map，每次遍历一个数，都去map中查找指定结果与当前数的差值。如果找到，则结束，否则将当前数的值以及对应索引放进map。

## 实际代码
Python
```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        res=[0,0]
        data={}
        for i,num in enumerate(nums):
            temp=target-num
            if temp in data:
                res[0]=data[temp]
                res[1]=i
                return res
            else:
                data[num]=i
```

C++
```code

```

JAVA
```code

```

GO
```code

```


SCALA
```code

```
## 思路拓展
1.如果目标是找出三个数，使得他们的和等于目标结果咧?即ThreeSum问题。
2.如果目标是找出两个数，使得他们的乘积等于目标结果咧？即TwoMultiply问题。
3.能找出所有满足条件的数之集合么？
## 总结
通过借用了map的数据结构完成了解答。
