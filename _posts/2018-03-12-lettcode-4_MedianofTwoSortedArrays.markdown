---
layout: post
title:  "4_MedianofTwoSortedArrays"
date:   2018-04-13 23:22:28 +0800
categories: [lettcode]
---

* TOC
{:toc}

[TOC]

## 问题
[两个有序数组的中值](https://leetcode.com/problems/median-of-two-sorted-arrays/description/)

There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).


Examples：
```python
nums1 = [1, 3]
nums2 = [2]

The median is 2.0
```

```python
nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5
```
## 思路分析
参考这个[实现方案](https://www.cnblogs.com/zuoyuan/p/3759682.html)，采用递归的方法解决问题。关注递归结束的判别条件。得到一个第$k$大数字的普遍方法。

## 实际代码
Python
```python
class Solution(object):
    def getKth(self,nums1, nums2,k):
        len1=len(nums1)
        len2=len(nums2)
        if len1>len2:
            return self.getKth(nums2,nums1,k)
        if len1==0:
            return nums2[k-1]
        if k==1:
            return min(nums1[0],nums2[0])
        pa=min(k/2,len1)
        pb=k-pa
        if nums1[pa-1]<=nums2[pb-1]:
            return self.getKth(nums1[pa:],nums2,pb)
        else:
            return self.getKth(nums1,nums2[pb:],pa)

    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        len1=len(nums1)
        len2=len(nums2)
        if(len1+len2)%2==1:
            return self.getKth(nums1,nums2,(len1+len2)/2+1)
        else:
            return (self.getKth(nums1,nums2,(len1+len2)/2)+self.getKth(nums1,nums2,(len1+len2)/2+1))*0.5
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
1.如果是两个无序数组？
## 总结
实际上是得到一个第$k$个大的数字的普遍方法。
