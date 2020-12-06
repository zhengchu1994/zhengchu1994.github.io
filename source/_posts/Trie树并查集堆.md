---
title: 算法Basic--Trie树并查集堆
mathjax: true
date: 2019-10-20 21:53:32
tags: [Acwing, Algorithm]
categories: 算法与数据结构
visible:
---




#### 手写堆

* 需要的操作：

  * 插入一个数：heap[++size] = x; up(size);

  * 求集合当中的最小值 ：heap[1]；

  *  删除最小值 ：heap[1] = heap[size]； size --; down(1)

  * 删除任意一个元素：heap[k] = heap[size]; size-- ; down(k); up(k)；

  * 修改任意一个元素：heap[k] = x; down(k); up(k);

     

* 结构：完全二叉树；

* 小根堆定义：根节点小于左右节点（递归定义)。

* 根节点为x：左儿子是2x , 右儿子是 2x+1。

两个操作：down（x）、up（x）.





#### 字符串哈希

* 字符串前缀哈希法。

字符串看成p进制的数，

比如`ABCD`，第一位为`A`，第2位为`B`,... ，其中`A`视为 1，`B`视为2，

那么`ABCD`可以看为p进制上的`1234`，那么有`(1*p3 + 2 * p2 + 3*p1 + 4*p0) mod Q`，取模`Q`是因为

前面的数和太大，这样就可以把该字符串映射到`0~Q`之间。

* 每个字符都不能映射成0；
* `Rp`足够好，不存在冲突。
* p=131或13331，Q=2^64.



求前缀的公式： h(i) = h(i- 1) * P + str[i]

求【L,R】子串哈希值的公式：h(R) - h(L) *  P^(R - L + 1)

