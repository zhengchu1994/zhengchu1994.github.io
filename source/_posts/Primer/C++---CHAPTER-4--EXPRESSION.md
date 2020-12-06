---
title: C++---CHAPTER-4---EXPRESSION
mathjax: true
date: 2019-09-04 21:53:22
tags: PRIMER
categories: C++
visible:
---


####  类型转换
* 转换成常量：
```
int i;
const int   &j = i; //非常量转换成const int 的引用
const int *p = &i; //非常量的地址转换成const的地址
int &r = j, *q =p;  //错误，不能用const转换为非常量
```


* 强制类型转换的形式：
```
cast-name<type>(expression);
```

* `static_cast`：只要不包含低层`const`就可以使用`static_cast`:
```
double slope = static_cast<double>(j) / i; //转换后做浮点数除法

void *p = &d; // 任何非常量对象的地址都能存入* void
double *dp = static_cast<double*>(p); //将void*转换回初始的指针类型
```

* `const_cast`：用于函数重载；
