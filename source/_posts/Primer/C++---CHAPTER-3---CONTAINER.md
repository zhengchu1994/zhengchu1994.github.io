---
title: C++---CHAPTER-3---CONTAINER
mathjax: true
date: 2019-09-04 21:53:15
tags: PRIMER
categories: C++
visible:
---


* 迭代器：
标准容器迭代器的运算符, 其中 `->` 运算符把解引用和成员访问两个操作结合在一起：
```
*iter 返回迭代器iter所指元素的引用
iter->mem   等价于 (*iter).mem ，解引用iter并获取该元素的名为mem的成员
++iter 令iter指示容器中的下一个元素
--iter 零iter指示容器中的上一个元素
iter1 == iter2   iter1 != iter2  判断两个迭代器是否相等
```
   -  迭代器的类型：
`const_iterator`和常量指针差不多：
```
string:: const_iterator it4;  //it4只能读字符，不能写字符

const vector<int> cv;
auto it2 = cv.begin();  // it2的类型是vector<int>::const_iterator
```
  *  警告：任何使用迭代器的循环体，不该向迭代器所属的容器添加元素，如`push_back`等。

* string与vector的迭代器支持更多运算符：
```
auto mid = vi.begin() + vi.size() / 2

if (it < mid)   // 迭代器的比较，返回difference_type类型的有符号整型数
```
------

### 数组
定义数组必须指定数组的类型，不允许用`auto`关键字由初始值的列表推断类型。
与`vector`一样，数组元素应为对象，不存在引用的数组。

不允许将数组的内容拷贝给其他数组作为初始值，也不能用数组为其他数组赋值：
```
int a[] = {0, 1, 2};
int a2[] = a;  //错误
a2 = a;   // 错误
```
复杂的数组声明：
```
int (*Parray)[10] = &arr; // Parray指向一个含有10个整数的数组
int (&arrRef)[10] = arr; // arrRef引用一个含有10个整数的数组
int *(&arry)[10] = ptrs; // arry是一个引用，arry引用的对象时一个大小为10的数组，数组的类型是指向int的指针。
```
* 数组下标的类型通常定义为`size_t`类型，一种机器相关的无符号类型。

* 指针和数组
编译器一般把数组转换成指针,数组名字为一个指向数组首元素的指针：
```
string *p2 = nums; // 等价于p2 = &nums[0]
```
使用`auto`推断数组得到的类型是指针，但是用关键字`decltype`关键字时上述转换不会发生：
```
int ia = {0,1,2,3,4,5,6,7,8,9};
auto ia2(ia); // ia2是一个整型指针，指向ia的第一个元素
decltype (ia) ia3 ={0,1,2,3,4,5,6,7,8,9}; //ia3是一个含有10个整数的数组 
```
* 指针也是迭代器
那么获得组数指针尾元素的操作这样：
```
int *e = &ia[10]; //指向不存在的尾元素下一位置的指针
```

* 为了获得头尾元素更简单，c++加入了`begin`和`end`函数：
```
int *beg = begin(ia);
int *last = end(ia);
```
* 指针运算
指向同一数组的两个指针相减，返回的类型是`ptrdiff_t` 类型，一种带符号类型。


* 警告：内置的下标运算符所用的索引值不是无符号类型。

