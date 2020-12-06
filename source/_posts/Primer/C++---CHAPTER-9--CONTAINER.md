---
title: C++---CHAPTER-9---CONTAINER
mathjax: true
date: 2019-09-04 21:53:33
tags: PRIMER
categories: C++
visible:
---

* 顺序容器

| vector       	| 尾部之外的位置插入或删除元素可能很慢          	|
|--------------	|-----------------------------------------------	|
| deque        	| 头尾位置插入、删除速度很快                    	|
| list         	| 任何位置插入、删除速度很快                    	|
| forward_list 	| 只支持单向顺序访问,任何位置插入、删除都很快。 	|
| array        	| 固定大小数组。不能添加删除元素                	|
| string       	| 随机访问快、尾部插入删除很快。                	|

* 容器操作

| 类型别名            	|                                                    	|
|---------------------	|----------------------------------------------------	|
| iterator            	| 此容器类型的迭代器类型                             	|
| const_iterator      	| 读取元素，不能修改元素的迭代器类型                 	|
| size_type           	| 无符号类型，保存此容器类型的最大可能容器的大小     	|
| difference_type     	| 带符号整数类型，两个迭代器之间的距离               	|
| value_type          	| 元素类型                                           	|
| reference           	| 元素的左值类型；与value_type&含义相同              	|
| const_reference     	| 元素的const左值类型（vonst vlue_type&）            	|
| 构造函数            	|                                                    	|
| C c;                	| 默认构造函数，构造空容器                           	|
| C c1(c2);           	| 构造c2的拷贝到c1                                   	|
| C c(b,e);           	| 构造c,将接待器b和e范围的元素拷贝到c（不支持array） 	|
| C c{a, b, c,...}    	| 列表初始化c                                        	|
| 赋值与swap          	|                                                    	|
| C1 = c2             	|                                                    	|
| C1 = {a, b, c,...}  	|                                                    	|
| a.swap(b);swap(a,b) 	| 交换a与b的元素                                     	|
| 大小                	|                                                    	|
| c.size()            	| c中元素的数目（不支持forward_list）                	|
| c.max_size()        	| c可保存的最大元素数目                              	|
| c.empty()           	|                                                    	|
| 添加或删除元素       	| 在不同容器中，操作的接口不同                       	|
| c.insert(args)      	| 将args中的元素拷贝到c                              	|
| c.emplace(inits)    	| 使用inits构造c中的一个元素                         	|
| c.erase(args)       	| 删除args指定的元素                                 	|
| c.clear()           	| 删除c中的所有元素，返回void                        	|
| 关系运算符          	| 所有容器都支持：如==、！=                          	|
| 获取迭代器          	|                                                    	|
| c.cbegin(),c.cend() 	| 返回const_iterator                                 	|
| reverse_iterator    	| 逆序寻址的迭代器                                   	|


#### 容器定义和初始化
```
C seq(n);       // seq包含n个元素，都被值初始化
C seq(n, t);  // seq包含n个值为t的元素
list<string> authors = {"Milton", "Austen"}; //列表初始化
```
* notes: 1. 将一个容器初始化为另一个容器的拷贝时，两个容器的容器类型和元素类型都必须相同。
 2. 顺序容器的构造函数才接受大小参数，关联容器并不支持。


* 标准库 `array`
使用`array`必须指定元素类型和大小：
```
array<int, 42> // 类型为：保存42个int的数组
 array<int>:: size_type j;  //错误
```
不能对内置数组类型进行拷贝，但是`array`类型是可以的：
```
array<int ,10> digits = {0,1,2,3,4,5,6,7,8,9};
array<int, 10> copy = digits; // right
```
#### 容器赋值运算
赋值和`swap`
```
array<int, 10> a1 = {...};
array<int 10> a2 = {0}; //所有元素均为 0
a2 = {0}; //错误，不能将一个花括号列表赋予数组
```

```
// 通常swap比从c2向c1拷贝元素要快
swap（c1,c2);
c1.swap(c2);  

seq.assign(b, e); // 将seq中的元素替换为迭代器b和e所表示的范围的元素，迭代器b、e不能指向seq中的元素
seq.assign(il); // 将seq中的元素替换为初始化列表il中的元素
seq.assign(n,t); // 将seq中的元素替换为n个值为t的元素
```
* 警告：赋值相关运算会导致指向左边容器内部的迭代器、引用和指针失效。而swap操作将容器内容交换不会导致指向容器的迭代器、引用和指针失效（array和string除外）。

* 顺序容器的`assign`
允许我们从一个不同但相容的类型赋值，或者从容器的一个子序列赋值。`assign`操作用参数所指定的元素（的拷贝）替换左边容器中的所有元素。
```
list<string> names;
vector<const char*> oldstyle;
names = oldstyle; //错误，容器类型不匹配
names.assign(oldstyle.cbegin(), oldstyle.cend()); //正确 可以将const char* 转换为string
```
#### 向顺序容器添加元素

| c.push_back(t) , c.emplace_back(args) | c的尾部创建一个值为t或由args创建的元素。  	|
|--------------	|-----------------------------------------------	|
| c.insert(p,t) , c.emplace(p,args) 	|  在迭代器p指向的元素之前创建一个值为p或由args创建的元素。                	|
| c.insert(p, n , t)       	| 在迭代器p指向的元素之前插入n个值为t的元素。返回新添加的第一个元素的迭代器，若n为0，返回p      	|
| c.insert(p, b, e)	| 将迭代器b、e指定的范围内的元素插入到迭代器p指向的元素之前。b、e不能指向c中的元素。 返回新添加的第一个元素的迭代器，若n为0，返回p      	|
| c.insert(p , il)      	| il是一个花括号保卫的元素值列表，将这些值插入到p指向的元素之前。  返回新添加的第一个元素的迭代器，若n为0，返回p      	|


* insert的返回值
下面的例子，`iter`每次都指向新加入元素的位置：
```
list<string> lst;
auto iter = lst.begin();
while(cin>>word)
  iter = lst.insert(iter, word); //等价于调用 push_front
```


* 使用`emplace`操作
假设`c`里保存的是`Sales_data`成员：
```
c.emplace_back("970", 25, 15.99);
c.push_back("970", 25, 15.99); //错误，没有接受三个参数的push_back
c.push_back(Sales_data("970", 25, 15.99));
```
在调用`emplace_back`时，会在容器管理的内存空间中直接创建对象，而调用push_back则会创建一个局部临时对象，并压入容器中。

#### 在顺序容器汇总访问元素

* 容器中没有元素，访问操作是未定义的。
* 包括`array`在内每个顺序容器都有一个`front`成员函数，除去`forward_list`之外的所有顺序容器都有`back`成员函数。
```
  vector<int> c={1,2,3,4,5,6};
  if (!c.empty()){
    // val和val2是c中第一个元素值的拷贝
    auto val = *c.begin(), val2 = c.front();
    // val3和val4是c中最后一个元素值的拷贝
    auto last =c.end();
    auto val3 =*(--last); // 不能递减forward_list迭代器
    auto val4 = c.back();
    cout << val << " " << endl;
    cout << val2 << " " << endl;
    cout << val3 << " " << endl;
    cout << val4 << " " << endl;
  }
```
* 访问成员函数返回的是引用：`front`、`back`、`at`、下标都是返回引用，如果容器是一个`const`对象，返回值是`const`的引用。
```
  vector<int> c={1,2,3,4,5,6};
  if (!c.empty()){
    c.front() = 42;
    auto &v = c.back(); // v是c.back()的一个引用
    v = 1024;
    cout << c.back() << endl;
    auto v2 = c.back(); // v2不是一个引用，它是c.back()的一个拷贝
    v2 = 0;
    cout <<c.back();
  }


1024
1024
```
* 下标操作和安全的随机访问：`at`在下标越界的情况下，会抛出一个`out_of_range`的异常。
```
vector<string> svec;
cout << svec[0]; // 运行时错误
cout << svec.at(0); //抛出一个异常
```


#### 顺序容器的删除操作
* *注意*：删除`deque`中除首尾位置之外的任何元素都会是所有迭代器、引用、指针失效。指向`vector`、`string`中删除点之后的迭代器、引用、指针失效。

| c.pop_back()| 删除c中的尾元素。c为空，函数行为未定义。函数返回void  	|
|--------------	|-----------------------------------------------	|
| c.pop_front()	|  删除c中的首元素。c为空，函数行为未定义。函数返回void                	|
| c.erase(p)      	| 删除迭代器p指定的元素，返回一个指向被删除元素之后元素的迭代器，若p是尾后迭代器，则函数行为未定义  	|
| c.erase(b, e)    	| 将迭代器b、e指定的范围内的元素删除。若e是尾后迭代器，则函数返回尾后迭代器      	|
| c.clear()    	| 删除所有元素，返回void    	|

#### 改变容器的大小
`resize`增大或缩小容器，`array`不支持。如果当前大小大于所要求的大小，容器后部的元素被删除；如果当前大小小于新大小，新元素会添加：
```
list<int> ilist(10, 42); // 10个int：每个值为42
ilist.resize(15); // 将5个值为0的元素添加到ilist尾部
ilist.resize(25, -1); //  将10个值为-1的元素添加到ilist末尾
ilist.resize(5); // 从ilist末尾删除20个元素
```
#### 
