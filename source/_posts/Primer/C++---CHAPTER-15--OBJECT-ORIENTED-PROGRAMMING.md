---
title: C++---CHAPTER-15---OBJECT-ORIENTED-PROGRAMMING
mathjax: true
date: 2019-09-04 21:53:40
tags: PRIMER
categories: C++
visible:
---

### 概述
* OOP的核心思想：
   * 数据抽象：类的接口与实现分离；
  * 继承：定义相似的类型并对其相似关系建模；
  * 动态绑定：一定程度上忽略相似类型的区别，而以同一的方式使用它们的对象。


* 继承 inheritance
有关与基类（base class）和派生类（derived class）的一个例子是：`Quote`类作为一个基类，`Quote`的对象表示按原价销售的书籍，`Quote`的一个派生类`Bulk_quote`，它的对象表示打折销售的书籍，基类定义的两个函数是：
* `isbn()`，返回书籍的ISBN编号。
* `net_price(size_t)`，返回书籍的实际价格。

可以看到，第二个函数跟具体的类型相关。类似这样的函数，基类希望它的派生类各自定义适合自身的版本，此时基类就将这些函数声明为虚函数（virtual function），而派生类必须在其内部对所有重新定义的虚函数进行声明。

```
Quote类
class Quote{
  public:
    std::string isbn() const;
    virtual double net_price(std::size_t n) const;
}
```

派生类通过类派生列表（冒号+访问说明符+逗号分割的基类列表）表明哪些是基类：
```
class Bulk_quote : Quote{
  public:
      double net_price(std::size_t) const Override;
}
```
* 动态绑定

### 定义基类和派生类
#####  定义基类

* 成员函数与继承

* 访问控制与继承

##### 定义派生类

* 派生类中的虚函数

* 派生类对象及派生类向基类的类型转换

* 派生类构造函数


* 派生类使用基类的成员

* 继承与静态成员


* 派生类的声明

* 被用作基类的类

* 防止继承的发生

#####  类型转换与继承
* 静态类型与动态类型

* 不存在从基类向派生类的隐式类型转换

* 在对象之间不存在类型转换


### 虚函数
* 对虚函数的调用可能在运行时才被解析

```
Quote base("0-201-82470-1", 50); //基类
print_total(count, base, 10); // 调用Quote::net_price
Bulk_quote derived("0-201-82470-1" ,50,  5,  0.19);
print_total(cout, derived, 10); //调用Bulk_quote::net_price
```

* 虚函数与默认实参：和其他函数一样，虚函数也可以拥有默认实参。通过基类的引用或指针调用函数，则使用基类中定义的默认参数。

  BestPractice:如果虚函数使用默认实参，则基类和派生类中定义的默认实参最好一致。


* 回避虚函数机制：即对虚函数的调用不进行动态绑定，而是强迫执行虚函数的某个特定版本。使用作用域符实现：
```
不管baseP实际指向的对象类型是什么，调用Quote的net_price函数

double undiscounted = baseP -> Quote::net_price(42);
```


### 抽象基类

### 构造函数与拷贝控制
如果一个类没有定义拷贝控制操作，则编译器将为它合成一个版本；

* 虚析构函数：继承体系中，删除一个指针，可能出现指针的静态类型与被删除对象的动态类型不符合的情况：
```
class Quote{
public:
    //如果删除一个指向派生类对象的基类指针，则需要虚析构函数
    virtural ~Quote() = default; //动态绑定析构函数
}
```

一般来说，如果一个类需要析构函数，那么它也同样需要拷贝和赋值操作，但是基类的析构函数是一个例外。

* 虚析构函数将阻止合成移动操作：如果一个类定义了析构函数，即是它用`=default`的形式使用了合成的版本，编译器也不会为这个类合成移动操作。

### 容器与继承

