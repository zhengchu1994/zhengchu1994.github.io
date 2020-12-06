---
title: C++---CHAPTER-2---VARIABLE
mathjax: true
date: 2019-09-04 21:53:15
tags: PRIMER
categories: C++
visible:
---


* 列表初始化：
使用列表初始化且初始值存在丢失信息的风险，则编译器报错：
```
int unit_sold = {0};  // list initialization
long double ld =3.14159;
int a{id}, b = {id}; //错误
int c(id), d = ld; // 正确
```

* 默认初始化：
定义于任何函数体之外的内置类型变量被初始化为0，定义在函数体内部的内置类型变量不被初始化。

* 声明与定义：
变量只能定义一次，但能被多次声明；
关键字`extern`表示声明变量；
任何包含了显示初始化的声明即成为定义；
函数内部初始化一个由extern关键字标记的变量，会报错。
```
extern int i; //声明而非定义i
int j; //声明并定义j
extern double pi = 3.1416; // 定义
``` 
* 复合类型：指基于其他类型定义的类型，如引用，指针等。
声明语句的组成：基本数据类型 + 声明符，每一个声明符命名了一个变量并指定该变量为与基本数据类型有关的某种类型。
  - 引用类型：引用另一种类型,是已经存在的对象的别名，因此不是对象，所以必须初始化。
```
int ival = 1024;
int &refVal = ival; 
int &refVal2; // 报错
```

  - 指针：
```
int ival = 42;
int *p = &ival;  // &取地址符，p存放变量ival的地址。
cout << *p ; // *解引用符，得到p所指的对象。
```
  - 空指针：不指向任何对象.
```
int *p1 = nullptr;
int *p2 = 0;
int *p3 = NULL; // 需要include<cstdlib>
```

* 指向指针的指针
```
int ival = 1024;
int *pi = &ival; // pi是指向int的指针
int **ppi = &pi; // ppi是指向int指针的指针
```
* 指向指针的引用：
指针是对象，存在对指针的引用，但是引用不是对象，不存在指向引用的指针；
离变量最近的符号对变量的类型有最直接的影响，因此r是一个引用。
```
int i = 42;
int *p; 
int *&r = p ; //r是一个对指针p的引用
```
### const
`const`限定的变量，其值不能改变，因此必须初始化。
`const`变量初始化另一个对象不会改变这个变量：
```
int i = 42;
const int ci = i;
int j = ci; //正确；拷贝完成，新的对象和原来的对象就没有关系了；
```

* **默认状态下，const对象仅在文件内有效**，因此，当多个文件都要用这个`const`变量的时候，在每个文件都要定义该`const`变量，
只在一个文件中定义`const`,而在其他文件中声明且使用它的办法：在定义前加`extern`关键字：
```
//file.cc 定义并初始化了一个常量
extern const int bufSize = fcn();
// file_1.h头文件
extern const int bufSize; // 与file.cc中定义的bufSize是同一个
```

* 对常量的引用

```
const int ci = 1024;
const int &r1=ci; //正确
int &r2 =ci; //错误
```

对`const`的引用可能引用一个并非const的对象：

```
int i = 42;
int &r1 = i;
const int &r2 =i; 
r1 = 0; //正确
r2 = 0; //错误
```
* 指向常量的指针
```
const double pi = 3.1415;
double *ptr = &pi; //错误
const double *cptr = &pi; //正确


double dval = 3.14;
cptr = &dval; //正确
```

* `const`指针 ：形式为 `*const`，说明指针是一个常量：
```
int errNumb = 0;
int *const curErr = &errNumb; //只能指向errNumb 

const double pi = 3.14159;
const double *const pip = &pi; // pip是一个指向常量对象的常量指针

*pip = 2.72; // 错误，pip是一个指向常量的指针
* curEr = 0; // 正确，所指对象的值重置
```
* 顶层const
顶层`const` 表示指针本身是个常量，底层`const`表示指针所指的对象是一个常量。
指针类型即可以是顶层`const`也可以是低层`const`：
```
int i =0 ;
int *const p1 = &i; // 不能改变p1的值，是一个顶层const
const int ci = 42; //不能改变ci的值，是一个顶层const
const int *p2 = &ci; // 允许改变p2的值，这是一个低层const
const int *const p3 = p2; //靠右边的const是顶层const，靠左边的是低层const
const int &r =ci; // 用于声明引用的cosnt都是低层const
```



* constexpr常量表达式：变量声明为`constexpr`类型以便编译器验证变量是否是一个常量表达式：
```
constexpr int mf = 20;
cosntexpr int limit = mf +1;
constexpr int sz = size(); //只有size是一个constexpr函数时才是一条正确的声明语句
```


`constexpr`声明中如果定义指针，限定符`constexpr`仅对指针有效，与指针所指的对象无关：
```
const int *p = nullptr; // p是一个指向整型常量的指针
constexpr int *q = nullptr; //q是一个指向整型的常量指针 ，顶层const
```
与其他常量一样，`constexpr`能指向常量与非常量：
```
int j = 0;
constexpr int i = 42; // i的类型是整型常量

constexpr const int *p = &i ; // p是常数指针，指向整型常量i
constexpr int *p1 = &j; //p1是常量指针，指向整数j
```

* 类型别名
```
typedef double wages; //wages 是double的同义词
using SI = Sales_item; // SI是Sales_item的同义词
```


* auto类型说明符

一条语句声明多个变量，但是变量的初始数据类型必须一样：
```
auto i =0, *p = &i; //正确：i是整数、p是整型指针
auto sz=0, pi = 3.14; //错误，类型不一样
```
`auto`类型和初始值类型并不一定一样，如忽略顶层`const`，保留低层`const`：
```
int i =0, &r = i; 
auto a = r; // a是一个整数 

const int ci = i, &cr  = ci;
auto b = ci; // b是一个整数（ci的顶层const特性被忽略）
auto c = cr; // c是一个整数 （cr是ci的别名，ci本身是一个顶层const）
auto d = &i; // d是一个整型指针（整数的地址就是指向整数的指针）
auto e = &ci; // e是一个指向整数常量的指针（对常量对象去地址是一种低层const）
```
希望推断出的`auto`类型是一个顶层`const`,需要明确：
```
const auto f = ci; //  ci的推演类型是int，f是const int
```
引用的类型设为`auto`,上述规则依然适用：
```
auto &g = ci; //g是一个整型常量引用，绑定到ci
auto &h = 42; //错误，不能为非常量引用绑定字面值
const auto &j = 42; //正确：可以为常量引用绑定字面值
```

* `decltype`类型指示符：推断要定义的变量的类型
`decltype`使得变量，则返回该变量的类型：
```
const int ci = 0, &cj = ci;
decltype(ci)  x = 0; //x的类型是const int
decltype(cj) y = x; // y的类型是const int&, y绑定到变量x
```
`decltype((variable))`的结果永远是引用：
```
int i = 42;
decltype((i)) d; //错误，d是int&，必须初始化；
```

### 术语表

| 声明（declaration）   	| 声称存在一个变量、函数或是别处定义的类型。名字必须在定义或者声明之后才能使用。              	|
|-------	|----------------------------	|
| 定义（defination）    	| 为某一特定类型的变量申请存储空间，可以选择初始化该变量。  名字必须在定义或者声明之后才能使用。            	|
