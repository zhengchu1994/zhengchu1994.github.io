---
title: C++---CHAPTER-7---CLASS
mathjax: true
date: 2019-09-04 21:53:32
tags: PRIMER
categories: C++
visible:
---

* 引入`this`
调用一个成员函数，编译器用请求该函数的对象地址初始化`this`，所以`this`的目的总是指向这个对象，因此`this`是一个常量指针。

* 引入`const`成员函数
以下是`Sales_data`类的一个成员函数的定义,参数列表之后的`const`作用是修改隐式`this`指针的类型，因为默认情况下，`this`的类型是指向类类型非常量版本的常量指针。所以默认情况不能把`this`绑定到一个常量对象上。这样使用`const`的成员函数被称为常量成员函数。
```
std::string isbn() const {return bookNo;}
```

* 类作用域
编译器首先编译成员的声明，然后是成员函数体，所以成员函数体可以随意使用类中的其他成员无须在意成员的顺序。

* 定义一个返回`this`对象的函数
函数类似于某个内置运算符时，应该令该函数的行为尽量模仿这个运算符。内置赋值运算符把它的左侧运算对当左值返回，意味着这些函数返回的是对象本身而非对象的副本，因此combine必须返回引用类型。
```
Sales_data& Sales_data::combine(const Sales_data &rhs)
{
  units_sold += rhs.units_sold;
  revenue += rhs.revenue;
  return *this;  //解引用指针获取执行该函数的对象。
}
```
* 类相关的非成员函数
如果函数在概念上属于类但是不定义在类中，则它一般应与类声明在同一个头文件。
默认情况下拷贝类的对象，拷贝的是对象的数据成员。


* 构造函数
构造函数不能声明为`const`,直到构造函数完成初始化过程，对象才能取得"常量"属性。因此，构造函数在`const`对象的构造过程中可以向其写值。
  * 合成的默认构造函数：类没有显示的定义构造函数，那么编译器会为我们隐式地定义一个合成的默认构造函数，安照类内初始值初始化成员，没有的话则默认初始化成员。只有当类没有声明任何构造函数的时候，才会默认构造函数。使用`= default`要求编译器生成构造函数。

  * 构造函数初始值列表：
```
Sales_data = (const std::string &s, unsigned n, double p): bookNo(s), units_sold(n), revenue(p*n) {}
```


* 一个`const`成员函数如果以引用的形式返回`*this`指针，那么它的返回类型将是常量引用。

基于`const`的重载
如下，当一个成员调用另一个成员的时候，`this`指针在其中隐式地传递，当`display`的非常量版本调用`do_display`的时候，它的`this`指针隐式地从指向非常量的指针转换成指向常量的指针。
```
#include <iostream>
#include <string>
using namespace std;

class Screen{
public:
  typedef std::string::size_type pos;
  
  Screen() = default; // 因为要写另一个构造函数
  Screen(pos ht, pos wd, char c): height(ht), width(wd), contents(ht* wd, c) {}
  
  char get() const
  {return contents[cursor];} // 类内部声明定义的隐式内联函数
  
  inline char get(pos ht, pos wd) const; // 显示内联

  Screen &set(char);
  SCreen &set(pos, pos, char);
  
  Screen &move(pos r, pos c); //能在之后被设为内联函数

  Screen &display(std::ostream &os)
  {do_display(os); return *this;}

  const Screen &display(std::ostream &os) const
  {do_display(os); return *this;}


private:
  pos cursor = 0;  //光标的意思
  pos height = 0, width = 0;
  std::string contents;

  void do_display(std::ostream &os) const{os << contents;}

};

inline  // 在函数的定义处指定为内联函数
Screen &Screen::move(pos r, pos c)
{
  pos row = r * width;
  cursor = row + c; 
  return *this;  //左值
}

char Screen::get(pos r, pos c) const
{
  pos row = r * width;
  return contents[row + c];
}

inline
Screen &Screen::set(char c)
{
  contents[cursor] = c;
  return *this;
}

inline 
Screen &Screen::set(pos r, pos col, char ch)
{
  contents[r*width +col] =ch;
  return *this;
}

int main() {
  Screen myscreen(5, 5, 'F');
  char ch =myscreen.get();
  ch = myscreen.get(0, 0);
  cout << ch << endl;
}
```
注：转换成常量：指向T类型的指针或引用分别转换成指向`const T`的指针或引用：

```
int i ;
const int & j = i; //非常量转换成const int 的引用
const int *p = &i; //非常量的地址转换成const的地址
```

* 类的声明
```
class Screen; // Screen类的声明
```
前项声明（是一种不完全类型）的使用：可以定义指向这种类型的指针或者引用，声明（但不能定义）以不完全类型作为参数或者返回类型的函数。
必须完成类的定义，编译器才能知道存储数据成员需要多少空间。因为只有当类全部完成后才算被定义，因此不能有一个类的成员类型是该类自己。
但是类的名字出现后，声明了该类，因此类允许包含自身类型的引用或者指针：
```
class Link_Screen{
  Screen window;
  Link_Screen *next;
  Link_Screen *prev;
}
```

* 友元函数
一个类指定了友元类，则友元类的成员函数可以访问此类的所有成员。

* 类的作用域

* 构造函数再谈

构造函数初始值列表的必要性：如果成员是`const`、引用，或者属于某种未提供默认构造函数的类类型，必须通过构造函数初始值列表为这些成员提供初值。
```
class ConstRef{
public:
  ConstRef(int ii);
private:
  int i;
  const int c;
  int &ri;
}

ConstRef::ConstRef(int ii)
{
  i = ii; //正确
  c = ii; //错误 ：不能给const复制
  ri = i; //错误 ：ri没被初始化
}
```
那么成员初始化的顺序是：与他们在类中的定义一致。

默认实参与构造函数：可以重写一个使用默认实参的构造函数
```
class Sales_data{
public:
  //定义默认构造函数,接受一个字符串初始值
  Sales_data(std::string s = " ") bookNo(s) {}
private:
  ....
}

```

* 委托构造函数：delegating constructor,委托构造函数就是把自己的初始化全部交给其他构造函数，受委托的构造函数初始值列表和函数体执行后才轮到委托构造函数。


* 类的静态成员：与类直接相关的成员，不包含`this`指针，静态成员函数不能声明为`const`的，不能再`static`函数体内使用`this`指针。
```
class Account{
  public:
    void calculate() {amount += amount * interestRate;}
    static double rate() {return interestRate;}
    static void rate(double);
  private:
    std::string owner;
    double amount;
    static constexpr int period = 30;  //period是常量表达式
    static double interestRate;
    static double initRate();
}

// static关键字出现在类内部
void Account::rate(double newRate)  
{
  interestRate = newRate;
}
```
由上看出，必须在内外初始化每个静态成员。
类内初始化必须要求静态成员是字面值常量类型的`constexpr`。

* 静态成员与普通成员的不同：
1. 静态数据成员的类型可以就是她所属的类类型；
2.可使用静态成员做默认实参；
```
class Account{
  public:
    void calculate() {amount += amount * interestRate;}
    static double rate() {return interestRate;}
    static void rate(double);
  private:
    std::string owner;
    double amount;
    static double interestRate;
    static double initRate();
}

void Account::rate(double newRate)
{
  interestRate = newRate;
}
```
