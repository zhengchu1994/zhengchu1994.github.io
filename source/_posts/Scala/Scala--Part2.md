---
title: Scala-part2
mathjax: true
date: 2020-06-11 13:43:00
tags: Scala
categories: Scala
visible:

---



# Scala-part2

## >函数<



* 在scala中，函数是可重用的命名表达式。
* 尽可能构建纯函数，类似数学意义上的函数。

1. 无输出函数：

```scala
// 无输出函数
scala> def hi = "hi"
def hi: String

scala> hi
val res75: String = hi


scala> def multiplier(x: Int, y: Int): Int = {x * y}
def multiplier(x: Int, y: Int): Int

scala> multiplier(6, 7)
val res77: Int = 42

scala> def safeTrim(s: String): String={
     | if(s == null) return null
     | s.trim()
     | }
def safeTrim(s: String): String

scala> val t = safeTrim("   abc  ")
val t: String = abc

```





**过程**：没有返回值的函数，**当一个函数没有显式的返回类型，而且最后是一个语句**，scala推导这个喊你数的返回类型是**Unit**，表示没有值。

```scala
scala> def log(d: Double): Unit = println(f"Got value $d%.2f")
def log(d: Double): Unit

scala> log(2.23454)
Got value 2.23

```



#### 空括号定义的函数

**scala约定当函数会修改其范围外的数据时，定义时就应该加空括号**。

```scala
scala> def hi(): String = "hi"
def hi(): String

scala> hi()
val res79: String = hi
```



#### 块表达式调用函数

中间不会保存为局部值，提高效率。



#### 递归函数

`@annotation.tailrec`标识判断递归函数必须为**尾递归（tail-recursion）**，尾递归使得递归调用不使用额外的栈空间，而是使用当前的栈空间。

尾递归的最后语句是该递归函数本身。

```scala
scala> @annotation.tailrec
     | def power(x: Int, n: Int, t: Int = 1) : Int = {
     | if (n < 1) t
     | else power(x, n - 1, x * t)
     | }
def power(x: Int, n: Int, t: Int): Int

scala> power(2, 8)
val res84: Int = 256
```



#### nested函数

即使**内部函数**与**外部函数**同名且参数数目相同，**內部函数优先级高**于外部。

```scala
scala> def max(a: Int, b: Int, c: Int) = {
     |  def max(x: Int, y: Int) = if (x > y) x else y
     |  max(a, max(b, c))
     | }
def max(a: Int, b: Int, c: Int): Int

scala> max(42, 181, 19)
val res85: Int = 181
```



#### vararg参数



参数类型+`*`即代表可以传入多个同类型参数。

```scala
scala> def sum(item: Int*): Int = {
     |  item.sum
     | }
def sum(item: Int*): Int

scala> sum(10, 20, 69)
val res87: Int = 99

```



#### 类型参数



类型匹配规则：函数名的后面紧跟`[type]`表示。

```scala
scala> def identity[A](a: A): A = a
def identity[A](a: A): A

scala> identity("Hekko")
val res90: String = Hekko

scala> identity(12312.123)
val res91: Double = 12312.123

```





#### 首类函数

* 首类函数（first-class）：函数不仅能得到声明和调用，还可以**作为一个数据类型**用在这个语言的任何地方。
* 高阶函数（higher-order）：接受其他函数作为参数，或使用函数作为返回值，如map/reduce。
* 声明式编程（Declarative Programming）：要求使用高阶函数或其他机制声明要做的工作，而不手动实现。
* 命令式编程（Imperative Programming）：you know that bro～



```scala
scala> def double(x: Int): Int = x * 2
def double(x: Int): Int

//方法1: myDouble等于一个数值，但是可以调用double函数
scala> val myDouble: Int => Int = double
val myDouble: Int => Int = $Lambda$1180/0x000000080112a040@50de907a

scala> myDouble(5)
val res2: Int = 10

//函数可以赋值给变量
scala> val myDoubleCopy = myDouble
val myDoubleCopy: Int => Int = $Lambda$1180/0x000000080112a040@50de907a

scala> myDoubleCopy(5)
val res5: Int = 10

//方法2:用通配符_ : 下划线表将来的一个函数调用，它返回一个函数值，存入变量mDouble中。
scala> val mDouble = double _
val mDouble: Int => Int = $Lambda$1188/0x000000080112e040@4d992634

scala> mDouble(20)
val res7: Int = 40
```



多个参数传递用`()`：

```scala
scala> def max(a: Int, b: Int) = if (a > b) a else b
def max(a: Int, b: Int): Int
//(输入参数类型) =>输出参数类型 = 函数
scala> val maximize: (Int, Int) => Int = max
val maximize: (Int, Int) => Int = $Lambda$1189/0x0000000801138840@4a1bb556

scala> maximize(9, 9)
val res10: Int = 9
```





#### 高阶函数

```scala
scala> def safeStringOp(s: String, f: String => String) = {
     | if(s != null) f(s) else s
     | }
def safeStringOp(s: String, f: String => String): String

scala> def reverser(s: String) = s.reverse
def reverser(s: String): String

scala> safeStringOp(null, reverser)
val res13: String = null

scala> safeStringOp("zc2077", reverser)
val res17: String = 7702cz

```







#### 函数字面量（lambda函数）

```scala

scala> val maximize = (a: Int, b: Int) => if(a > b) a else b
val maximize: (Int, Int) => Int = $Lambda$1227/0x000000080115e840@413bf74f

scala> maximize(2020, 2077)
val res20: Int = 2077


scala> safeStringOp("zc2077", (s: String) => s.reverse)
val res22: String = 7702cz

// 因为高阶函数已经申明了类型，可以去掉显式类型
//进一步可以去掉小括号，因为输入参数只有一个
scala> safeStringOp("zc2077", s => s.reverse)
val res24: String = 7702cz
```



#### 占位符语法

* 使用情况：
  * 函数的显式类型在字面量之外指定；
  * 参数最多只使用一次；



* 多个`_`按位置替换输入参数：

```scala
scala> def combination(x: Int, y: Int, f: (Int, Int) => Int) = f(x, y)
def combination(x: Int, y: Int, f: (Int, Int) => Int): Int

scala> combination(23, 12, _ * _ )
val res29: Int = 276

scala> combination(23, 12, _ + _ )
val res31: Int = 35

```



* 输入与输出类型都进行参数化：

```scala
//A:输入类型；B：输出类型
scala> def tripleOp[A, B](a: A, b: A, c: A, f: (A, A, A) => B) = f(a, b, c) 
def tripleOp[A, B](a: A, b: A, c: A, f: (A, A, A) => B): B

scala> tripleOp[Int, Int](3, 4, 5, _ * _ + _)
val res33: Int = 17

```





#### 函数部分利用、函数柯里化

```scala

scala> def factorOf(x: Int, y: Int) = y % x == 0
def factorOf(x: Int, y: Int): Boolean

//函数部分利用
scala> def multipleOf3 = factorOf(3, _: Int)
def multipleOf3: Int => Boolean

scala> multipleOf3(6)
val res37: Boolean = true
```





```scala
//多个参数表的函数视为--->多个函数的一个链
scala> def factorOf(x: Int)(y: Int) = y % x == 0
def factorOf(x: Int)(y: Int): Boolean

scala> val isEven = factorOf(2) _
val isEven: Int => Boolean = $Lambda$1255/0x000000080117f840@1956d3a2

scala> isEven(32)
val res39: Boolean = true
```





#### 传名参数

形式：	`<identifier>: => <type>`

* 若传递函数值，每次在函数內使用传递的参数时都会调用该函数。

```scala

scala> def doubles(x: => Int) = {
     | println("Now doubling" + x)
     | x * 2
     | }
def doubles(x: => Int): Int
//传递常规值，正常操作
scala> doubles(3)
Now doubling3
val res40: Int = 6
　
scala> def f(i: Int) = {println(s"Hello from f($i)"); i}
def f(i: Int): Int

//传递函数值，由于doubles内部两次调用参数x，因此f函数也会调用两次
scala> doubles( f(8) )
Hello from f(8)
Now doubling8
Hello from f(8)
val res41: Int = 16

```





#### 偏函数

感觉像是强制匹配函数，只能部分应用于输入数据的函数，比如除法函数不能除以0就是偏函数。

* 偏函数：一系列`case`模式，输入数据必须匹配至少一个

```scala
scala> val statusHanndler: Int => String = {
     |  case 200 => "Okay"
     |  case 400 => "Your Error"
     |  case 500 => "Our error"
     | }
val statusHanndler: Int => String = $Lambda$1271/0x000000080117d840@4123788

scala> statusHanndler(200)
val res43: String = Okay
```



#### 调用字面量

```scala
//1.穿值参数，第二个参数是一个函数，表示用表达式块语法调用
scala> def safeStringOp(s: String)(f: String => String) = {
     |  if(s != null) f(s) else s
     | }
def safeStringOp(s: String)(f: String => String): String


scala> val uuid = java.util.UUID.randomUUID.toString()
val uuid: String = ec4be3aa-7d05-4f89-9ecb-57a75a0214e1

scala> uuid
val res46: String = ec4be3aa-7d05-4f89-9ecb-57a75a0214e1

// 第二个参数用表达式块语法调用
scala> val timedUUID = safeStringOp(uuid) { s =>
     | val now = System.currentTimeMillis
     | val timed = s.take(24) + now
     | timed.toUpperCase
     | }
val timedUUID: String = EC4BE3AA-7D05-4F89-9ECB-1591853574681

```



