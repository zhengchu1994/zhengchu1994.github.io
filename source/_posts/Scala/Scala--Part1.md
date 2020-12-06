---
title: Scala-part1
mathjax: true
date: 2020-06-08 16:00:00
tags: Scala
categories: Scala
visible:
---





# Scala-part1



变量不能改变为它指定的类型，所以不能将一个变量重新赋值为类型不兼容的数据。

```scala
val x: <type> = xxx
```





* 类型转换：

`scala`不允许从较高等级类型自动转换到较低等级类型。

```scala
scala> val l: Long = 20
val l: Long = 20

scala> val i: Int = l
                    ^
       error: type mismatch;
        found   : Long
        required: Int
```

因为每一个数据类型都是一个类，可以有很多方法，这里用`toInt`转换；

```scala
scala> val i: Int = l.toInt
val i: Int = 20
```



### 字符串

```scala
scala> val signature = "With Regards, \n Your friend"
val signature: String =
With Regards,
 Your friend
```

```scala
scala> val greeting = "Hello, " + "World"
val greeting: String = Hello, World

scala> greeting
val res27: String = Hello, World

//与java不同，会检查真正的想等性；
scala> val mathched = (greeting == "Hello, World")
val mathched: Boolean = true

//支持数学运算符
scala> val theme = "Na " * 16  + "Batman!"
val theme: String = Na Na Na Na Na Na Na Na Na Na Na Na Na Na Na Na Batman!

//字符串段落
scala> val greeting = """ I
     | am
     | zhengchu. """
val greeting: String =
" I
am
zhengchu. "
```





```scala
scala> val approx = 355/113f
val approx: Float = 3.141593

scala> println("Pi approximate is : " + approx  + " . ")
Pi approximate is : 3.141593 .
```



#### 字符串內插

scala的字符串內插记法是在字符串的第一个双引号前面加一个“s”前缀，然后用💲指示外部数据的引用。

```scala
scala> println(s"Pi appromimate is : $approx.")
Pi appromimate is : 3.141593.
```



* `${xx}`: xx是非字字符如算式，或当引用的xx与周围文本无法区分时使用。

```scala
scala> s"How doo you like them ${item}s?"
val res34: String = How doo you like them apples?

scala> s"How doo you like them ${"Pepper " * 3}salt?"
val res35: String = How doo you like them Pepper Pepper Pepper salt?
```



* 用`r`操作符把字符串转换为正则表达式类型；
* 

```scala
//字符串
scala> val input = "Enjoying this apple 3.14159 times today"
val input: String = Enjoying this apple 3.14159 times today

//正则表达式
scala> val pattern = """.* apple ([\d.]+) times .*""".r
val pattern: scala.util.matching.Regex = .* apple ([\d.]+) times .*

//应用上面的正则表达式产生的值赋给amountText
scala> val pattern(amountText) = input
val amountText: String = 3.14159

scala> val amount = amountText.toDouble
val amount: Double = 3.14159
```









* `Unit`字面量是一对空的小括号()，`Unit`类型通常定义函数与表达式。
* 类似C中的void关键字。

```scala
scala> val nada = ()
val nada: Unit = ()
```



* 其他：

```scala
scala> 5.asInstanceOf[Long]
val res42: Long = 5

scala> (7.0 / 5).getClass
val res44: Class[Double] = double

scala> "A".hashCode
val res46: Int = 65

scala> 20.toByte
val res49: Byte = 20

scala> (4.0).toString
val res51: String = 4.0

```



#### 元组

```scala
scala> val info = (5, "Koben", true)
val info: (Int, String, Boolean) = (5,Koben,true)

scala> info._1
val res52: Int = 5

// 2.关系操作符->生成元组
scala> val name = "zheng"-> "chu"
val name: (String, String) = (zheng,chu)
```





#### 表达式

表达式可以返回数据而不修改现有数据，这就允许使用不可变数据，是函数式编程的关键概念。

* 表达式块：块中的最后一个表达式作为整个表达式块的返回值。

```scala
scala> val max = {val x = 20
     | val y = 10
     | if (x > y) x else y
     | }
val max: Int = 20
```



#### 匹配表达式

```scala
//expression match
scala> val max = x > y match{
     | case true => x
     | case false => y
     | }
val max: Int = 20


// 多个模式重用case块
scala> val day = "MON"
val day: String = MON

scala> val kind  = day match{
     | case "MON" | "TUE" | "WED" | "THU" | "FRI" =>
     | "weekday"
     | case "SAT" | "SUN" => "weekend"
     | }
val kind: String = weekday

```



* 通用模式匹配

1. 值绑定（变量绑定）：把匹配表达式的输入绑定到一个局部值。

```scala
scala> val status = ms  match{
     | case "OK" => 200
     | case other =>{ //值绑定
     | println(s"Couldn't parse $other")
     | -1
     | }
     | }
Couldn't parse NO
val status: Int = -1

```



2.下划线`_`匹配：

```scala
scala> val status = message match{
     | case "ok" => 200
     | case _ =>{ //都一样
     |  println(s"Could't parse $message")
     | -1
     |  }
     | }
Could't parse Unauthorized
val status: Int = -1

```



#### 循环

`yield`关键字：如果表达式制定了这个关键字，调用的所有表达式的**返回值将作为一个集合返回**。

```scala
scala> for (day<-1 until 5) print(s"Day : $day ")
Day : 1 Day : 2 Day : 3 Day : 4

scala> for(x <- 1 to 7) println(s"Day $x:")
Day 1:
Day 2:
Day 3:
Day 4:
Day 5:
Day 6:
Day 7:



//集合返回，可以调用
scala> for(x <- 1 to 7) yield {s"Day $x:"}
val res64: IndexedSeq[String] = Vector(Day 1:, Day 2:, Day 3:, Day 4:, Day 5:, Day 6:, Day 7:)

scala> for(day <- res64) print(day + ", ")
Day 1:, Day 2:, Day 3:, Day 4:, Day 5:, Day 6:, Day 7:,


// 迭代器哨兵

scala> val quote = "Faith,Hope,,,Charity"
val quote: String = Faith,Hope,,,Charity


scala> for{
     | t <- quote.split(",")
     | if t != null
     | if t.size > 0
     | }
     | println(t)
Faith
Hope
Charity

```





* 多重循环：

```scala
scala> for{ x <- 1 to 2
     |      y <- 1 to 3}
     | print(s" ($x, $y) ")
 (1, 1)  (1, 2)  (1, 3)  (2, 1)  (2, 2)  (2, 3)
```



* 值绑定+for循环

```scala
scala> val mi = for (i <- 0 to 8; pow = 1 << i) yield pow
val mi: IndexedSeq[Int] = Vector(1, 2, 4, 8, 16, 32, 64, 128, 256)
```



* `do/while`

```scala
scala> do println(s"Here I am, x = $x") while (x > 0)
Here I am, x = 0
```







