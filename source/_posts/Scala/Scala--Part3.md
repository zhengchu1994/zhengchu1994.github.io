---
title: Scala-part3
mathjax: true
date: 2020-06-12 13:43:00
tags: Scala
categories: Scala
---











## 不可变集合



* `Nil`是`Nothing`类型实例，`Nothing`类型是所有其他`Scala`类型的一个不可实例化的子类型。

* `List`： 空的表等于`Nil`，表尾部后一个是`Nil`。

  * 头尾取不一样：

  ```scala
  scala> primes
  val res3: List[Int] = List(2, 3, 5, 7, 11, 13)
  
  scala> primes.head
  val res5: Int = 2
  
  scala> primes.tail
  val res6: List[Int] = List(3, 5, 7, 11, 13)
  
  ```

* `foreach`、`map`、`reduce`( 两个参数)

* `::`右操作符是`List`的放法：

  ```scala
  scala> val first = Nil.::(1)
  val first: List[Int] = List(1)
  //生成List
  scala> val first = 1:: Nil
  val first: List[Int] = List(1)
  
  //用::在List前面增加元素
  scala> val second = 2 :: first
  val second: List[Int] = List(2, 1)
  
  scala> second.tail == first
  val res39: Boolean = true
  ```

* 其他操作符：

  * `操作符记法`：`List drop 2`
  * 点记法：`List.flatten`
  * 差别：没有参数时，必须采用点记法。

  ```scala
  //空List前面加元素
  scala> val first = 1:: Nil
  val first: List[Int] = List(1)
  
  //List合并List
  scala> List(1, 2) ::: List(2, 3)
  val res40: List[Int] = List(1, 2, 2, 3)
  
  //List合并集合
  scala> List(1, 2)  ++ Set(3, 4, 3)
  val res41: List[Int] = List(1, 2, 3, 4)
  
  //List比较
  scala> List(1, 2) == List(1, 2)
  val res42: Boolean = true
  
  //取唯一元素
  scala> List(2, 3, 4, 5, 3).distinct
  val res43: List[Int] = List(2, 3, 4, 5)
  
  //丢带前k个元素
  scala> List('a', 'b', 'c', 'd') drop 2
  val res44: List[Char] = List(c, d)
  
  //谓词函数塞选
  scala> List(23, 8, 14, 21) filter (_ > 18)
  val res45: List[Int] = List(23, 21)
  
  //归并List
  scala> List(List(1, 2), List(3, 4)).flatten
  val res46: List[Int] = List(1, 2, 3, 4)
  
  //按条件分割List
  scala> List(1,2,3,4,5) partition ( _ < 3)
  val res48: (List[Int], List[Int]) = (List(1, 2),List(3, 4, 5))
  
  //切割[i，j）內的元素
  scala> List(1, 2, 3) slice (1, 3)
  val res49: List[Int] = List(2, 3)
  
  //谓词函数排序
  scala> List("apple", "to") sortBy (_.size)
  val res51: List[String] = List(to, apple)
  
  //字典顺序排序
  scala> List("apple", "to").sorted
  val res52: List[String] = List(apple, to)
  
  // 在位置做切割
  scala> List(2, 3, 5, 7) splitAt 2
  val res53: (List[Int], List[Int]) = (List(2, 3),List(5, 7))
  
  //取出前k个
  scala> List(2, 3, 5, 7, 11, 13) take 3
  val res54: List[Int] = List(2, 3, 5)
  
  //zip
  scala> List(1, 2) zip List ("a", "b")
  val res55: List[(Int, String)] = List((1,a), (2,b))
  
  // List尾部追加元素
  scala> val appended = List(1, 2, 3, 4) :+ 5
  val appended: List[Int] = List(1, 2, 3, 4, 5)
  
  // 取右边k个值
  scala> val suffix = appended takeRight 3
  val suffix: List[Int] = List(3, 4, 5)
  
  //丢弃右边k个值
  scala> val middle = suffix dropRight 2
  val middle: List[Int] = List(3)
  ```



* 映射函数：

```scala
scala> List(0, 1, 0) collect {case 1 => "ok"}
val res56: List[String] = List(ok)

//flatMap：使用一个给定函数转换各个元素，将结果列表“扁平化”到这个列表中
scala> List("milk,tea") flatMap (_.split(','))
val res57: List[String] = List(milk, tea)


scala> List("milk", "tea") map (_.toUpperCase)
val res59: List[String] = List(MILK, TEA)


scala> List(List("dog","pig"), List("tea", "coffee")) map (_.last)
val res9: List[String] = List(pig, coffee)

scala> List(List("dog","pig"), List("tea", "coffee")) flatMap (_.last)
val res10: List[Char] = List(p, i, g, c, o, f, f, e, e)
```



* 规约函数：

```scala
scala> List(32, 29) contains 29
val res64: Boolean = true

scala> List(0, 4, 3) endsWith List(4, 3)
val res65: Boolean = true

scala> List(23 , 4) exists (_ < 23)
val res75: Boolean = true

scala> List (2, 3, 4) forall ( _ < 10)
val res68: Boolean = true

scala> List(2, 4, 3) startsWith List(2)
val res72: Boolean = true
```



* 抽象规约函数：

```scala
scala> def reduceOp[A, B](l: List[A], start: B) (f: (B, A) => B): B = {
     | var a = start
     | for (i <- l) a = f(a, i)
     | a
     | }
def reduceOp[A, B](l: List[A], start: B)(f: (B, A) => B): B

scala> var included = reduceOp(List(46, 19, 92.0), 0.0)(_ + _)
var included: Double = 157.0

scala> var included = reduceOp(List(46, 19, 92), false){
     | (a, i) => if (a) a else (i == 19)
     | }
var included: Boolean = true

```



* 系统自带的该类函数：

  * 顺序地迭代与无序迭代区别开：提高分布式下的效率；
  * 左归约效率高于右归约

  ```scala
  //给定一个起始值和一个归约函数归约列表
  scala> List(4, 5, 6).fold(0)(_ + _)
  val res77: Int = 15
  
  scala> List(4, 5, 6).fold(1)(_ + _)
  val res78: Int = 16
  
  //从左到右归约
  scala> List(4, 5, 6).foldLeft(0)(_ + _)
  val res80: Int = 15
  
  //无序归约
  scala> List(4, 5, 6).reduce(_ + _)
  val res85: Int = 15
  
  //从左开始
  scala> List(4, 5, 6).reduceLeft(_ + _)
  val res89: Int = 15
  
  //返回各个累加值
  scala> List(4, 5, 6).scan(0)(_ + _)
  val res90: List[Int] = List(0, 4, 9, 15)
  
  
  //tag,j是自定义的变量名，false取过来初始化tag，j被List本身初始化
  scala> val included = List(46, 19, 92).foldLeft(false) {
       | (tag, j) => if(tag) tag else (j == 19)
       | }
  val included: Boolean = true
  
  //不给初始化值的时候，用List的第一个值初始化
  scala> List(24, 44, 33).reduceLeft(_ + _)
  val res93: Int = 101
  ```





#### 转换集合

```scala
scala> List(24, 44, 33).reduceLeft(_ + _)
val res93: Int = 101

scala> List(24, 99, 104).mkString(",")
val res94: String = 24,99,104

scala> List('f','t').toBuffer
val res95: scala.collection.mutable.Buffer[Char] = ArrayBuffer(f, t)

scala> Map("a"->1, "b"->2).toList
val res96: List[(String, Int)] = List((a,1), (b,2))

scala> Set(1->true,3->true).toMap
val res97: scala.collection.immutable.Map[Int,Boolean] = Map(1 -> true, 3 -> true)

scala> List(1,2,3,3).toSet
val res98: scala.collection.immutable.Set[Int] = Set(1, 2, 3)

scala> List(2, 4, 2).toString
val res99: String = List(2, 4, 2)

```



#### JAVA集合兼容性

```scala
import scala.jdk.CollectionConverters._
List(21, 32).asJava
```



#### 集合模式匹配

```scala
//匹配头节点
scala> val head = List('t', 'v') match {
     |  case x :: _ => x
     |  case Nil => ' '
     |  }
val head: Char = t

scala> val head = List() match {
     |  case x :: _ => x
     |  case Nil => ' '
     |  }
val head: Char =

```



```scala
scala> val msg = List(500, 404) match {
     |  case x if x contains(500)  => "has error"
     |  case _ => "okey"
     | }
val msg: String = has error
```



* 元组匹配：

  ```scala
  scala> val code = ('h', 204, true) match {
       |  case (_, _, false) => 501
       |  case ('c',_, true) => 302
       |  case ('h', x, true) => x
       |  case (c, x, true) => {
       |  println(s"Did not expect code $c")
       |  x
       |  }
       | }
  
  val code: Int = 204
  ```








## 可变集合



* `collection.immutable`会自动增加到scala的命名空间，所以用可变集合要写全了。

```scala
scala> val nums = collection.mutable.Buffer(1)
val nums: scala.collection.mutable.Buffer[Int] = ArrayBuffer(1)

scala> for (i <- 2 to 10) nums += i

scala> nums
val res1: scala.collection.mutable.Buffer[Int] = ArrayBuffer(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

//空集合需要声明类型
scala> val nums = collection.mutable.Buffer[Int]()
val nums: scala.collection.mutable.Buffer[Int] = ArrayBuffer()

scala> for(i <- 1 to 10) nums += i

scala> nums
val res3: scala.collection.mutable.Buffer[Int] = ArrayBuffer(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

//变为不可变List
scala> val l = nums.toList
val l: List[Int] = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

```





* `Builder`：`Buffer`的简化形式，只支持追加操作，生成集合类型。适合用于迭代的构建一个可变集合并转换为不可变集合的情况。



```scala
//创建Buffer
scala> val b = Set.newBuilder[Char]
val b: scala.collection.mutable.Builder[Char,scala.collection.immutable.Set[Char]] = scala.collection.immutable.SetBuilderImpl@66ee4640
//追加方式1
scala> b += 'h'
val res4: b.type = scala.collection.immutable.SetBuilderImpl@66ee4640
//追加方式2
scala> b ++= List('e', 'l', 'l', 'o')
val res5: b.type = scala.collection.immutable.SetBuilderImpl@66ee4640
//转为不可变集合
scala> val res = b.result
val res: scala.collection.immutable.Set[Char] = Set(h, e, l, o)
```





* `Array`：Java数组类型的包装器，除非JVM中使用。

  ```scala
  scala> val colors = Array("red", "green", "blue")
  val colors: Array[String] = Array(red, green, blue)
  
  ```





#### Seq

* `Seq`是所有序列的根类型，下面是`IndexedSeq`索引序列（`Vector`、`Range`），和`LinearSeq`线型链表序列(`List`,`stream`,`Queue/Stack`)



```scala
scala> val hi = "hello, " ++ "worldly" take 12 replaceAll("w", "W")
val hi: String = hello, World

```

#### LazyList懒集合

第一次访问元素时才会把这个元素增加到集合中；

LazyList生成的元素会缓存，以备以后获取，确保每个元素只生成一次。

<!--流可以用LazyList表示结束，对应于`List.Nil`。-->

```scala
//第1种生成
scala> def inc(i: Int): LazyList[Int]  = LazyList.cons(i, inc(i + 1))
def inc(i: Int): LazyList[Int]

scala> val s = inc(1)
val s: LazyList[Int] = LazyList(<not computed>)

scala> s.take(5).toList
val res158: List[Int] = List(1, 2, 3, 4, 5)
//第2种生成
scala> def inc(head: Int): LazyList[Int] = head #:: inc(head + 1)
def inc(head: Int): LazyList[Int]

scala> inc(10).take(10).toList
val res160: List[Int] = List(10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
```



有界的`LazyList`：

```python
scala> def to(head: Char, end: Char): LazyList[Char] = (head > end) match {
     |  case true => LazyList.empty
     |  case false => head #:: to((head + 1).toChar, end)
     | }
def to(head: Char, end: Char): LazyList[Char]

scala> val hexChars = to('A', 'F').take(20).toList
val hexChars: List[Char] = List(A, B, C, D, E, F)
```



#### 一元集合

* **option**集合：判断一个值存在`some`，或者不存在`null`。
* 任何操作都只能应用于`Some`，不会用到`None`.

```scala
scala> def divide(amt: Double, divisor: Double): Option[Double] = {
     |  if ( divisor == 0) None
     |  else Option(amt/ divisor)
     | }
def divide(amt: Double, divisor: Double): Option[Double]

scala> val illigal = divide(3, 0)
val illigal: Option[Double] = None

scala> val legal = divide(5, 2)
val legal: Option[Double] = Some(2.5)
```

* 判断空`List`：

```scala
scala> val odds = List(1, 3, 5)
val odds: List[Int] = List(1, 3, 5)

scala> val firstOdd = adds.headOption
val firstOdd: Option[Int] = Some(1)

scala> val evens = odds filter (_ % 2 == 0)
val evens: List[Int] = List()

scala> val firstEven = evens.headOption
val firstEven: Option[Int] = None
```



* 类型安全的过滤，不会导致`null`异常。

```scala
scala> val lowercase = Some("risible")
val lowercase: Some[String] = Some(risible)

scala> val filtered = lowercase filter (_ endsWith "ible") map (_.toUpperCase)
val filtered: Option[String] = Some(RISIBLE)

//filter作用于None上，返回None
scala> val exactSize = filtered filter (_.size > 15) map (_.size)
val exactSize: Option[Int] = None

```



```scala
scala> def nextOption = if (util.Random.nextInt > 0) Some(1) else None
def nextOption: Option[Int]

scala> nextOption match { case
     | Some(x) => x; case None => -1}
val res175: Int = -1

scala> nextOption match { case
     | Some(x) => x; case None => -1}
val res176: Int = 1
```



#### Try集合

```scala
scala> def loopAndFail(end: Int, failAt: Int): Int = {
     |  for (i <- 1 to end) {
     |  println(s"$i) ")
     |  if (i == failAt) throw new Exception("Too many iterations")
     | }
     | end
     | }
def loopAndFail(end: Int, failAt: Int): Int

scala> loopAndFail(10, 3)
1)
2)
3)
java.lang.Exception: Too many iterations
  at $anonfun$loopAndFail$1(<console>:4)
  at scala.collection.immutable.Range.foreach$mVc$sp(Range.scala:190)
  at loopAndFail(<console>:2)
  ... 32 elided

scala> val t1 = util.Try( loopAndFail(2, 3))
1)
2)
val t1: scala.util.Try[Int] = Success(2)

```





#### Future集合

scala代码在JVM上执行，同时也在java的线程中执行；

调用`future`并提供一个函数会在一个单独线程中执行该函数，而当前线程仍继续操作。

```scala
scala> import concurrent.ExecutionContext.Implicits.global

scala> val f = concurrent.Future {Thread.sleep(10000); println("hi")}
val f: scala.concurrent.Future[Unit] = Future(<not completed>)

scala> println("waiting")
waiting

scala> hi
```



同步（阻断当前线程）：`concurrent.Await.result`取一个线程和最大等待时间，等待时间内线程完成就返回结果，否则异常。

```scala
scala> import concurrent.duration._
import concurrent.duration._

scala> val maxTime = Duration(10, SECONDS)
val maxTime: scala.concurrent.duration.FiniteDuration = 10 seconds

scala> val amount = concurrent.Await.result(nextFtr(5), maxTime
val maxTime: scala.concurrent.duration.FiniteDuration
scala> val amount = concurrent.Await.result(nextFtr(5), maxTime)


val amount: Int = 6

```

