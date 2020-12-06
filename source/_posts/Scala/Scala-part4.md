---
title: Scala-part4
mathjax: true
date: 2020-06-20 13:43:00
tags: Scala
categories: Scala

---







## 类



#### 继承



`java.lang.Object`类是JVM中所有实例的根，包括Scala，等价于Scala根类型`Any`。

`AnyRef`是`Any`的子类，是所有可实例化的类型的根。`

```scala
#
scala> class A {
     |  def hi = "hello from A"
     |  override def toString = getClass.getName
     | }
class A

scala> getClass.getName
val res127: String =

scala> class B extends A
class B

scala> class C extends B {override def hi = "hi C ->" + super.hi}
class C

scala> val hiA = new A().hi
val hiA: String = hello from A

scala> val hiB = new B().hi
val hiB: String = hello from A

scala> val hiC = new C().hi
val hiC: String = hi C ->hello from A

```



#### 重写

```scala
scala> class User(val name: String){
     |  def greet: String = s"Hello  from $name"
     |  override def toString = s"User($name)"
     | }
class User
```



#### 抽象类	

* 抽象类自己不能实例化。

```scala
scala> abstract class Car {
     |  val year: Int
     |  val automatic: Boolean = true
     |  def color: String
     | }
class Car


scala> class Mini(val year: Int, var color: String) extends Car
class Mini
```





#### 匿名类

使得类定义**不必是稳定的或者可重用的**。

```scala
scala> abstract class Listener {def trigger : Unit}

scala> class listening{
     |  var listener: Listener = null
     |  def register(l: Listener): Unit = {listener = l}
     |  def sendNotification(): Unit = {listener.trigger}
     | }


scala> notification.register(new Listener {
     |  def trigger: Unit = {println(s"Trigger at ${new java.util.Date}")}
     | })

scala> notification.sendNotification
Trigger at Sun Jun 14 21:06:59 CST 2020
```





  #### 重载方法

```scala
scala> class Printer(msg: String){
     |  def print(s: String): Unit = println(s"$msg: $s")
     |  def print(l: Seq[String]): Unit = print(l.mkString(", "))
     | }
class Printer

scala> new Printer("Today's Report").print("Foggy" :: "Rainy" :: "Hot" :: Nil)
Today's Report: Foggy, Rainy, Hot
```



#### apply:对象的默认方法

```scala
scala> class Multiplier(factor: Int) {
     |  def apply(input: Int) = input * factor
     | }
class Multiplier

scala> val triple = new Multiplier(3)
val triple: Multiplier = Multiplier@661d75f

scala> triple.apply(10)
val res25: Int = 30

//默认调用了对象的apply方法
scala> triple(30)
val res27: Int = 90

```







#### 懒值

`lazy val`：值在第一次实例化这些值时才创建。 

```
scala> class RandomPoint{
     |  val x = { println("creating x"); util.Random.nextInt}
     |  lazy val y = {println("now creating y"); util.Random.nextInt}
     | }
class RandomPoint

scala> val p = new RandomPoint()
creating x
val p: RandomPoint = RandomPoint@26338386

scala> println(s"Location is ${p.x}, ${p.y}")
now creating y
Location is -61352728, -1241964372

```



#### 导入方式

```scala
# 1.
scala> import collection.mutable._
import collection.mutable._

# 2.
scala> import collection.mutable.{Map=>MutMap}
import collection.mutable.{Map=>MutMap}

# 3.
scala> import collection.mutable.{Queue, ArrayBuffer}
import collection.mutable.{Queue, ArrayBuffer}

scala> val b = new ArrayBuffer[String]
val b: scala.collection.mutable.ArrayBuffer[String] = ArrayBuffer()

scala> b += "Hello"
val res6: b.type = ArrayBuffer(Hello)

scala> b += "zhengchu"
val res7: b.type = ArrayBuffer(Hello, zhengchu)

```





#### protected:同一类或子类可以访问

#### private：只有定义改方法的类才可以访问

```scala
scala> class User(private var password: String){
     |  def update(p: String): Unit = {
     |  println("Modifying the password!")
     |  password = p
     | }
     | def validate(p: String) = p == password
     | }
class User

scala> val u = new User
def <init>(password: String): User
scala> val u = new User("1234")
val u: User = User@37b45722

scala> u.validate("2345")
val res19: Boolean = false

scala> u.validate("1234")
val res20: Boolean = true

scala> u.update("1345")
Modifying the password!
```





