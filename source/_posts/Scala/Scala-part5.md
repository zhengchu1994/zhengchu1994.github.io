

## 定义一个案例类

一个最简单的案例类定义由关键字`case class`，类名，参数列表（可为空）组成：



```scala
scala> case class Book(isbn: String)
class Book

scala> val frankenstein = Book("2o84u23")
val frankenstein: Book = Book(2o84u23)
```



注意在实例化案例类`Book`时，并没有使用关键字`new`，这是因为案例类有一个默认的`apply`方法来负责对象的创建。



当你创建包含参数的案例类时，这些参数是公开（public）的`val`

```scala
case class Message(sender: String, recipient: String, body: String)
val message1 = Message("guillaume@quebec.ca", "jorge@catalonia.es", "Ça va ?")

println(message1.sender)  // prints guillaume@quebec.ca

message1.sender = "travis@washington.us"  // this line does not compile
```



你不能给`message1.sender`重新赋值，因为它是一个`val`（不可变）。在案例类中使用`var`也是可以的，但并不推荐这样。







---------------





资源：https://blog.softwaremill.com/how-to-improve-your-scala-programming-skills-a5b05e8b1629

