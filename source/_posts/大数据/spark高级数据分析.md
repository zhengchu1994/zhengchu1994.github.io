---
title: spark基础1
mathjax: true
date: 2020-07-13 00:00:10
tags: 大数据
categories: 大数据
visible:
---





# 一、大数据分析

Spark 继承了 MapReduce 的线性扩展性和容错性，同时对它做了一些重量级扩展。



差异：Spark 摒弃了 `MapReduce` 先 `map` 再 `reduce `这样的严格方式，Spark 引擎可以执行更通用的有向无环图(directed acyclic graph，DAG)算子。这就意味着，**在 MapReduce 中需要将中间结果写入分布式文件系统时，Spark 能将中间结果直接传到流水作业线的下一步**。



跨集群节点内存：Spark它的` Dataset` 和` DataFrame` 抽象使开发人员将流水处理线上的任何点物化在跨越集群节点的**内存中**。**这样后续步骤如果需要相同数据集就不必重新计算或从磁盘加载。**



反应式应用：Spark 非常适用于涉及大量迭代的算法，这些算法需要多次遍历相同的数据集。**Spark 也适用于反应式(reactive)应用，这些应用需要扫描大量内存数据并快速响应用户的查询。**



构建：**由于构建于 JVM 之上，它可以利用 Java 技术栈里的许多操作和调 试工具。**



继承：Spark 还紧密集成 Hadoop 生态系统里的许多工具。它能读写 MapReduce 支持的所有数据格式，可以与 Hadoop 上的常用数据格式，如 Apache Avro 和 Apache Parquet(当然也 包括古老的 CSV)，进行交互。它能读写 NoSQL 数据库，比如 Apache HBase 和 Apache Cassandra。它的流式处理组件 Spark Streaming 能连续从 Apache Flume 和 Apache Kafka 之类的系统读取数据。它的 SQL 库 SparkSQL 能和 Apache Hive Metastore 交互，而且通过 Hive on Spark，Spark 还能替代 MapReduce 作为 Hive 的底层执行引擎。它可以运行在Hadoop 集群调度和资源管理器YARN 之上，这样 Spark 可以和 MapReduce 及 Apache Impala 等其他处理引擎动态共享集群资源和管理策略。





### 弹性分布式数据集(Resilient Distributed Dataset，RDD)

存在两个问题：

* RDD 难以高效且稳定地执行任务。由于依赖 Java 和 Python 对象，RDD 对内存的使用效率较低，而且会导致 Spark 程序受长时间垃圾回收的影响。
* 第二，Spark 的 API 忽视了一个事实—— 数据往往能用一个结构化的关系形式来表示;当出现这种情况的时候，API 应该提供一些原语，使数据更加易于操作。



### Spark2.0

**Spark 2.0 用 Dataset 和 DataFrame 替换掉 RDD 来解决上述问题**.



* **Dataset**：Dataset 与 RDD 十分相似，不同之处在于 **Dataset** 可以将它们所代表的对象映射到**编码器**(encoder)，从而实现了 一种更为高效的内存表示方法。
* **DataFrame**：**DataFrame** 是 **Dataset** 的子类，专门用于存储关系型数据(也就是用行和固定列表示的数据)。DataFrame 还可以与 **Spark SQL** 互相操作，这意味着用户可以写一 个 **SQL** 查询来获取一个 DataFrame，然后选择一种 Spark 支持的语言对这个 DataFrame 进行编程操作。

* Spark 2.0 包含了 **Spark ML API**，它引入了一个框架，可以将多种机器学习算法 和特征转换步骤**管道化**。





# 二、用Scala和Spark进行数据分析



### Scala

优势： Spark 框架是用 **Scala** 语言编写的，在向数据科学家介绍 Spark 时，采用与底层框架相同的编程语言有很多好处：

​	• 性能开销小：为了能在基于 JVM 的语言(比如 Scala）上运行用 R 或 Python 编写的算法，我们必须 在不同环境中传递代码和数据，在转换过程中信息时有丢失。但 是，**如果数据分析算法用 Spark Scala API 编写，你会对程序正确运行更有信心。**

• 能用上最新的版本和最好的功能：**Spark 的机器学习、流处理和图分析库全都是用 Scala 写的**，而新功能对 Python 和 R 绑 定支持可能要慢得多。

• 有助于你了解Spark的原理：你知道如何在 Scala 中使用 Spark，即使你平时主要还是在 其他语言中使用 Spark，**你还是会更理解系统，因此会更好地“用 Spark 思考”**。



*  **使用 Spark 和 Scala 做数据分析则是一种完全不同的体验**，因为你可以选择用同样的语言完成所有事情。



### Spark编程模型



* Spark 编程始于数据集，而数据集往往存放在分布式持久化存储之上，比如 **HDFS**。

  数据处理相关步骤：

   (1) 在输入数据集上**定义一组转换**。
   (2) **调用 action**，可以将转换后的数据集保存到持久化存储上，或者把结果返回到驱动程序的本地内存。
   (3) **运行本地计算**，处理分布式计算的结果。本地计算有助于你确定下一步的转换和 action。



### 小试牛刀:Spark shell和SparkContext

**Spark-shell**: spark-shell 是 Scala 语言的一个 **REPL** 环 境，它同时针对 **Spark** 做了一些扩展。如果这是你第一次见到 REPL 这个术语，可以把它 看成一个类似 R 的控制台:可以在其中用 Scala 编程语言定义函数并操作数据。

如果手头有 Hadoop 集群，可以先在 HDFS 上为块数据创建一个目录，然后将数据集文件 复制到 HDFS 上:

```shell
$ hadoop fs -mkdir linkage
$ hadoop fs -put block_*.csv linkage
```



如果你有一个 **Hadoop** 集群，并且 **Hadoop** 版本支持 **YARN**，通过为 **Spark master** 设定 **yarn** 参数值，就可以在集群上启动 **Spark** 作业：

```shell
$ spark-shell --master yarn --deploy-mode client
```



* 配置：**Installing Hadoop on Mac**：https://medium.com/beeranddiapers/installing-hadoop-on-mac-a9a3649dbc4d
* **spark-shell 设置资源为yarn**：https://blog.csdn.net/weixin_42660202/article/details/88040644



本地模式：

```shell
$ spark-shell --master local[*]
```



`sc`：**SparkContext** 对象的一个引用，含有很多方法，常用是创建RDD，RDD 是 Spark 所提供的最基本的抽象，代表分布在集群中多台机器上的对象集合。

* 用 SparkContext 基于外部数据源创建 RDD
* 在一个或多个已有 RDD 上执行转换操作来创建 RDD



* 运行依赖外部类库的代码：需要在 Spark 进程中通过 classpath 将所需类库的 JAR 文件包含进来。
* 为此，一种好的做法 是使用 **Maven** 来打包 **JAR**，使生成的 JAR 包含应用程序的所有依赖文件。接着在启动 shell 时通过 --jars 属性引用该 JAR。



#### scala

**只要在 Scala 中定义新变量，就必须在变量名称前加上 val 或 var**。

名称前带 **val** 的变量是不可变变量。一旦给不可变变量赋完初值，就不能改变它，让它指向另一个值。

而以 **var** 开头的变量则可以改变其指向，让它指向同一类型的不同对象





```scala
scala> val rawbloks = sc.textFile("linkage")
rawbloks: org.apache.spark.rdd.RDD[String] = linkage MapPartitionsRDD[1] at textFile at <console>:24

scala> rawbloks.first
res0: String = "id_1","id_2","cmp_fname_c1","cmp_fname_c2","cmp_lname_c1","cmp_lname_c2","cmp_sex","cmp_bd","cmp_bm","cmp_by","cmp_plz","is_match"
```



* `parallelize` 方法:	**第一个参数代表待并行化的对象集合，第二个参数代表分区的个数**。

* `collect`方法：如果知道 RDD 只包含少量记录，可以用 collect 方法向客户返回一个包含所有 RDD 内容 的数组；
* `take` 方法，这个方法在 `first` 和 `collect` 之间做了一些折衷，可以向客户端返回 一个包含指定数量记录的数组.



```scala
scala> val head = rawbloks.take(10)
head: Array[String] = Array("id_1","id_2","cmp_fname_c1","cmp_fname_c2","cmp_lname_c1","cmp_lname_c2","cmp_sex","cmp_bd","cmp_bm","cmp_by","cmp_plz","is_match", 607,53170,1,?,1,?,1,1,1,1,1,TRUE, 88569,88592,1,?,1,?,1,1,1,1,1,TRUE, 21282,26255,1,?,1,?,1,1,1,1,1,TRUE, 20995,42541,1,?,1,?,1,1,1,1,1,TRUE, 27989,34739,1,?,1,?,1,1,1,1,1,TRUE, 32442,69159,1,?,1,?,1,1,1,1,1,TRUE, 24738,29196,1,1,1,?,1,1,1,1,1,TRUE, 9904,89061,1,?,1,?,1,1,1,1,1,TRUE, 29926,36578,1,?,1,?,1,1,1,1,1,TRUE)

scala> head.length
res1: Int = 10

```



```scala
scala> val rdd = sc.parallelize(Array(1, 2, 2, 4), 4)
rdd: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[2] at parallelize at <console>:24

//count 动作返回 RDD 中对象的个数:
scala> rdd.count()
res6: Long = 4

```



* **创建 RDD 的操作并不会导致集群执行分布式计算**。
* 相反，RDD 只是定义了作为计算过程中间步骤的逻辑数据集。**只有调用 RDD 上的 action(动作)时分布式计算才会执行**。

```scala
//collect 动作返回一个包含 RDD 中所有对象的 Array(数组):
scala> rdd.collect()
res7: Array[Int] = Array(1, 2, 2, 4)
```





* **saveAsTextFile**方法： 动作将 RDD 的内容保存到持久化存 储(比如HDFS)上: `rdd.saveAsTextFile("xxx")`

  



*  **foreach** 方法：并结合 println 来打印 出数组中的每个值，并且每一行打印一个值

```scala
//把函数 println作为参数传递给另一个函数以执行某个动作
scala> head.foreach(println)
"id_1","id_2","cmp_fname_c1","cmp_fname_c2","cmp_lname_c1","cmp_lname_c2","cmp_sex","cmp_bd","cmp_bm","cmp_by","cmp_plz","is_match"
607,53170,1,?,1,?,1,1,1,1,1,TRUE
88569,88592,1,?,1,?,1,1,1,1,1,TRUE
21282,26255,1,?,1,?,1,1,1,1,1,TRUE
20995,42541,1,?,1,?,1,1,1,1,1,TRUE
27989,34739,1,?,1,?,1,1,1,1,1,TRUE
32442,69159,1,?,1,?,1,1,1,1,1,TRUE
24738,29196,1,1,1,?,1,1,1,1,1,TRUE
9904,89061,1,?,1,?,1,1,1,1,1,TRUE
29926,36578,1,?,1,?,1,1,1,1,1,TRUE

```





目标：过滤掉一个标题

* 和 Python 类似，Scala 声明函数用关键字 **def**。
* 和 Python 不同，我们**必须为函数指定参数类型**:在示例中，我们指明 line 参数是 String
* 可以紧跟在参数列表后面声明返回类型

```scala
scala> def isHead(line: String) = line.contains("id_1")
isHead: (line: String)Boolean
```

通过用 Scala 的 **Array 类的 filter 方法**打印出结果：

```scala
// 过滤出满足filter条件的元素并打印
scala> head.filter(isHead).foreach(println)
"id_1","id_2","cmp_fname_c1","cmp_fname_c2","cmp_lname_c1","cmp_lname_c2","cmp_sex","cmp_bd","cmp_bm","cmp_by","cmp_plz","is_match"

// 过滤出不满足filter的元素 1.
scala> head.filterNot(isHead).length
res3: Int = 9

//2.
scala> head.filter(x => !isHead(x)).length
res5: Int = 9


//3. Scala 允许使用下划线(_)表示匿名函数的参数：
scala> head.filter(!isHead(_)).length
res7: Int = 9
```





### 把代码从客户端发送到集群

* 在Spark 里把刚写好的代码应用到关联记录数据集 RDD rawblocks

```scala
scala> val noheader = rawbloks.filter(!isHead(_))
noheader: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[2] at filter at <console>:27
```





### 从RDD到DataFrame

DataFrame 是一个构建在 RDD 之上的 Spark 抽象，它专门为**结构规整的数据集而设计**，

DataFrame 的一条记录就是一行，每行都由若干个列组成，每一列的数据类型都有 严格定义。

可以把 DataFrame 类型实例理解为 Spark 版本的关系数据库表

要为记录关联数据集建立一个 DataFrame，我们需要用到 **SparkSession 对象 spark**

```scala
// spark是sparkSession的一个对象
scala> spark
res10: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@6098a311
```

```scala
//创建DataFrame
scala> val prev = spark.read.csv("linkage")
//CSV 文件中的每一列都是 string 类型，列名默认为 _c0、_c1、_c2，等等
prev: org.apache.spark.sql.DataFrame = [_c0: string, _c1: string ... 10 more fields]
//查看前面几行
prev.show()
```



```scala
//处理dummy数据
scala> val parse = spark.read.option("header", "true").
     | option("nullValue", "?").
     | option("inferSchema", "true").
     | csv("linkage")
parse: org.apache.spark.sql.DataFrame = [id_1: string, id_2: string ... 10 more fields]

//输出经过解析的 DataFrame 的模式信息
scala> parse.printSchema()
root
 |-- id_1: string (nullable = true)
 |-- id_2: string (nullable = true)
 |-- cmp_fname_c1: string (nullable = true)
 |-- cmp_fname_c2: string (nullable = true)
 |-- cmp_lname_c1: string (nullable = true)
 |-- cmp_lname_c2: string (nullable = true)
 |-- cmp_sex: string (nullable = true)
 |-- cmp_bd: string (nullable = true)
 |-- cmp_bm: string (nullable = true)
 |-- cmp_by: integer (nullable = true)
 |-- cmp_plz: integer (nullable = true)
 |-- is_match: boolean (nullable = true)
```





为了完成模式推断，**Spark 需要遍历数据集两 次:第一次找出每列的数据类型，第二次才真正进行解析**。如果预先知道某个文件的模式，你可以创建一个 **org.apache.spark.sql.types.StructType** 实例，并使用模式函数将它 传给 Reader API。



其他格式文件读取：

**json**

支持 CSV 格式具有的模式推断功能。 

**parquet和orc**

两种二进制列式存储格式，这两种格式可以相互替代。

**jdbc**

通过 JDBC 数据连接标准连接到关系型数据库。 

**libsvm**

一种常用于表示特征稀疏并且带有标号信息的数据集的文本格式。

**text**

文件的每行作为字符串整体映射到 DataFrame 的一列。

```scala
val d1 = spark.read.format("json").load("file.json") val d2 = spark.read.json("file.json")
```



#### 文件保存

枚举类型 **SaveMode**：可以选择强制覆盖(**Overwrite**)、在文件末尾追加(**Append**)，或者文件已存在时跳过这次写入(**Ignore**):

```scala

d1.write.format("parquet").save("file.parquet") 
d1.write.parquet("file.parquet")

d2.write.mode(SaveMode.Ignore).parquet("file.parquet")
```





### 2.8 用DataFrame API来分析数据

每次处理数据集中的数据时， Spark 得重新打开文件，再重新解析每一行，然后才能执行所需的操作，例如显示前几行 或计算记录的总数

```scala
scala> parse.count()
res18: Long = 5749136
```

调用 **cache** 方法，告诉 RDD 或 DataFrame 在创建时将它缓存在内存中:

```scala
parse.cache()
```



虽然默认情况下 DataFrame 和 RDD 的内容是临时的，但是 Spark 提供了一种持久化底 层数据的机制:

```scala
cached.cache() 

cached.count() 

cached.take(10)
```



cache() 是 persist(StorageLevel.Memory) 的简写，它将所有 Row 对象存储为未序列化的 Java 对象

在对象需要频繁访问或低延访问时，适合使用`StorageLevel.MEMORY`，因为它可以避免序列化的开销.

Spark 也提供了 MEMORY_SER 的存储级别，用于在内存中分配大字节缓冲区，以存储记 录的序列化内容



**带来的问题**：StorageLevel. MEMORY 的问题是要占用更大的内存空间。另外，大量小对象会对 Java 的垃圾回收施加 压力，会导致程序停顿和常见的速度缓慢问题。



经验：一般来说，当数据可 能被多个操作依赖时，并且相对于集群可用的内存和磁盘空间而言，如果数据集较小， 而且重新生成的代价很高，那么数据就应该被缓存起来。

```scala
//DataFrame 封装的 RDD 由 org.apache.spark.sql.Row 的实例组成，
//包括通过索引位置(从 0 开始计数)获取每个记录中值的访问方法，
//以及允许通过名称查找给定类型的字段的 getAs[T] 方法。
scala> parse.rdd.map(_.getAs[Boolean]("is_match")).countByValue()


res20: scala.collection.Map[Boolean,Long] = Map(true -> 20931, false -> 5728205)
```



有两种方式引用 DataFrame 的列名:

* 作为字面量引用，例如 `groupBy ("is_match")`;
* 或者作为 Column 对象应用，例如 count 列上使用的特殊语法 "\<col\>"。
* 这 两种方法在大多数情况下都是合法的，但是在 count 列上调用 desc 方法时需要使用语法 "\<col\>".

```scala
scala> parse.groupBy("is_match").count().orderBy($"count".desc).show()
+--------+-------+
|is_match|  count|
+--------+-------+
|   false|5728201|
|    true|  20931|
|    null|      4|
+--------+-------+
```

* `agg`  from DataFrame, avg/stddev from spark.sql.functions
* 默认情况下 Spark 只计算样本标准差;要计算总体标准差，需要使用 `stddev_ pop `函数。

```scala
scala> parse.agg(avg($"cmp_sex"), stddev($"cmp_sex")).show()
+------------------+--------------------+
|      avg(cmp_sex)|stddev_samp(cmp_sex)|
+------------------+--------------------+
|0.9550012294607436|  0.2073014119031234|
+------------------+--------------------+
```







* DataFrame 都看作数据库中的一张表，并且可以使用熟悉而又强大的 SQL 语法来表达我们的问题。
* 首先，将 DataFrame 对象 parsed 所关联的表名告诉 Spark SQL 引擎，因为 parsed 这个变量名对于 Spark SQL 引擎是不可用的

```scala
//注册临时表
parsed.createOrReplaceTempView("linkage")

//用spark SQL进行查询（也可以用hiveQL）
spark.sql("""
SELECT is_match, COUNT(*) cnt FROM linkage
GROUP BY is_match
ORDER BY cnt DESC
""").show()
```



* 调用 SparkSession Builder API 的 enableHiveSupport 方法来使用 HiveQL 语法进行查询：	

```scala
val sparkSession = SparkSession.builder. master("local[4]")
.enableHiveSupport()
.getOrCreate()
```



* `describe()`函数总结

```scala
scala> parse.describe()
res30: org.apache.spark.sql.DataFrame = [summary: string, id_1: string ... 10 more fields]

scala> val summary = parse.describe()
summary: org.apache.spark.sql.DataFrame = [summary: string, id_1: string ... 10 more fields]

scala> summary.show()
```



为了让 `summary `的统计信息更便于阅读和比较，我们可以使用` select` 方法来选出一部分列：

```scala
summary.select("summary", "cmp_fname_c1", "cmp_fname_c2").show()
```



既可以使用 SQL 风格的 `where` 语法，也可以使用 DataFrame API 的 Column 对象来过滤 DataFrame

```scala
//where 函数是 filter 函数的一个别名
val matches = parsed.where("is_match = true") 
val matchSummary = matches.describe()

//需要对列 $"is_match" 使用 === 操作符，并且还需要用 lit 方法封装布尔文字 false，这样就可以将其转换成能与 is_ match 做对比的 Column 对象
val misses = parsed.filter($"is_match" === false) 
val missSummary = misses.describe()

```



#### 宽表\长表

宽表中行代表指标，列代表变量;

长表的每一行代 表一个指标、一个变量，以及指标和变量对应的值.



* flatMap: 将宽表转换成长表，可以利用 DataFrame 的 flatMap 方法，它是 RDD.flatMap 的一个封装。 flatMap 是 Spark 中最有用的转换函数之一:**它接受一个函数作为参数，该函数处理一条输入记录，并返回一个包含零条或多条输出记录的序列**。
* 你可以将 flatMap 看作我们使用过的 map 和 filter 转换函数的一般形式:map 是 flatMap 的一种特殊形式，即一条输入记录仅产 生一条输出记录;filter 是 flatMap 的另一种特殊形式，即输入和输出类型相同，并且基于 一个布尔函数决定返回零条或一条记录。



```scala
//取出第一个字符串作为指标row.getString(0)
// 对第二个到最后一个元素，每一个都被映射为一个tuple：(metric, schema(i).name, row.getString(i).toDouble)

scala> val summary = parse.describe()
scala> val longForm = summary.flatMap(row => {
     | val metric = row.getString(0)
     | (1 until row.size).map(i=>{
     | (metric, schema(i).name, row.getString(i).toDouble)
     | })
     | })
longForm: org.apache.spark.sql.Dataset[(String, String, Double)] = [_1: string, _2: string ... 1 more field]
```





* toDouble 方法是隐式转换的一个实例；

* 隐式转换：隐式转换的工作原理如下:当在 Scala 的对象上调用一个方法，并且 Scala 编译器没有在该 对象上的类定义中找到这个方法，那么编译器就会尝试将你的对象转换成拥有这个方法的 类的实例

  

longForm是` Dataset[T]` 接口，`DataFrame` 其实是 `Dataset[Row] `类型的别名.

总是可以将 `Dataset` 转换回` DataFrame`:

````scala
scala> val longDF = longForm.toDF("metric", "field", "value")
longDF: org.apache.spark.sql.DataFrame = [metric: string, field: string ... 1 more field]

scala> longDF.show()
+------+------------+-------------------+
|metric|       field|              value|
+------+------------+-------------------+
| count|        id_1|          5749136.0|
| count|        id_2|          5749133.0|
| count|cmp_fname_c1|          5748126.0|
| count|cmp_fname_c2|           103699.0|
| count|cmp_lname_c1|          5749133.0|
| count|cmp_lname_c2|             2465.0|
| count|     cmp_sex|          5749133.0|
| count|      cmp_bd|          5748338.0|
| count|      cmp_bm|          5748338.0|
| count|      cmp_by|          5748337.0|
| count|     cmp_plz|          5736289.0|
|  mean|        id_1|  33324.47979999771|
````





`Pivot.scala`文件，`:load Pivot.scala `加载到REPL中：



`pivot` 操作需要知道转置列的所有不同值， 对列 values 使用 agg(first) 操作，我们就可以指定宽表中每个单元格的值，因为每个 field 和 metric 的组合都只有一个值



```scala
import org.apache.spark.sql.DataFrame 
import org.apache.spark.sql.functions.first

def pivotSummary(desc: DataFrame): DataFrame = {
    val schema = desc.schema
    import desc.sparkSession.implicits._

    val lf = desc.flatMap(row =>{
        val metric = row.getString(0)
        (1 until row.size).map(i =>{
            (metric, schema(i).name, row.getString(i).toDouble)
        })
    }).toDF("metric", "field", "value")
    lf.groupBy("field").pivot("metric", Seq("count", "mean", "stddev",
    "min", "max")).agg(first("value"))
}
```



```scala
scala> matchSummaryT.show()
+------------+-------+------------------+--------------------+-------+-------+
|       field|  count|              mean|              stddev|    min|    max|
+------------+-------+------------------+--------------------+-------+-------+
|        id_2|20931.0| 51259.95939037791|   24345.73345377519|10010.0|99996.0|
|     cmp_plz|20902.0|0.9584250310975027| 0.19962063345931919|    0.0|    1.0|
|cmp_lname_c1|20931.0|0.9970152595958817|0.043118807533945126|    0.0|    1.0|
|cmp_lname_c2|  475.0| 0.969370167843852| 0.15345280740388917|    0.0|    1.0|
|     cmp_sex|20931.0| 0.987291577086618| 0.11201570591216435|    0.0|    1.0|
|      cmp_bm|20925.0|0.9979450418160095|0.045286127452170664|    0.0|    1.0|
|cmp_fname_c2| 1333.0|0.9898900320318176| 0.08251973727615237|    0.0|    1.0|
|cmp_fname_c1|20922.0|0.9973163859635038| 0.03650667584833679|    0.0|    1.0|
|        id_1|20931.0| 34575.72117911232|   21950.31285196913|10001.0|99946.0|
|      cmp_bd|20925.0|0.9970848267622461| 0.05391487659807981|    0.0|    1.0|
|      cmp_by|20925.0|0.9961290322580645| 0.06209804856731055|    0.0|    1.0|
+------------+-------+------------------+--------------------+-------+-------+

```



