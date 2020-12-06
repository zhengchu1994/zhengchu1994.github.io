

#### Structured APIs：

**DataFrames**：相当于分布式的excel表格，

Partitions：To allow every executor to perform work in parallel, Spark breaks up the data into chunks called ***partitions*.**

* An important thing to note is that **with DataFrames you do not (for the most part) manipulate partitions manually** or individually.

**Transformations**：

There are two types of transformations: those that specify ***narrow dependencies*,** and those that specify ***wide dependencies***.（前者是一对一的变换，后者是1对多的变换）

**With narrow transformations,** Spark will automatically perform an operation called *pipelining*, meaning that if we specify multiple filters on DataFrames, they’ll all be **performed in-memory**

When we perform a **shuffle**（wide tranformation的另一种称谓，此时**Spark will exchange partitions across the cluster**.）, Spark writes the results to disk.

```scala
val flightData2015 = spark.read.option("inferSchema", "true").option("header", "true").csv("/data/flight-data/csv/2015-summary.csv")

//Remember, sort does not modify the DataFrame.
flightData2015.sort("count").explain()
```

![image-20200719233949410](https://tva1.sinaimg.cn/large/007S8ZIlgy1ggwpjmoi05j314e0jeqao.jpg)

* `sort`是wide transformation的原因： **rows will need to be compared with one another.**

* 另一方面， 在物理分配上，DataFrame自动的把数据进行**partitions**。

* By default, when we perform a shuffle, Spark outputs 200 shuffle partitions。设置为5份划分：`spark.conf.set("spark.sql.shuffle.partitions", "5")`。

  

  

  

  **SQL**：There is no performance difference between writing SQL queries or writing DataFrame code；

  创建SQL视图：make any DataFrame into a table or view with one simple method call `flightData2015.createOrReplaceTempView("flight_data_2015")`。

  ```scala
  val sqlWay = spark.sql("""
  SELECT DEST_COUNTRY_NAME, count(1) FROM flight_data_2015
  GROUP BY DEST_COUNTRY_NAME
  """)
  
  //等价于
  dataFrameWay = flightData2015\ .groupBy("DEST_COUNTRY_NAME")\ .count()
  
  //通过explain查看二者的解析一样的
  sqlWay.explain() 
  dataFrameWay.explain()
  
  //1. sql
  spark.sql("SELECT max(count) from flight_data_2015").take(1) 
  //2. scala
  import org.apache.spark.sql.functions.max flightData2015.select(max("count")).take(1)
  ```





**Datasets:** 相当于用自定义的type-safe functions作用到数据，并放入DataFrame。

The Dataset API gives users the ability to **assign a Java/Scala class to the records within a DataFrame** and manipulate it as a collection of typed objects, similar to a Java ArrayList or Scala Seq

形式： Dataset<T> in Java and Dataset[T] in Scala.

* For example, a Dataset[Person] will be guaranteed to contain objects of

  class Person.

* As of Spark 2.0, the supported types **are classes following the JavaBean pattern in Java and case classes in Scala**.

```scala
// in Scala
case class Flight(DEST_COUNTRY_NAME: String, ORIGIN_COUNTRY_NAME: String,
count: BigInt) 
val flightsDF = spark.read.parquet("/data/flight-data/parquet/2010-summary.parquet/") 
val flights = flightsDF.as[Flight]


flights.filter(flight_row => flight_row.ORIGIN_COUNTRY_NAME != "Canada").map(flight_row => flight_row).take(5)
```











-------



# 中文版



#### 第一章

```scala
scala> flightData2015.createOrReplaceTempView("flight_data_2015")

scala> val sqlWay = spark.sql("""
     | select DEST_COUNTRY_NAME, count(1)
     | from flight_data_2015
     | group by DEST_COUNTRY_NAME
     | """)



scala> sqlWay.show
+--------------------+--------+
|   DEST_COUNTRY_NAME|count(1)|
+--------------------+--------+
|             Moldova|       1|
|             Bolivia|       1|
|             Algeria|       1|
|Turks and Caicos ...|       1|
|            Pakistan|       1|
|    Marshall Islands|       1|
|            Suriname|       1|
|              Panama|       1|
|         New Zealand|       1|
|             Liberia|       1|
|             Ireland|       1|
|              Zambia|       1|
|            Malaysia|       1|
|               Japan|       1|
|    French Polynesia|       1|
|           Singapore|       1|
|             Denmark|       1|
|               Spain|       1|
|             Bermuda|       1|
|            Kiribati|       1|
+--------------------+--------+
only showing top 20 rows

```





```scala
scala>  val maxSql = spark.sql("""
     |      | select DEST_COUNTRY_NAME, sum(count) as destination_total
     |      | from flight_data_2015
     |      | group by DEST_COUNTRY_NAME
     |      | order by sum(count) desc
     |      | limit 5
     | """)
maxSql: org.apache.spark.sql.DataFrame = [DEST_COUNTRY_NAME: string, destination_total: bigint]

scala> maxSql.show
+-----------------+-----------------+
|DEST_COUNTRY_NAME|destination_total|
+-----------------+-----------------+
|    United States|           411352|
|           Canada|             8399|
|           Mexico|             7140|
|   United Kingdom|             2025|
|            Japan|             1548|
+-----------------+-----------------+




scala> import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.functions.desc

scala> flightData2015.groupBy
groupBy   groupByKey

scala> flightData2015.groupBy("DEST_COUNTRY_NAME").
     | sum("count").withColumnRenamed("sum(count)", "destination_total").
     | sort(desc("destination_total")).limit(5).show()

```







#### 流处理

```scala
scala> val staticDataFrame = spark.read.format("csv").option("header", "true").option("inferSchema", "true").
     | load("../../retail-data/by-day/*.csv")


scala> staticDataFrame.selectExpr("CustomerId",
     | "(UnitPrice * Quantity) as total_cost",
     | "InvoiceDate").groupBy(
     | col("CustomerId"), window(col("InvoiceDate"), "1 day")).sum("total_cost").show(5)
+----------+--------------------+------------------+
|CustomerId|              window|   sum(total_cost)|
+----------+--------------------+------------------+
|   14075.0|[2011-12-05 08:00...|316.78000000000003|
|   18180.0|[2011-12-05 08:00...|            310.73|
|   15358.0|[2011-12-05 08:00...| 830.0600000000003|
|   15392.0|[2011-12-05 08:00...|304.40999999999997|
|   15290.0|[2011-12-05 08:00...|263.02000000000004|
+----------+--------------------+------------------+
only showing top 5 rows
             
 
 // "maxFilesPerTrigger"：每次读完一个文件后都会被触发            
 val streamingDataFrame =  spark.readStream.schema(staticSchema).option("maxFilesPerTrigger", 1).
     | format("csv").option("header", "true").load("../../retail-data/by-day/*.csv")

 scala> streamingDataFrame.isStreaming
res19: Boolean = true

scala> val purchaseByCustomerPerHour = streamingDataFrame.selectExpr("CustomerId",
     | "(UnitPrice * quantity) as total_cost", "InvoiceDate").groupBy(
     | $"CustomerId", window($"InvoiceDate", "1 day")).sum("total_cost")
             
scala> purchaseByCustomerPerHour.writeStream.format("memory").queryName("customer_purchases").outputMode("complete").start()  //存入内存；表名；保存表中全部记录
             
             
             
scala> spark.sql("""
     | select *
     | from customer_purchases
     | order by 'sum(total_cost)' desc
     | """).show(5)
             
+----------+--------------------+------------------+
|CustomerId|              window|   sum(total_cost)|
+----------+--------------------+------------------+
|   15290.0|[2011-02-22 08:00...|             -1.65|
|   15237.0|[2011-12-08 08:00...|              83.6|
|   14788.0|[2011-11-13 08:00...|            368.45|
|   12464.0|[2011-07-13 08:00...|45.599999999999994|
|   17416.0|[2011-05-20 08:00...|1005.7000000000003|
+----------+--------------------+------------------+
only showing top 5 rows
             
 
  
             
 scala>  //打印到控制台 purchaseByCustomerPerHour.writeStream.format("console").queryName("customer_purchases_2").outputMode("complete").start()                                                                            
res25: org.apache.spark.sql.streaming.StreamingQuery = org.apache.spark.sql.execution.streaming.StreamingQueryWrapper@c857612

scala> -------------------------------------------
Batch: 0
-------------------------------------------
+----------+--------------------+------------------+
|CustomerId|              window|   sum(total_cost)|
+----------+--------------------+------------------+
|   14239.0|[2011-03-03 08:00...|             -56.1|
|   17700.0|[2011-03-03 08:00...| 602.6099999999999|
|   15932.0|[2011-03-03 08:00...|             -7.65|
|   16191.0|[2011-03-03 08:00...|             -1.65|
|   17646.0|[2011-03-03 08:00...|            345.85|
|   18041.0|[2011-03-03 08:00...|            148.49|
|   18102.0|[2011-03-03 08:00...|            1396.0|
|   13630.0|[2011-03-03 08:00...|             -14.4|
|   17652.0|[2011-03-03 08:00...|             222.3|
|   17567.0|[2011-03-03 08:00...|            535.38|
|   15596.0|[2011-03-03 08:00...|            303.03|
|   13476.0|[2011-03-03 08:00...| 727.5999999999999|
|   14524.0|[2011-03-03 08:00...|            210.05|
|   12500.0|[2011-03-03 08:00...|            249.84|
|   12524.0|[2011-03-03 08:00...|475.44000000000005|
|   15304.0|[2011-03-03 08:00...|105.44999999999999|
|   17389.0|[2011-03-03 08:00...|            124.68|
|   18218.0|[2011-03-03 08:00...|            309.38|
|   17856.0|[2011-03-03 08:00...|482.81000000000006|
|   15005.0|[2011-03-03 08:00...|            277.57|
+----------+--------------------+------------------+
only showing top 20 rows

-------------------------------------------
Batch: 1
-------------------------------------------
+----------+--------------------+------------------+
|CustomerId|              window|   sum(total_cost)|
+----------+--------------------+------------------+
|   18168.0|[2011-03-17 08:00...|218.83000000000007|
|   14239.0|[2011-03-03 08:00...|             -56.1|
|   17700.0|[2011-03-03 08:00...| 602.6099999999999|
|   15433.0|[2011-03-17 08:00...| 372.6399999999999|
|   15932.0|[2011-03-03 08:00...|             -7.65|
|   12514.0|[2011-03-17 08:00...|1017.6800000000002|
|      null|[2011-03-17 08:00...| 7876.000000000018|
|   13650.0|[2011-03-17 08:00...|372.11999999999995|
|   16839.0|[2011-03-17 08:00...|449.87000000000006|
|   12937.0|[2011-03-17 08:00...|217.42999999999998|
|   17725.0|[2011-03-17 08:00...|            337.71|
|   17652.0|[2011-03-03 08:00...|             222.3|
|   17567.0|[2011-03-03 08:00...|            535.38|
|   15596.0|[2011-03-03 08:00...|            303.03|
|   16422.0|[2011-03-17 08:00...|           -104.11|
|   17961.0|[2011-03-17 08:00...|               8.8|
|   13185.0|[2011-03-17 08:00...|              71.4|
|   16191.0|[2011-03-03 08:00...|             -1.65|
|   17716.0|[2011-03-17 08:00...| 665.4600000000002|
|   17646.0|[2011-03-03 08:00...|            345.85|
+----------+--------------------+------------------+
only showing top 20 rows

-------------------------------------------
            
```



* `window`：







#### 由Rows得到DataFrame

```scala
scala> import org.apache.spark.sql.types._
import org.apache.spark.sql.types._

scala> val myManualSchema = new StructType(Array(
     | new StructField("some", StringType, true),
     | new StructField("col", StringType, true),
     | new StructField("names", LongType, false)))

scala> val myRows = Seq(Row("Hello", null, 1L))
scala> val myRDD = spark.sparkContext.parallelize(myRows)

scala> val myDf = spark.createDataFrame(myRDD, myManualSchema)
myDf: org.apache.spark.sql.DataFrame = [some: string, col: string ... 1 more field]

scala> myDf.show()
+-----+----+-----+
| some| col|names|
+-----+----+-----+
|Hello|null|    1|
+-----+----+-----+	
```

```scala
//由于对null类型支持不稳定，不推荐在生产中使用；
scala> val myDF = Seq(("Hello", 2, 1L)).toDF("col1", "col2", "col3")
myDF: org.apache.spark.sql.DataFrame = [col1: string, col2: int ... 1 more field]

scala> myDF.show
+-----+----+----+
| col1|col2|col3|
+-----+----+----+
|Hello|   2|   1|
+-----+----+----+
```





#### 字面量lit

```scala
scala> df.select(expr("*"), lit(1).as("one")).show(2)
+-----------------+-------------------+-----+---+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|one|
+-----------------+-------------------+-----+---+
|    United States|            Romania|   15|  1|
|    United States|            Croatia|    1|  1|
+-----------------+-------------------+-----+---+
only showing top 2 rows

//另一种更规范的添加列的方式
scala> df.withColumn("numberOne", lit(1)).show(2)
+-----------------+-------------------+-----+---------+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|numberOne|
+-----------------+-------------------+-----+---------+
|    United States|            Romania|   15|        1|
|    United States|            Croatia|    1|        1|
+-----------------+-------------------+-----+---------+
only showing top 2 rows



scala> df.withColumn("withCountry", expr("ORIGIN_COUNTRY_NAME == DEST_COUNTRY_NAME")).show(2)
+-----------------+-------------------+-----+-----------+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|withCountry|
+-----------------+-------------------+-----+-----------+
|    United States|            Romania|   15|      false|
|    United States|            Croatia|    1|      false|
+-----------------+-------------------+-----+-----------+
only showing top 2 rows


//重命名某一列
scala> df.withColumn("Rename_Destination", expr("DEST_COUNTRY_NAME")).columns
res49: Array[String] = Array(DEST_COUNTRY_NAME, ORIGIN_COUNTRY_NAME, count, Rename_Destination)
//第二种重命名方式
scala> df.withColumnRenamed("DEST_COUNTRY_NAME","dest").columns
res50: Array[String] = Array(dest, ORIGIN_COUNTRY_NAME, count)

```





#### 删除列

```scala
scala> df.drop("ORIGIN_COUNTRY_NAME").columns
res53: Array[String] = Array(DEST_COUNTRY_NAME, count)
```



#### 类型改变

```scala
scala> df.withColumn("count2", col("count").cast("long")).show(2)
+-----------------+-------------------+-----+------+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|count2|
+-----------------+-------------------+-----+------+
|    United States|            Romania|   15|    15|
|    United States|            Croatia|    1|     1|
+-----------------+-------------------+-----+------+
only showing top 2 rows

```



#### 过滤行

```scala
scala> df.filter(col("count") < 2).show(2)
scala> df.where("count < 2").show(2)

+-----------------+-------------------+-----+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|
+-----------------+-------------------+-----+
|    United States|            Croatia|    1|
|    United States|          Singapore|    1|
+-----------------+-------------------+-----+
only showing top 2 rows

```



多重过滤：

```scala
scala> df.where(col("count") < 2).where(col("ORIGIN_COUNTRY_NAME") =!= "Croatia").show(2)
+-----------------+-------------------+-----+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|
+-----------------+-------------------+-----+
|    United States|          Singapore|    1|
|          Moldova|      United States|    1|
+-----------------+-------------------+-----+
only showing top 2 rows



scala> df.filter($"key2"===$"key3"-1).show
+----+----+----+
|key1|key2|key3|
+----+----+----+
| aaa|   1|   2|
| bbb|   3|   4|
+----+----+----+

其中, ===是在Column类中定义的函数，对应的不等于是=!=。
$”列名”这个是语法糖，返回Column对象


```



#### 去重

```scala
scala> df.select("ORIGIN_COUNTRY_NAME", "DEST_COUNTRY_NAME").distinct().count()
res71: Long = 256
```




#### 采样

```scala
scala> val seed = 5
seed: Int = 5

scala> val withReplacement = false
withReplacement: Boolean = false

scala> val fraction = 0.5
fraction: Double = 0.5


scala> df.sample(withReplacement, fraction, seed).count()
res75: Long = 126
```



#### 按比例分割数据

```scala
//分割比例的和不是1，比例参数自动被归一化
scala> val dataFrames = df.randomSplit(Array(0.25, 0.75), seed)


scala> dataFrames(0).count()
res79: Long = 60

scala> dataFrames(1).count()
res80: Long = 196

```





#### 合并union

```scala
scala> val schema = df.schema
scala> val newRows = Seq(
     | Row("New Country", "Other Country", 5L),
     | Row("New Country", "Other Country 3", 1L))

scala> val parallelizedRows = spark.sparkContext.parallelize(newRows)
scala> val newDF = spark.createDataFrame(parallelizedRows, schema)

// =!=不仅可以比较字符串，也可以比较表达式
scala> df.union(newDF).where("count = 1").where($"ORIGIN_COUNTRY_NAME" =!= "United States").show()
```



#### 行排序：sort与orderBy相等价，默认升序

```scala
scala> df.sort("count").show(5)
+--------------------+-------------------+-----+
|   DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|
+--------------------+-------------------+-----+
|               Malta|      United States|    1|
|Saint Vincent and...|      United States|    1|
|       United States|            Croatia|    1|
|       United States|          Gibraltar|    1|
|       United States|          Singapore|    1|
+--------------------+-------------------+-----+
only showing top 5 rows


scala> df.orderBy(col("count"), col("DEST_COUNTRY_NAME")).show(5)
scala> df.orderBy("count", "DEST_COUNTRY_NAME").show(5)


```

`asc_nulls_first`:指示空值安排在升序排列的前面；`desc_nulls_first`类似；

`asc_nulls_last`:指示空值安排在升序排列的后面；`desc_nulls_last`类似；

* **优化**：在进行别的转换之前，先对每个分区进行内部排序

  ```scala
  scala> spark.read.format("json").load("../../flight-data/json/*-summary.json").sortWithinPartitions("count")
  ```

  

#### 限制大小

```scala
scala> df.limit(5).show()
+-----------------+-------------------+-----+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|
+-----------------+-------------------+-----+
|    United States|            Romania|   15|
|    United States|            Croatia|    1|
|    United States|            Ireland|  344|
|            Egypt|      United States|   15|
|    United States|              India|   62|
+-----------------+-------------------+-----+
```



#### 重划分、合并



```scala
scala> df.rdd.getNumPartitions  
res91: Int = 1

scala> df.repartition
repartition   repartitionByRange

scala> df.repartition(5)
res92: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [DEST_COUNTRY_NAME: string, ORIGIN_COUNTRY_NAME: string ... 1 more field]

scala> df.repartition
repartition   repartitionByRange

//经常按照某一列执行过滤操作，则根据该列做分区
scala> df.repartition(col("DEST_COUNTRY_NAME"))

scala> df.repartition(5, col("DEST_COUNTRY_NAME"))

//合并分区不会导致数据的全面洗牌，但会尝试合并分区；
scala> df.repartition(5, col("DEST_COUNTRY_NAME")).coalesce(2)
```



#### 驱动器获取行

`collect`函数从整个`DataFrame`中获取所有数据；

`toLocalIterator`:以串行的方式一个一个分区地迭代整个数据集；

```scala
scala> collectDF.collect()

scala> collectDF.toLocalIterator()

```





####  布尔数据类型处理

`show()` 等价于 `show(20)`，`show(false)`等价于全部打印出来。

```scala
scala> val df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("../../retail-data/by-day/2010-12-01.csv")
scala> df.printSchema()
scala> df.createOrReplaceTempView("dfTable")



scala> df.where(col("InvoiceNo").equalTo(536365)).
     | select("InvoiceNo", "Description").show(5, false)
+---------+-----------------------------------+
|InvoiceNo|Description                        |
+---------+-----------------------------------+
|536365   |WHITE HANGING HEART T-LIGHT HOLDER |
|536365   |WHITE METAL LANTERN                |
|536365   |CREAM CUPID HEARTS COAT HANGER     |
|536365   |KNITTED UNION FLAG HOT WATER BOTTLE|
|536365   |RED WOOLLY HOTTIE WHITE HEART.     |
+---------+-----------------------------------+
only showing top 5 rows

//相等的操作
scala> df.where(col("InvoiceNo") === 536365).select("InvoiceNo", "Description").show(5, false)

//字符串形式的谓词表达式


```



scala中应该使用`===`(等于)或者`=!=`（不等于）；



最好用链式连接的方式形成顺序执行的过滤器：

`isin`:一个布尔表达式，如果该参数的求值包含该表达式的值，则该表达式为true。

```scala
scala> val priceFilter = col("UnitPrice") > 600
scala> val descripFilter = col("Description").contains("POSTAGE")


scala> df.where(col("stockCode").isin("DOT")).where(priceFilter.or(descripFilter)).show()
+---------+---------+--------------+--------+-------------------+---------+----------+--------------+
|InvoiceNo|StockCode|   Description|Quantity|        InvoiceDate|UnitPrice|CustomerID|       Country|
+---------+---------+--------------+--------+-------------------+---------+----------+--------------+
|   536544|      DOT|DOTCOM POSTAGE|       1|2010-12-01 14:32:00|   569.77|      null|United Kingdom|
|   536592|      DOT|DOTCOM POSTAGE|       1|2010-12-01 17:06:00|   607.49|      null|United Kingdom|
+---------+---------+--------------+--------+-------------------+---------+----------+--------------+

```



等价的过滤：

```scala
scala> df.withColumn("isExpensive", not(col("UnitPrice").leq(250))).filter("isExpensive").select("Description", "UnitPrice").show(5)
+--------------+---------+
|   Description|UnitPrice|
+--------------+---------+
|DOTCOM POSTAGE|   569.77|
|DOTCOM POSTAGE|   607.49|
+--------------+---------+



scala> df.withColumn("isExpensive", expr("NOT UnitPrice <= 250")).filter("isExpensive").select("Description", "UnitPrice").show(5)
+--------------+---------+
|   Description|UnitPrice|
+--------------+---------+
|DOTCOM POSTAGE|   569.77|
|DOTCOM POSTAGE|   607.49|
+--------------+---------+


//处理空值的情况
scala> df.where(col("Description").eqNullSafe("hello")).show()
+---------+---------+-----------+--------+-----------+---------+----------+-------+
|InvoiceNo|StockCode|Description|Quantity|InvoiceDate|UnitPrice|CustomerID|Country|
+---------+---------+-----------+--------+-----------+---------+----------+-------+
+---------+---------+-----------+--------+-----------+---------+----------+-------+
```





#### 数值类型处理







#### Drop删除空行



```scala
df.na.drop()
df.na.drop("any") //指定为any，当存在一个值为null，就删除该行；
df.na.drop("all") //当所有的值为null或者NaN时，才能删除行；
df.na.drop("all", Seq("StockCode", "InvoiceNo")) //指定某几列做删除空值操作
```



#### fill填充列中na

```scala
df.na.fill("All Null values become this string")


//填充为Integer或者double类型
df.na.fill(5, Seq("StockCode", "InvoiceNo"))

//多列填充
scala> val  filllColValues = Map("StockCode" -> 5, "Description" -> "No value")
scala> df.na.fill(filllColValues)
```



 

#### replace替换为类型相同的值

```scala
scala> df.na.replace("Description", Map("" -> "UNKOWN"))
```



#### 结构体

结构体视为`DataFrame`中的`DataFrame`：

```scala
scala> df.selectExpr("struct(Description, InvoiceNo) as complex", "*")
scala> complexDF = df.select(struct("Description", "InvoiceNo").alias("complex"))

scala> val complexDF = df.select(struct("Description", "InvoiceNo").alias("complex"))
scala> complexDF.createOrReplaceView("complexDF")

//使用.或者getField来访问内部的complex（作为DataFrame）
scala> complexDF.select("complex.Description")
scala> complexDF.select(col("complex").getField("Description")).show(2)
+--------------------+
| complex.Description|
+--------------------+
|WHITE HANGING HEA...|
| WHITE METAL LANTERN|
+--------------------+
only showing top 2 rows

//使用*查询结构体中的所有值
complex.select("complex.*")
```







#### 分布式共享变量

