

get_json_object使用的是堆外内存，默认堆外内存只有max( executorMemory * 0.10），可以考虑通过

--conf "spark.yarn.executor.memoryOverhead=4G" 设置堆外内存。

https://blog.csdn.net/weixin_43267534/article/details/100978755





有用的summary：https://www.cnblogs.com/tomato0906/articles/7291178.html



a 1 2 3 

b 4 5 6







```scala
val df = sc.parallelize(Seq((1, "zhengchu", "tt"), (2, "zc", null), (3, null, null))).toDF("x", "y","z")

df.agg(
       (sum(when(col("x").isNull, 1).otherwise(0))).alias("num_x"),
       (sum(when(col("y").isNull, 1).otherwise(0))).alias("num_y"),
       (sum(when(col("z").isNull, 1).otherwise(0))).alias("num_z")
       )

scala> res5.show
+-----+-----+-----+
|num_x|num_y|num_z|
+-----+-----+-----+
|    0|    1|    2|
+-----+-----+-----+


val name = df.columns.toArray
val countNull = name.map(x=>{
       val res = sum(when(col(x).isNull, 1).otherwise(0)).alias("num_" + x)
       res
       })
scala> df.agg(countNull.head, countNull.tail:_*).show
+-----+-----+-----+
|num_x|num_y|num_z|
+-----+-----+-----+
|    0|    1|    2|
+-----+-----+-----+
```

