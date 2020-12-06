---
title: Hadoop学习记录
mathjax: true
date: 2020-07-13 00:00:10
tags: 大数据
categories: 大数据
visible:

---



### 前言



从 2003 年到 2006 年，Google 分别在 ODSI 与 SOSP 发表了 3 篇论文，引起了业界对于分布式系统的广泛讨论，这三篇论文分别是：

SOSP2003：The Google File System；

ODSI2004：MapReduce: Simplifed Data Processing on Large Clusters；

ODSI2006：Bigtable: A Distributed Storage System for Structured Data。



据此实现的Hadoop1.0：MapReduce+HDFS；



Hadoop 2.0 最大的改动就是引入了资源管理与调度系统 YARN，代替了原有的计算框架；YARN 将集群内的所有**计算资源抽象成一个资源池**，资源池的维度有两个：**CPU 和内存**。同样是基于 HDFS，我们可以认为 **YARN 管理计算资源，HDFS 管理存储资源**。上层的计算框架地位也大大降低，变成了 YARN 的一个用户。



Hadoop 可以理解为是一个计算机集群的操作系统，而 Spark、MapReduce 只是这个操作系统支持的编程语言而已，HDFS 是基于所有计算机文件系统之上的文件系统抽象。同理，YARN 是基于所有计算机资源管理与调度系统之上的资源管理与调度系统抽象，Hadoop 是基于所有计算机的操作系统之上的操作系统抽象。



#### YARN

目前的宏观调度机制一共有 3 种：集中式调度器（Monolithic Scheduler）、双层调度器（Two-Level Scheduler）和状态共享调度器（Shared-State Scheduler）

* 集中式调度器：只有一个中央调度器；

* 双层调度器：将整个调度工作划分为两层：**中央调度器**和**框架调度器**。中央调度器管理集群中所有资源的状态，它拥有集群所有的资源信息，按照一定策略（例如 FIFO、Fair、Capacity、Dominant Resource Fair）将资源粗粒度地分配给**框架调度器**，各个框架调度器收到资源后再根据应用申请细粒度将资源分配给容器执行具体的计算任务。在这种双层架构中，每个框架调度器看不到整个集群的资源，只能看到中央调度器给自己的资源

* 状态共享调度器：





## hadoop集群：hadoop + YARN

Hadoop后台通过一组守护进程实现存储和处理，在linux系统上，为什么每一个守护进程都在单独的JVM上运行？

hadoop集群中的节点分为：

* 主节点：协调集群工作服务；
* 工作节点：在主节点运行的进程之下进行；是存储数据与执行具体计算的节点。



hadoop服务：

* NameNode：运行在主节点上，负责管理与HDFS存储有关的元数据；
* Secondary NameNode 和 Standby NameNode：减轻NameNode的负担；
* DataNode：Linux文件系统之上存储HDFS数据块的工作节点，与NameNode保持联系。



YARN服务：

* RM（ResourceManager）：是主节点上工作的单一服务进程，负责额集群资源分配与运行任务的调度；
* AM（ApplicationManager）：集群中运行的每个应用都有该服务，与RM协商获取应用所需的资源；
* NM（NodeManager）：运行在**几乎**每一个工作节点上，在工作节点上运行、管理任务；与RM保持联系。



## Hadoop的分布式文件存储系统：HDFS

特点：

* 处理大数据集：PB级别；
* 容错性：默认情况下， 数据在Hadoop中被复制三次；
* 数据流式访问：批处理，数据流式访问；
* 简单数据一致性模型：一次写多次读取的访问模型，一旦将文件写入HDFS，就无法修改内容，也不能使用现有名称覆盖文件。不存在数据一致性问题。



1、HDFS数据在任何时候都不会通过**工作节点上的dataNode**传播，客户端始终访问驻留在DataNode上的文件系统（HDFS）；

2、写入一个HDFS文件：只有当Hadoop成功将所有数据块副本都放置在目标节点中时（3份），写操作才算成功。

3、HDFS自动复制任何未经复制的数据块；

NameNode将信息保存在磁盘上的fsimage文件中；更改信息一般先记录在编辑文件edits上，原因在于保留对命名空间的修改，如果NameNode崩溃，则存储在内存中的信息将丢失。

默认情况下，Secondary NameNode每小时合并一次edits和fsimage文件，并截断旧的edits文件；

但当NameNode不可用时，Secondary NameNode绝不担当NameNode的角色。

Secondary NameNode使用NameNode中的edits日志更新自己的fsimage，然后将更新的fsimage复制到NameNode上。

用两个NameNode，第二个称为StandBy NameNode，此时不需要Secondary NameNode了，使得集群保持高可用性，避免了集群中发生NameNode单点故障。



* ZooKeeper：为Hadoop的HDFS提供分布式同步和组服务，NameNode的高可用性依赖于ZooKeeper。



潜在不平衡数据问题：有的节点上数据多，新的节点数据少；

解决方案：Balancer数据平衡工具。



## Hadoop的操作系统：YARN

YARN：单个RM+多个节点上的NM组成的数据计算框架。

```
客户端------>RM<--------->NM
						^            ｜
					  ｜ 					｜
						|						 ⬇️
						AM<-------->Container
```

RM：纯调度器，只需将资源分配给发出请求的应用程序，无论应用程序或框架的类型如何。

Container：特定资源如RAM、CPU的抽象；

NM：协调它运行的DataNode得资源使用；

AM：是一个特定的框架，提供资源容错，负责以资源容器的形式向RM请求资源以支持程序应用。

​					



## Hadoop 指令



The File System (FS) shell includes various shell-like commands that directly interact with the Hadoop Distributed File System (HDFS) as well as other file systems that Hadoop supports, such as Local FS, WebHDFS, S3 FS, and others. The FS shell is invoked by:

```shell
#现在是   hdfs dfs 
bin/hadoop fs <args>
```



**put：**

Usage: `hadoop fs -put [-f] [-p] [-l] [-d] [ - | <localsrc1> .. ]. <dst>`

Copy single src, or multiple srcs from local file system to the destination file system. Also reads input from stdin and writes to destination file system if the source is set to “-”

Copying fails if the file already exists, unless the -f flag is given.

Options:

- `-p` : Preserves access and modification times, ownership and the permissions. (assuming the permissions can be propagated across filesystems)
- `-f` : Overwrites the destination if it already exists.
- `-l` : Allow DataNode to lazily persist the file to disk, Forces a replication factor of 1. This flag will result in reduced durability. Use with care.
- `-d` : Skip creation of temporary file with the suffix `._COPYING_`.

Examples:

- `hadoop fs -put localfile /user/hadoop/hadoopfile`
- `hadoop fs -put -f localfile1 localfile2 /user/hadoop/hadoopdir`
- `hadoop fs -put -d localfile hdfs://nn.example.com/hadoop/hadoopfile`
- `hadoop fs -put - hdfs://nn.example.com/hadoop/hadoopfile` Reads the input from stdin.

Exit Code:

Returns 0 on success and -1 on error.



Reference：https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/FileSystemShell.html



#### mkdir

```shell
hdfs dfs -mkdir -p filename
```



#### mac安装hadoop

* Run JPS to check the services

Access Hadoop web interface by connecting to

Resource Manager: [http://localhost:9870](http://localhost:9870/)

JobTracker: http://localhost:8088/

Node Specific Info: http://localhost:8042/



Reference：https://medium.com/beeranddiapers/installing-hadoop-on-mac-a9a3649dbc4d





#### Apache Maven 

Maven 是一个项目管理和构建自动化工具。

Maven 使用惯例优于配置的原则 。它要求在没有定制之前，所有的项目都有如下的结构：

| 目录                          | 目的                            |
| :---------------------------- | :------------------------------ |
| ${basedir}                    | 存放 pom.xml和所有的子目录      |
| ${basedir}/src/main/java      | 项目的 java源代码               |
| ${basedir}/src/main/resources | 项目的资源，比如说 property文件 |
| ${basedir}/src/test/java      | 项目的测试类，比如说 JUnit代码  |
| ${basedir}/src/test/resources | 测试使用的资源                  |

一个 `maven` 项目在默认情况下会产生 `JAR` 文件;

编译后 的 `classes` 会放在\${basedir}/target/classes 下面， `JAR` 文件会放在\${basedir}/target 下面。



Reference：https://www.oracle.com/cn/java/technologies/apache-maven-getting-started-1.html



## Spark

#### YARN与Spark如何合作

Spark应用程序充当客户端，将作业提交给YARN的RM；AM由处理框架的类库提供，spark提供了自己的AM。

#### 加载数据

```scala
//1.从HDFS加载：
val File = sc.textFile("xxx")
//2.一次性访问整个文件
val File = sc.wholeTextfiles("xxx")
//缓存该RDD
val File_cache = File.cache
//3.特定格式加载数据
val File = SparkContext.newAPIHadoopFile("xxxx")
//4. JdbcRDD可以将关系型数据库表作为RDD加载 === 工具：Apache Sqoop
import org.apache.spark.rdd.JdbcRDD
import java.sql.{Connection, DriverManager, ResultSet}
Class.forName("com.mysql.jdbc.Driver").newInstance
val Rdd = new JdbcRDD(sc, () => DriverManager.getConnection(url, username, password)),
.....)
//5.永久存储在HDFS上
myRDD.saveAsTextFile("xxxx")
```



#### Spark-shell引入第三方包

如何引入

```shell
spark-shell --jars path/nscala-time_2.10-2.12.0.jar
```

若有多个jar包需要导入，中间用逗号隔开即可。

* scala shell引入

```shell
scala -classpath ~/Downloads/json4s-native_2.11-3.2.11.jar
```



#### spark懒执行模式

：spark在action启动后开始计算；

当spark在工作节点运行其函数时，会将该函数中使用的变量复制到工作节点，但会限制两种类型的共享变量：

* 广播变量
* 累加器



#### RDD

RDD是一种抽象，表示在集群的节点上分区并且可以并行操作的元素的只读集合。由于RDD不可变，Spark不会更改原始RDD中的数据。

查看RDD谱系

```scala
scala> mydata.toDebugString
```



RDD操作：

* action：不会创建新的RDD，`take`;  `first`；`top`.
  * 转换：创建新的RDD，如`flatMap`;`sortBy`; `distinct`

RDD持久化：

RDD采用容错策略， 如果RDD的分区丢失，spark会重新计算它。





## BUGs

https://blog.petehouston.com/complete-apache-hadoop-troubleshooting/





