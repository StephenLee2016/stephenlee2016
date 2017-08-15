---
layout:     post
title:      Spark Streaming + Kafka入门实战
subtitle:   （二）Spark Streaming 实现 WordCount
date:       2017-08-09
author:     Stephen
header-img: img/home-bg-o.jpg
tags:
    - Spark Streaming
    - Kafka
    - WordCount
---

# （二）Spark Streaming 实现 WordCount    

---


> 在上一篇文章中，主要是介绍了在Windows单机环境上如何搭建Kafka的环境，如何创建一个Topic，用producer生产数据，consumer去消费数据。这篇文章将会用Spark Streaming + Kafka实现一个实时统计词频的Demo


实践的环境：

- win7 64位
- spark 2.1.0
- kafka 2.10
- zookeeper 3.4.8
- Intellij 
- scala 2.11

---

Intellij下利用Maven创建名为sparktest的工程，在pom.xml中添加必要的依赖：

```xml
 <dependencies>
    <dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-core_2.11</artifactId>
      <version>2.1.0</version>
    </dependency>
    <dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-sql_2.11</artifactId>
      <version>2.1.0</version>
    </dependency>
    <dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-hive_2.11</artifactId>
      <version>2.1.0</version>
    </dependency>
    <dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-streaming_2.11</artifactId>
      <version>2.1.0</version>
    </dependency>
    <dependency>
      <groupId>org.apache.hadoop</groupId>
      <artifactId>hadoop-client</artifactId>
      <version>2.7.3</version>
    </dependency>
    <dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-streaming-kafka-0-8_2.11</artifactId>
      <version>2.1.0</version>
    </dependency>
    <dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-mllib_2.11</artifactId>
      <version>2.1.0</version>
    </dependency>
    <dependency>
      <groupId>org.scala-lang</groupId>
      <artifactId>scala-library</artifactId>
      <version>2.11.7</version>
    </dependency>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.4</version>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>org.specs</groupId>
      <artifactId>specs</artifactId>
      <version>1.2.5</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
```

然后在src\main\scala目录下新建scala object文件，具体代码如下：

```scala
/**
  * Created by jrlimingyang on 2017/8/8.
  */
import kafka.serializer.StringDecoder
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.streaming.dstream.InputDStream
import org.apache.spark.streaming.kafka.KafkaUtils
import org.apache.spark.streaming.{Duration, StreamingContext}

object SparkKafka {
  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("kafka-spark-demo").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")

    val scc = new StreamingContext(conf, Duration(5000))
    scc.sparkContext.setLogLevel("ERROR")
    scc.checkpoint("C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\checkpoint")   //因为用到了 updateStateByKey, 所以必须要设置checkpoint
    val topics = Set("test")  //我们需要消费的kafka数据的topic
    val brokers = "10.9.45.10:9092"
    val kafkaParam = Map[String, String](
      // "zookeeper.connect" -> "192.168.21.181:2181",
      // "group.id" -> "test-consumer-group",
      "metadata.broker.list" -> brokers, //kafka的broker list地址
      "serializer.class" -> "kafka.serializer.StringEncoder"
    )

    val stream: InputDStream[(String, String)] = createStream(scc, kafkaParam, topics)
    stream.map(_._2)  //取出value
      .flatMap(_.split(" "))  //将字符串使用空格分割
      .map(r => (r,1))   //每个单词映射成一个pair
      .updateStateByKey[Int](updateFunc)  //用当前batch的数据区更新已有的数据
      .print()   //打印前十个数据
    scc.start()  //真正启动程序
    scc.awaitTermination()   //阻塞等待
  }

  val updateFunc = (currentValues: Seq[Int], preValue: Option[Int]) => {
    val curr = currentValues.sum
    val pre =  preValue.getOrElse(0)
    Some(curr + pre)
  }

  /**
    * 创建一个从kafka获取数据的流
    * @param scc        spark streaming上下文
    * @param kafkaParam kafka相关配置
    * @param topics   需要消费的topic集合
    */
  def createStream(scc: StreamingContext, kafkaParam: Map[String, String], topics: Set[String]) = {
    KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](scc, kafkaParam, topics)
  }
}
```

代码中注释比较详细，主要提两个关键的地方吧：

> 1、 val topics = Set("test")  这里的"test"是上一篇中建立的topic  
> 2、 val brokers = "10.9.45.10:9092"  这里的ip替换为你自己的ip即可

直接运行程序，由于现在并没有接收到实时数据，所以程序的结果显示为空：

![spark-streaming-run](http://a.hiphotos.baidu.com/image/pic/item/b8389b504fc2d562adcbb109ed1190ef77c66c16.jpg)

这里我们手动输入几条数据，看一下spark streaming的计算结果：

![producer-input](http://e.hiphotos.baidu.com/image/pic/item/3ac79f3df8dcd100de3a98c5788b4710b8122fab.jpg)

这里输入了两句话，看一下程序的运行结果：

![spark-streaming-result](http://f.hiphotos.baidu.com/image/pic/item/cdbf6c81800a19d8f1b2e4e139fa828ba71e46a4.jpg)

结果可以看到，对于接收到的实时数据，spark streaming 会统计词的频率。