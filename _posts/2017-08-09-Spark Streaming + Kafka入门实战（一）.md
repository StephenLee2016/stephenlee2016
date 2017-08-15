---
layout:     post
title:      Spark Streaming + Kafka入门实战
subtitle:   （一）Windows 环境 Kafka 安装演示
date:       2017-08-09
author:     Stephen
header-img: img/home-bg-o.jpg
tags:
    - Spark Streaming
    - Kafka
    - Windows
---

# （一）Windows 环境 Kafka 安装演示    

---


> 最近对流式计算很感兴趣，便参考网上有关Spark Streaming的文档实践了一番，发现对流式数据的处理并没有想象中的难，这里把实践的整个过程记录下来，以便后续复习。

说起流式计算，就不能不提Kafka，那么这货是个什么东西呢？去翻阅Kafka的说明文档，发现原来就是一个消息队列(消息中间件)，存储生产者产生的数据供消费者消费。

实践的环境：

- win7 64位
- spark 2.1.0
- kafka 2.10
- zookeeper 3.4.8

---

由于只是简单的实践，所以没有用到分布式环境，只是在单机上进行。如果想随便玩玩，知道流式计算的流程，单机完全ok。

## 1 、Windows下 zookeeper 的安装

我这里安装的是3.4.8版本的zookeeper，可以到[zookeeper 下载地址][1]下载对应的版本。

下载之后，将其解压到相应的文件夹下，我这里是解压到了D:盘下，修改名称为zk：

![zookeeper](http://g.hiphotos.baidu.com/image/pic/item/e850352ac65c1038ed846220b8119313b17e895e.jpg)

进入到zk目录下，新建data和log文件夹用来存放相应的文件：
![zk-data-log](http://g.hiphotos.baidu.com/image/pic/item/b3b7d0a20cf431ad0571ecdb4136acaf2fdd98b5.jpg)

接下来，就是最最关键的一步操作了，进入到zookeeper目录下的conf文件夹，复制zoo_sample文件并命名为zoo，将data和log文件的路径添加到zoo文件中：

>dataDir=D:\\zk\\data
>dataLogDir=D:\\zk\\log

![zk-setting](http://b.hiphotos.baidu.com/image/pic/item/024f78f0f736afc328ae9787b919ebc4b64512b5.jpg)

添加完保存文件，进入到D:\zk\zookeeper\bin目录下，打开命令行窗口，执行：

```shell
zkServer.cmd
```

![zkServer](http://e.hiphotos.baidu.com/image/pic/item/29381f30e924b899f59ffada64061d950b7bf6b5.jpg)

执行成功会出现上面的画面，很简单对吧~~

---

## 2 、 Windows下 Kafka 的安装

我这里安装的0.10版本的kafka，对应的scala是2.10版本，[Kafka 下载地址][2]。

下载到本地，解压到D:目录下，改名为kafka。让我们来看一下，kafka目录下都有些什么：

![kafka](http://g.hiphotos.baidu.com/image/pic/item/72f082025aafa40f65b8423fa164034f79f019d1.jpg)

其中logs是我自己新建的文件，用来存放log文件，你的可能没有这个文件。

在config路径下，修改zookeeper.properties文件，添加dataDir的路径：

> dataDir=D:\\zk\\data

![kafka-zk](http://e.hiphotos.baidu.com/image/pic/item/9f510fb30f2442a7fd137247db43ad4bd1130236.jpg)

在config路径下，修改server.properties文件，添加log.dirs的路径：

> log.dirs=D:\\kafka\\logs

![kafka-server](http://d.hiphotos.baidu.com/image/pic/item/4034970a304e251f66f36b1aad86c9177e3e53c8.jpg)

到此为止，kafka在Win 7单机环境上的安装就完成了，是不是超级简单？

进入到D:\kafka\bin\windows路径下，执行命令行(zookeeper确保是运行状态！！)：

> kafka-server-start.bat D:\\kafka\\config\\server.properties

运行成功会有如下显示：

![kafka-run](http://g.hiphotos.baidu.com/image/pic/item/5882b2b7d0a20cf4ff2a523c7c094b36adaf999f.jpg)

至此，已经在win7上安装了单机的kafka，可以继续后面的实践了。

---

## 3 、 Kafka命令行测试

接下来，会演示producer和consumer是如何进行工作的。首先，producer会先创建一个topic（不要关闭之前kafka和zookeeper的cmd窗口）：

在D:\kafka\bin\windows路径下，重新打开命令行界面，输入以下命令：

- 创建一个名为test的topic

> kafka-topics.bat --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test

- 查看topic

> kafka-topics.bat --list --zookeeper localhost:2181

![topic](http://f.hiphotos.baidu.com/image/pic/item/4b90f603738da9776c361bc8ba51f8198618e33c.jpg)

可以看到图上第一条语句执行后，显示test这个topic创建成功。第二条语句执行后，可以看到目前存在的topic。

创建一个producer然后输入一些数据：

> kafka-console-producer.bat --broker-list localhost:9092 --topic test

再创建一个consumer用来接收实时输入的数据：

> kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic test --from-beginning

![producer-consumer](http://h.hiphotos.baidu.com/image/pic/item/78310a55b319ebc4804df87a8826cffc1e17161f.jpg)

大功告成，consumer已经从producer接收到了实时的两条数据！

更详细的命令行工具可以参考[Apache Kafka 说明文档][3]


[1]: http://mirrors.tuna.tsinghua.edu.cn/apache/zookeeper/
[2]: https://www.apache.org/dyn/closer.cgi?path=/kafka/0.10.2.1/kafka_2.10-0.10.2.1.tgz
[3]: https://kafka.apache.org/quickstart