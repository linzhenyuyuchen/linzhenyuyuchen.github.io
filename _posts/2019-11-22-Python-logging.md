---
layout:     post
title:      Python logging
subtitle:   import logging
date:       2019-11-22
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - logging
    - Python
---


# logging

> 输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚

1. 可以通过设置不同的日志等级，在release版本中只输出重要信息，而不必显示大量的调试信息


2. print将所有信息都输出到标准输出中，严重影响开发者从标准输出中查看其它数据；logging则可以由开发者决定将信息输出到什么地方，以及怎么输出


## 基本设置

配置logging基本的设置，然后在控制台输出日志

logging.basicConfig函数各参数：

- filename：指定日志文件名；

- filemode：和file函数意义相同，指定日志文件的打开模式，'w'或者'a'；

- datefmt：指定时间格式，同time.strftime()；

- level：设置日志级别，默认为logging.WARNNING；

- stream：指定将日志的输出流，可以指定输出到sys.stderr，sys.stdout或者文件，默认输出到sys.stderr，当stream和filename同时指定时，stream被忽略

- format：指定输出的格式和内容，format可以输出很多有用的信息

```
%(levelno)s：打印日志级别的数值
%(levelname)s：打印日志级别的名称
%(pathname)s：打印当前执行程序的路径，其实就是sys.argv[0]
%(filename)s：打印当前执行程序名
%(funcName)s：打印日志的当前函数
%(lineno)d：打印日志的当前行号
%(asctime)s：打印日志的时间
%(thread)d：打印线程ID
%(threadName)s：打印线程名称
%(process)d：打印进程ID
%(message)s：打印日志信息
```

```python
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")
```

运行时，控制台输出，
```
2019-10-09 19:11:19,434 - __main__ - INFO - Start print log
2019-10-09 19:11:19,434 - __main__ - WARNING - Something maybe fail.
2019-10-09 19:11:19,434 - __main__ - INFO - Finish
```

logging中可以选择很多消息级别，如debug、info、warning、error以及critical

通过赋予logger或者handler不同的级别，开发者就可以只输出错误信息到特定的记录文件，或者在调试时只记录调试信息

## 将logger的级别改为DEBUG

```python
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

控制台输出了debug的信息:

```
2019-10-09 19:12:08,289 - __main__ - INFO - Start print log
2019-10-09 19:12:08,289 - __main__ - DEBUG - Do something
2019-10-09 19:12:08,289 - __main__ - WARNING - Something maybe fail.
2019-10-09 19:12:08,289 - __main__ - INFO - Finish
```

## 日志写入文件&&输出屏幕

> 设置logging，创建一个FileHandler，并对输出消息的格式进行设置，将其添加到logger，然后将日志写入到指定的文件中

```python
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)

# 写入到文件
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 输出到屏幕
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")
```

## 日志回滚

```python
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 定义一个RotatingFileHandler，最多备份3个日志文件，每个日志文件最大1K
rHandler = RotatingFileHandler("log.txt",maxBytes = 1*1024,backupCount = 3)
rHandler.setLevel(logging.INFO)
rHandler.setFormatter(formatter)
logger.addHandler(rHandler)

# 输出到屏幕
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)

logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")
```

## 设置消息的等级

可以设置不同的日志等级，用于控制日志的输出，

日志等级：使用范围

- FATAL：致命错误

- CRITICAL：特别糟糕的事情，如内存耗尽、磁盘空间为空，一般很少使用

- ERROR：发生错误时，如IO操作失败或者连接问题

- WARNING：发生很重要的事件，但是并不是错误时，如用户登录密码错误

- INFO：处理请求或者状态变化等日常事务

- DEBUG：调试过程中使用DEBUG等级，如算法中每个循环的中间状态




