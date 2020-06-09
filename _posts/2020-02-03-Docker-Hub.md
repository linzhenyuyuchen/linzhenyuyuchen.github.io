---
layout:     post
title:      Docker Hub
subtitle:   Docker Hub 的使用
date:       2020-02-03
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Docker
---

# Docker

## 注册账户

https://hub.docker.com/account/signup/

## 登陆账户

`sudo docker login`

## 获取并运行容器

```
docker pull docker.mirrors.ustc.edu.cn/ufoym/deepo

```

## 创建我们自己的镜像

更新镜像之前，我们需要使用镜像来创建一个容器。

```
docker run -t -i docker.mirrors.ustc.edu.cn/ufoym/deepo /bin/bash
```

注意：已创建容器ID a7263612ba1d,我们在后边还需要使用它。

## 保存容器为镜像

使用 docker commit 来提交容器副本

```
$ sudo docker commit a7263612ba1d lzy
```

## 使用我们的新镜像来创建一个容器

```
$ sudo docker run -t -i lzy /bin/bash
```

## 推送镜像到 Docker Hub

一旦你构建或创建了一个新的镜像，你可以使用 docker push 命令将镜像推送到 Docker Hub 。这样你就可以分享你的镜像了，镜像可以是公开的，或者你可以把镜像添加到你的私有仓库中。

```
$ docker push lzy
The push refers to a repository [ouruser/sinatra] (len: 1)
Sending image list
Pushing repository ouruser/sinatra (3 tags)
```

## 主机中移除镜像

```
$ docker rmi training/sinatra
```

