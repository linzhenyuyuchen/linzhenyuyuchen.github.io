---
layout:     post
title:      Docker Enable GPUs
subtitle:   Error response from daemon Unknown runtime specified nvidia.
date:       2020-01-05
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - Docker
---

# docker: Error response from daemon: Unknown runtime specified nvidia.


Why do I get the error Unknown runtime specified nvidia?

Make sure the runtime was registered to dockerd. You also need to reload the configuration of the Docker daemon.

Ref:`https://github.com/nvidia/nvidia-container-runtime#docker-engine-setup`


```
docker run --runtime=nvidia --rm lzy nvidia-smi
```
