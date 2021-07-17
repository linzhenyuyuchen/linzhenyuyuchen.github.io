---
layout:     post
title:      Neural Network Intelligence
subtitle:   搜参工具
date:       2021-06-16
author:     LZY
header-img: img/bg-20210718.jpg
catalog: true
tags:
    - Pytorch
    - 搜参
---



# NNI

[github](https://github.com/microsoft/nni/blob/master/README_zh_CN.md) | [doc](https://nni.readthedocs.io/zh/stable/) 

##### install

```shell
python3 -m pip install --upgrade nni
```

##### 代码修改

- 方法1： 使用json文件 [doc](https://nni.readthedocs.io/zh/stable/Tutorial/SearchSpaceSpec.html)

- 方法2：使用Annotation [doc](https://nni.readthedocs.io/zh/stable/Tutorial/AnnotationSpec.html) | [demo](https://nni.readthedocs.io/zh/stable/TrialExample/Trials.html)



##### run

```shell
cd /apdcephfs/private_zhenyulin/wsi/zhenyulin/wsi/trans
nnictl create --config nni_configs/nni_hebei.yml --port 8081
nnictl create --config nni_configs/nni_sicap.yml --port 8081
nnictl create --config nni_configs/nni_camelyon16.yml --port 8081
```



##### jizhi

```shell
jizhi_client start --scfg config_2class_vilt_NNI.json
```





[Command reference document ](https://nni.readthedocs.io/en/latest/Tutorial/Nnictl.html)


```shell

         commands                       description
    1. nnictl experiment show/status        show the information of experiments
    2. nnictl trial ls               list all of trial jobs
    3. nnictl top                    monitor the status of running experiments
    4. nnictl log stderr             show stderr log content
    5. nnictl log stdout             show stdout log content
    6. nnictl stop                   stop an experiment
    7. nnictl trial kill             kill a trial job by id
    8. nnictl --help                 get help information about nnictl

```

##### stop

```shell
nnictl stop 6P3RNCrY
```



##### load

```
nnictl experiment load [OPTIONS]
nnictl experiment load --path [path] --codeDir [codeDir]

```



| Name, shorthand      | Required | Default | Description                                                  |
| -------------------- | -------- | ------- | ------------------------------------------------------------ |
| –path, -p            | True     |         | the file path of nni package                                 |
| –codeDir, -c         | True     |         | the path of codeDir for loaded experiment, this path will also put the code in the loaded experiment package |
| –logDir, -l          | False    |         | the path of logDir for loaded experiment                     |
| –searchSpacePath, -s | True     |         | the path of search space file for loaded experiment, this path contains file name. Default in $codeDir/search_space.json |



##### weburl

```
nnictl webui url [Experiment ID]
```



##### open port

```shell
# use ngrok https://dashboard.ngrok.com/get-started/setup
cd /apdcephfs/private_zhenyulin/update/
wget -c https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-arm64.zip

# connect your account
cd /apdcephfs/private_zhenyulin/update/
./ngrok authtoken 1uYr0Hg3DquonpP7mVrfzwa6p0u_3pRKs8Cn4vhQgE6TfDUhk

# fire it up on romote server
cd /apdcephfs/private_zhenyulin/update/
./ngrok http 80
```

