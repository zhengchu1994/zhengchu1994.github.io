---
title: Docker记录
mathjax: true
date: 2020-07-12 19:07:10
tags: 工具
categories: 工具
visible:
---







#### 概念

Docker架构：客户端----服务器模式



客户端 + 服务器（守护进程） + 注册处（可选）

注册处：存储Docker映像和映像的元数据。

服务器（守护进程）：可以在任意多个服务器中运行，作用是构建、运行、管理容器。

客户端：告诉服务器做什么



Docker 包括三个基本概念:

- **镜像（Image）**：一种底层定义，指明把什么放入容器中，是容器的文件系统。           

  可以使用远程仓库中别人制作好的镜像文件，也可以自己制作镜像文件。要制作镜像文件就要编写 **Dockerfile** 文件，其类似于 Makefile 文件。

  ```shell
  # 列出本机的所有image文件
  docker image ls
  docker images
  
  # 删除指定的image文件
  docker image rm <image-name>
  docker rmi <image-name>
  
  # 将指定的image文件从远程仓库拉取到本地
  docker image pull <image-name>[:tag]
  docker pull <image-name>[:tag]
  
  # 利用当前文件夹中的Dockerfile制作一个名为demo、tag为0.0.1的image文件
  # 若不指定tag，则默认的标签为latest
  docker image build -t demo:0.0.1 .
  docker build -t demo:0.0.1 .
  ```

  

  

- `容器（Container）`：容器可以被创建、启动、停止、删除、暂停等。

  镜像文件生成的容器（container）实例，本身也是一个文件，称为**容器文件**。当关闭容器时，并不会删除容器文件，只是容器停止运行而已。

  类似于在虚拟机中安装的操作系统，其本身会在硬盘中创建一系列文件，当关闭操作系统时，相应的文件并不会删除。

  ```shell
  # 从指定的image文件生成一个正在运行的容器实例，
  # 若本地没有指定的image文件，会从远程仓库中自动拉取下来并运行
  # 使用参数`-it`返回容器实例的终端
  # `--rm`
  # 使用参数`-p`将容器内端口映射到主机端口
  # 使用参数`-v`将主机目录和容器内目录进行绑定
  docker container run <image-name>[:tag]
  docker run <image-name>[:tag]
  
  # 列出本机正在运行的容器，使用参数`-all`列出所有容器文件
  docker container ls
  docker ps
  
  # 删除指定的容器文件
  docker container rm <container-id>
  docker rm <container-id>
  
  # 启动指定的容器实例
  docker container start <container-id>
  docker start <container-id>
  
  # 重启指定的容器实例
  docker container restart <container-id>
  docker restart <container-id>
  
  # 关闭指定的容器实例
  docker container stop <container-id>
  docker stop <container-id>
  
  # 强制关闭指定的容器实例
  docker container kill <container-id>
  docker kill <container-id>
  
  docker cp
  docker attach
  docker exec（重要）
  ```

  

- `仓库（Repository）`：一个代码控制中心，用来保存镜像。DockerHub 是一个由 Docker 公司运行和管理的基于云的存储库。它是一个在线存储库，Docker 镜像可以由其他用户发布和使用。有两种库：公共存储库和私有存储库。如果你是一家公司，你可以在你自己的组织内拥有一个私有存储库，而公共镜像可以被任何人使用。

  **仓库**（repository）是不同标签的镜像的集合，注册处（registry）又是不同仓库的集合，Docker 的官方注册处是 [Docker Hub](https://hub.docker.com/)，类似于 GitHub



#### 镜像加速

直接阿里云注册

https://cr.console.aliyun.com/cn-hangzhou/instances/credentials

#### 命令

2、拉取（pull）

这个很像git，在git的客户端一般是通过git pull来拉取代码，而这里是通过 docker pull来拉取镜像。

拉取公有仓库镜像

```shell
docker pull hello-world
```

拉取私有仓库镜像(用阿里云登陆建立)

```pseudocode
docker pull registry.cn-hangzhou.aliyuncs.com/镜像名字
```

3、查看已下载的镜像

通过`docker images`来查看本地下载好的镜像。

4、上传镜像

镜像在本地环境构建或是打包好之后，就可以上传到 Registry。Registry表示地址，官网默认不用输入。

```
docker push registry.cn-hangzhou.aliyuncs.com/用户名/镜像名:版本号
```

5、启动一个容器

运行helloworld项目 官网镜像：

```
docker run hello-world
```

![image-20200712191546199](https://tva1.sinaimg.cn/large/007S8ZIlly1ggoekpk4wvj31180mojv0.jpg)

说明，如果运行一个不存在的镜像，会自动从官网拉取。

加速器镜像。

```
docker run registry.cn-hangzhou.aliyuncs.com/镜像名字
```



关闭一个容器`Ctrl + d`：

`docker ps -a `：来看终止状态的容器

使用`docker stop $CONTAINER_ID`来终止一个运行中的容器。

```
(base) [~] docker ps -a                                             master  ✭ ✱
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                       PORTS               NAMES
e9dec8dc6afc        ubuntu              "bash"              14 minutes ago      Exited (127) 9 seconds ago                       hardcore_maxwell
ebc138eceed0        hello-world         "/hello"            35 minutes ago      Exited (0) 35 minutes ago                        nifty_austin
```





启动一个容器（docker start  容器ID）

`docker start ebc138eceed0 `



`docker ps -n 5 `查看容器的信息

```
(base) [~] docker start ebc138eceed0                                                                                        master  ✭ ✱
ebc138eceed0
(base) [~] docker ps -n 5                                                                                                   master  ✭ ✱
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                       PORTS               NAMES
e9dec8dc6afc        ubuntu              "bash"              20 minutes ago      Exited (127) 6 minutes ago                       hardcore_maxwell
ebc138eceed0        hello-world         "/hello"            41 minutes ago      Exited (0) 5 seconds ago
```



## 问题

* Error response from daemon: conflict: unable to remove repository reference "hello-world" (must force) - container ebc138eceed0 is using its referenced image bf756fb1ae65





2.停止所有的container，这样才能够删除其中的images：

```
docker stop $(docker ps -a -q)
```



如果想要删除所有container的话再加一个指令：

```
docker rm $(docker ps -a -q)
```



4.删除images，通过image的id来指定删除谁

```
docker rmi 
```



想要删除untagged images，也就是那些id为的image的话可以用

```
docker rmi  $(docker images | grep "^<none>" | awk "{print $3}")
```

要删除全部image的话

```
docker rmi $(docker images -q)
```





----------



> REF:
>
> https://www.cnblogs.com/liuhongfeng/p/12505743.html
>
> https://www.cnblogs.com/linjiqin/p/8608975.html
>
> https://howiezhao.github.io/2018/08/31/docker/