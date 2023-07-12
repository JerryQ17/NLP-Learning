# 自然语言处理实践

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="./assets/PyTorch-ee4c2c.svg+xml"></a>[![Stars](./assets/{repo}.svg+xml)](https://github.com/JerryQ17/NLP-Learning/stargazers)

## 目录

- [目录](#目录)
- [实验环境](#实验环境)
- [实验准备](#实验准备)
  - [创建虚拟环境](#创建虚拟环境)
  - [激活虚拟环境](#激活虚拟环境)
  - [安装项目依赖](#安装项目依赖)
  - [安装PyTorch](#安装pytorch)
  - [退出虚拟环境](#退出虚拟环境)
- [实验过程](#实验过程)
- [实验结果](#实验结果)


---

## 实验环境

- 硬件
  - CPU: i7-12700H
  - RAM: 16GB DDR5
  - GPU: RTX 3060 Laptop 6GB
- 软件
  - Windows 11 x64
  - Python 3.10
  - Cuda 12.2

---

## 实验准备

因为之前没有使用过`conda`，所以我选择使用`venv`管理`python packages`。

你应当按照如下所述的方法配置本项目。

### 创建虚拟环境

在项目根目录中打开`powershell`，输入以下代码：

```powershell
python -m venv venv
```

创建了一个名为`venv`的虚拟环境。你会发现根目录中多了一个名为`venv`的文件夹。

### 激活虚拟环境

接着在刚刚的`powershell`中输入以下代码：

```powershell
.\venv\Scripts\activate.bat
```

激活了刚刚创建的`venv`环境，你会发现`powershell`左侧多出了`(venv)`的提示。

### 安装项目依赖

接着在刚刚的`powershell`中输入以下代码：

```powershell
pip install -r requirements.txt
```

这样就安装了项目的依赖包，除了`pytorch`。

### 安装PyTorch

接着在刚刚的`powershell`中输入以下代码：

```powershell
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

这样就安装了`pytorch preview for cuda 12.1`。

显然我不知道你需要哪个版本的`pytorch`，所以你应该前往[Pytorch官网](https://pytorch.org/)自行安装对应版本的`pytorch`。
记得`Package`选项要选`Pip`，安装命令要在`venv`环境中安装。

### 退出虚拟环境

这一步是可选的，如果你想立刻运行项目程序，则可以跳过这一步，如果你不想立刻运行项目程序，则可以直接退出虚拟环境。

接着在刚刚的`powershell`中输入以下代码：

```powershell
deactivate
```

这样就可以退出虚拟环境，并返回到本地环境中。

---

## 实验过程

本部分使用我封装过的代码来描述实验过程，具体的内部实现不在此赘述。

### TF-IDF

### Word to Vec

---

## 实验结果
