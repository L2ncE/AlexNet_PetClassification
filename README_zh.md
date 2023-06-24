![image](AlexNet%20PetClassification.png)

[English](README.md) | 中文

AlexNet PetClassification 是基于 AleNet 和 MindSpore 的一个简单的图像分类器，用于识别猫和狗。此项目为重庆邮电大学2023年春季学期《ModelArts 机器学习实践》课程的课程设计。

## 快速开始

### 安装依赖

根据系统和硬件环境，安装 MindSpore 的相关依赖。具体请参考 [MindSpore 安装指南](https://www.mindspore.cn/install)。

### 下载数据集

下载 Kaggle 平台开源的猫狗数据集。具体请参考 [猫狗数据集下载](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765)。

### 预处理

在 `pre_process.py` 中可以进行预处理，对不符合我们要求的图片进行处理。其中我们需要将我们下载的数据集改名为 `cat_dog.zip` 。

```shell
python pre_process.py
```

### 训练

```shell
python train.py
```

训练结束后会产生训练结束后端模型精确度。

## 许可证

项目在 Apache 许可证 2.0 版下开源。