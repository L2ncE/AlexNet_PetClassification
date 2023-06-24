![image](AlexNet%20PetClassification.png)

English | [中文](README_zh.md)

AlexNet PetClassification is a simple image classifier based on AleNet and MindSpore to recognize cats and dogs. This
project is a course design for the "ModelArts Machine Learning Practice" course of Chongqing University of Posts and
Telecommunications in the spring semester of 2023.

## Quick start

### Install Dependencies

According to the system and hardware environment, install the related dependencies of Mind Spore.
For details, please refer to [Mind Spore Installation Guide](https://www.mindspore.cn/install).

### Download the Dataset

Download the open source cat and dog dataset on the Kaggle platform.
For details, please refer
to [Cat and Dog Dataset Download](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765).

### Preprocessing

Preprocessing can be performed in `pre_process.py` to process images that do not meet our requirements.
Among them we need to rename the dataset we downloaded to `cat_dog.zip`.

```shell
python pre_process.py
```

### Train

```shell
python train.py
```

After training, the accuracy of the back-end model after training will be generated.

## License

This project is under the Apache License 2.0. See the LICENSE file for the full license text.