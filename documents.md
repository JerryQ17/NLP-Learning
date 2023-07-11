# 项目文档

## 项目结构

- directory `dataset`
    - file `IMDB Dataset.csv`
- directory `src`
    - file `__init__.py`
    - file [`convert.py`](#convert.py)
    - file [`dataset.py`](#dataset.py)
    - file [`lstm.py`](#lstm.py)
    - file [`models.py`](#models.py)
    - file [`svm.py`](#svm.py)
    - file [`train.py`](#train.py)
- directory `svm`
    - directory `model`
    - directory `train`

---

## convert.py

[源代码](./src/convert.py)

### class Converter

`Converter`是`Dataset`和`Trainer`之间的桥梁，它提供了一些易于使用的API，可以将数据集中的原始数据转换为`SVM`或`Neutral Network`便于处理的数据形式，从而简化了数据处理的流程。

> Tips:[`tfidf_matrix`](#Converter-tfidf-matrix) 、[`feature_names`](#Converter-feature-names) 、[`tfidf_dataset`](#Converter-tfidf-dataset)属性无需显式调用`tfidf()`
> 方法，内部会自动调用。即使你改变了[`dataset`](#Converter-dataset)属性，这些属性也会自动更新。

|                        属性                         |            类型            |      初始值      |                             描述                             |
| :-------------------------------------------------: | :------------------------: | :--------------: | :----------------------------------------------------------: |
|       <a id="Converter-dataset">`dataset`</a>       | `torch.utils.data.Dataset` |    `Required`    |                        要转换的数据集                        |
|                     `processes`                     |           `int`            | `os.cpu_count()` |                     `to_svm`方法的进程数                     |
|  <a id="Converter-tfidf-matrix">`tfidf_matrix`</a>  | `scipy.sparse.csr_matrix`  |        /         |        [`dataset`](#Converter-dataset)的`tfidf`值矩阵        |
| <a id="Converter-feature-names">`feature_names`</a> |      `numpy.ndarray`       |        /         |          [`dataset`](#Converter-dataset)的特征单词           |
|                       `items`                       |           `list`           |        /         |         [`dataset`](#Converter-dataset)中的所有元素          |
|                  `items_generator`                  |        `Generator`         |        /         |     [`dataset`](#Converter-dataset)中的所有元素的生成器      |
| <a id="Converter-tfidf-dataset">`tfidf_dataset`</a> |   `dataset.TfIdfDataset`   |        /         | 包含了[`dataset`](#Converter-dataset)的`tfidf`和`label`的数据集，用于后续训练 |

#### method \_\_init__()

初始化一个`Converter`实例

##### 输入

|     参数      |             类型             |       初始值        |       描述       |
|:-----------:|:--------------------------:|:----------------:|:--------------:|
|  `dataset`  | `torch.utils.data.Dataset` |    `Required`    |    要转换的数据集     |
| `processes` |           `int`            | `os.cpu_count()` | `to_svm`方法的进程数 |

##### 输出

`None`

#### method tfidf()

计算数据集的TF-IDF表示，同时更新自身的[`tfidf_matrix`](#Converter-tfidf-matrix) 、[`feature_names`](#Converter-feature-names) 、[`tfidf_dataset`](#Converter-tfidf-dataset)属性，返回[`tfidf_dataset`](#Converter-tfidf-dataset)

##### 输入

`None`

##### 输出

一个`dataset.TfIdfDataset`实例，其实就是[`self.tfidf_dataset`](#Converter-tfidf-dataset)

#### method word2vec() **TODO**

##### 输入

##### 输出

#### method to_svm()

将[`dataset`](#Converter-dataset)中的原始数据转换成`tfidf`矩阵（如果没有转换过的话），并将`tfidf`矩阵存储为标准`libsvm`格式，返回文件保存的绝对路径

##### 输入

|    参数     | 类型  | 初始值 |                             描述                             |
| :---------: | :---: | :----: | :----------------------------------------------------------: |
| `save_path` | `str` | `None` | 保存路径，默认保存在`'..\svm\data`文件夹，实际保存文件名见返回值 |

##### 输出

文件保存的绝对路径

---

## dataset.py

[源代码](./src/dataset.py)

---

## lstm.py

[源代码](./src/lstm.py)

---

## models.py

[源代码](./src/models.py)

---

## svm.py

[源代码](./src/svm.py)

---

## train.py

[源代码](./src/train.py)

---
