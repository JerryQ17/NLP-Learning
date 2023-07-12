# 项目文档

## 项目结构

- directory `dataset`
    - file `IMDB Dataset.csv`
- directory `src`
    - file `__init__.py`
    - file [`convert.py`](#convert.py)
        - class [`Converter`](#class-Converter)
    - file [`dataset.py`](#dataset.py)
        - class [`IMDBDataset`](#class IMDBDataset)
        - class [`TfIdfDataset`](#class TfIdfDataset)
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

>
Tips:[`tfidf_matrix`](#Converter-tfidf-matrix) 、[`feature_names`](#Converter-feature-names) 、[`tfidf_dataset`](#Converter-tfidf-dataset)
属性无需显式调用`tfidf()`
> 方法，内部会自动调用。即使你改变了[`dataset`](#Converter-dataset)属性，这些属性也会自动更新。

|                         属性                          |             类型             |       初始值        |                              描述                               |
|:---------------------------------------------------:|:--------------------------:|:----------------:|:-------------------------------------------------------------:|
|       <a id="Converter-dataset">`dataset`</a>       | `torch.utils.data.Dataset` |    `Required`    |                            要转换的数据集                            |
|                     `processes`                     |           `int`            | `os.cpu_count()` |                        `to_svm`方法的进程数                         |
|  <a id="Converter-tfidf-matrix">`tfidf_matrix`</a>  | `scipy.sparse.csr_matrix`  |        /         |          [`dataset`](#Converter-dataset)的`tfidf`值矩阵           |
| <a id="Converter-feature-names">`feature_names`</a> |      `numpy.ndarray`       |        /         |             [`dataset`](#Converter-dataset)的特征单词              |
|                       `items`                       |           `list`           |        /         |             [`dataset`](#Converter-dataset)中的所有元素             |
|                  `items_generator`                  |        `Generator`         |        /         |           [`dataset`](#Converter-dataset)中的所有元素的生成器           |
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

计算数据集的TF-IDF表示，同时更新自身的[`tfidf_matrix`](#Converter-tfidf-matrix) 、[`feature_names`](#Converter-feature-names) 、[`tfidf_dataset`](#Converter-tfidf-dataset)
属性，返回[`tfidf_dataset`](#Converter-tfidf-dataset)

##### 输入

`None`

##### 输出

一个`dataset.TfIdfDataset`实例，其实就是[`self.tfidf_dataset`](#Converter-tfidf-dataset)

#### method word2vec() **TODO**

##### 输入

##### 输出

#### method to_svm()

将[`dataset`](#Converter-dataset)中的原始数据转换成`tfidf`矩阵（如果没有转换过的话），并将`tfidf`矩阵存储为标准`libsvm`
格式，返回文件保存的绝对路径

##### 输入

|     参数      |  类型   |  初始值   |                   描述                    |
|:-----------:|:-----:|:------:|:---------------------------------------:|
| `save_path` | `str` | `None` | 保存路径，默认保存在`'..\svm\data`文件夹，实际保存文件名见返回值 |

##### 输出

文件保存的绝对路径

---

## dataset.py

[源代码](./src/dataset.py)

### class IMDBDataset

`IMDBDataset`继承了`torch.utils.data.Dataset`，它从`csv`文件中读取数据集，转化为特定的数据结构，便于开展后续的数据处理工作。

> Tips: `IMDBDataset`是可迭代的

|         属性          |       类型        |    初始值     |                                      描述                                      |
|:-------------------:|:---------------:|:----------:|:----------------------------------------------------------------------------:|
|    `save_memory`    |     `bool`      |  `False`   |                                **只读**，是否节省内存                                 |
| `get_item_by_tuple` |     `bool`      |  `False`   | `True`时，`__getitem__()`方法返回一个元组<br/>`False`时，`__getitem__()`方法返回一个`Review`实例 |
| `dataset_pathname`  |      `str`      | `Required` |                             `csv`文件的路径，**必须**存在                              |
|   `dataset_title`   |      `str`      |     /      |                           **只读**，`csv`文件的文件名，自动更新                            |
|       `item`        |    `Review`     |     /      |                 **只读**，数据集中的当前项<br/>`save_memory = False`时无效                 |
|       `items`       | `numpy.ndarray` |     /      |               **只读**，数据集中的所有项<br/>`save_memory = True`时为`None`               |

#### method \_\_init__()

初始化一个`IMDBDataset`实例

##### 输入

|         参数          |   类型   |    初始值     |                                      描述                                      |
|:-------------------:|:------:|:----------:|:----------------------------------------------------------------------------:|
| `dataset_pathname`  | `str`  | `Required` |                                数据集路径，**必须**存在                                |
|    `save_memory`    | `bool` |  `False`   |                                    是否节省内存                                    |
| `get_item_by_tuple` | `bool` |  `False`   | `True`时，`__getitem__()`方法返回一个元组<br/>`False`时，`__getitem__()`方法返回一个`Review`实例 |

##### 输出

`None`

### class TfIdfDataset

`TfIdfDataset`继承了`torch.utils.data.Dataset`，用于神经网络训练。一般由[`Converter`](#class-Converter)
自动生成，用户不需要自行创建该类的实例。

|    属性    |            类型             |    初始值     |   描述    |
|:--------:|:-------------------------:|:----------:|:-------:|
| `values` | `scipy.sparse.csr_matrix` | `Required` | TF-IDF值 |
| `labels` |      `numpy.ndarray`      | `Required` |  情感标签   |

#### method \_\_init__()

初始化一个`TfIdfDataset`实例

##### 输入

|    参数    |            类型             |    初始值     |   描述    |
|:--------:|:-------------------------:|:----------:|:-------:|
| `values` | `scipy.sparse.csr_matrix` | `Required` | TF-IDF值 |
| `labels` |      `numpy.ndarray`      | `Required` |  情感标签   |

##### 输出

`None`

---

## lstm.py

[源代码](./src/lstm.py)

### class LSTMModel

`LSTMModel`继承了`torch.nn.Module`，是一个`LSTM`模型，用于神经网络训练。

|       属性       |         类型         |                    初始值                    |                                  描述                                  |
|:--------------:|:------------------:|:-----------------------------------------:|:--------------------------------------------------------------------:|
|    `device`    |   `torch.device`   |           `torch.device('cpu')`           |                          **只读**，训练模型时使用的设备                           |
|  `input_dim`   |       `int`        |                `Required`                 |                             **只读**，输入维数                              |
|  `hidden_dim`  |       `int`        |                `Required`                 |                             **只读**，隐藏层维数                             |
|  `output_dim`  |       `int`        |                `Required`                 |                             **只读**，输出维数                              |
|  `num_layers`  |       `int`        |                `Required`                 |                            **只读**，LSTM层数                             |
|     `lstm`     |  `torch.nn.LSTM`   |                     /                     |                            **只读**，LSTM模型                             |
|      `fc`      | `torch.nn.Module`  | `torch.nn.Linear(hidden_dim, output_dim)` |                            **只读**，全连接层模型                             |
| `dropout_rate` |      `float`       |                    `0`                    | `dropout_rate∈[0,1]`<br/>`dropout`层要丢弃的神经元的比例<br/>等于`0`时禁用`dropout`层 |
|   `dropout`    | `torch.nn.Dropout` |                     /                     |              **只读**，`dropout`层模型，与`dropout_rate`属性自动同步               |
|   `sigmoid`    | `torch.nn.Sigmoid` |                     /                     |                         **只读**，`Sigmoid`层模型                          |

#### method \_\_init__()

初始化一个`LSTMModel`实例

##### 输入

|       参数       |        类型         |          初始值          |                                  描述                                  |
|:--------------:|:-----------------:|:---------------------:|:--------------------------------------------------------------------:|
|  `input_dim`   |       `int`       |      `Required`       |                                 输入维数                                 |
|  `hidden_dim`  |       `int`       |      `Required`       |                                隐藏层维数                                 |
|  `output_dim`  |       `int`       |      `Required`       |                                 输出维数                                 |
|  `num_layers`  |       `int`       |      `Required`       |                                LSTM层数                                |
|      `fc`      | `torch.nn.Module` |        `None`         |                                全连接层模型                                |
| `dropout_rate` |      `float`      |          `0`          | `dropout_rate∈[0,1]`<br/>`dropout`层要丢弃的神经元的比例<br/>等于`0`时禁用`dropout`层 |
|    `device`    |  `torch.device`   | `torch.device('cpu')` |                              训练模型时使用的设备                              |

##### 输出

`None`

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
