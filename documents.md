# 项目文档

## 项目结构

- `directory` `dataset`
    - `file` `IMDB Dataset.csv`
- `directory` `lstm`
- `directory` `src`
    - `file` `__init__.py`
    - `file` [`convert.py`](#convertpy)
        - class [`Converter`](#class-Converter)
    - `file` [`dataset.py`](#datasetpy)
        - `class` [`IMDBDataset`](#class-IMDBDataset)
        - `class` [`TfIdfDataset`](#class-TfIdfDataset)
    - `file` [`lstm.py`](#lstmpy)
        - `class` [`LSTMModel`](#class-LSTMModel)
    - `file` [`models.py`](#modelspy)
        - `class` [`Review`](#class-Review)
        - `class` [`GridResult`](#class-GridResult)
        - `class` [`NNTrainingState`](#class-NNTrainingState)
    - `file` [`svm.py`](#svmpy)
        - `class` [`SymType`](#class-SymType)
        - `class` [`KernelType`](#class-KernelType)
        - `class` [`SVM`](#class-SVM)
    - `file` [`train.py`](#trainpy)
        - `class` [`Trainer`](#class-Trainer)
- `directory` `svm`
    - `directory` `model`
    - `directory` `data`

---

## convert.py

[源代码](./src/convert.py)

### class Converter

[`Converter`](#class-Converter)是[`IMDBDataset`](#class-IMDBDataset)和[`Trainer`](#class-Trainer)之间的桥梁，它提供了一些易于使用的API，可以将数据集中的原始数据转换为`SVM`或`Neutral Network`便于处理的数据形式，从而简化了数据处理的流程。

> Tips:[`tfidf_matrix`](#Converter-tfidf-matrix)、[`feature_names`](#Converter-feature-names)、[`tfidf_dataset`](#Converter-tfidf-dataset)属性无需显式调用[`tfidf()`](#method-tfidf())方法，内部会自动调用。即使你改变了[`dataset`](#Converter-dataset)属性，这些属性也会自动更新。

|                         属性                          |                                                         类型                                                         |                                                   初始值                                                   |                              描述                               |
|:---------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------:|
|       <a id="Converter-dataset">`dataset`</a>       | [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) |                                               `Required`                                                |                            要转换的数据集                            |
|                     `processes`                     |                                                       `int`                                                        | [`os.cpu_count()`](https://docs.python.org/zh-cn/3/library/os.html?highlight=os cpu_count#os.cpu_count) |                          多进程并行处理的进程数                          |
|  <a id="Converter-tfidf-matrix">`tfidf_matrix`</a>  |   [`scipy.sparse.csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)   |                                                    /                                                    |          [`dataset`](#Converter-dataset)的`tfidf`值矩阵           |
| <a id="Converter-feature-names">`feature_names`</a> |               [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)               |                                                    /                                                    |             [`dataset`](#Converter-dataset)的特征单词              |
|                       `items`                       |                                                       `list`                                                       |                                                    /                                                    |             [`dataset`](#Converter-dataset)中的所有元素             |
|                  `items_generator`                  |                                                    `Generator`                                                     |                                                    /                                                    |           [`dataset`](#Converter-dataset)中的所有元素的生成器           |
| <a id="Converter-tfidf-dataset">`tfidf_dataset`</a> |                                   [`dataset.TfIdfDataset`](#class-TfIdfDataset)                                    |                                                    /                                                    | 包含了[`dataset`](#Converter-dataset)的`tfidf`和`label`的数据集，用于后续训练 |

#### method \_\_init__()

初始化一个[`Converter`](#class-Converter)实例

##### 输入

|     参数      |                                                         类型                                                         |                                                   初始值                                                   |       描述       |
|:-----------:|:------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:--------------:|
|  `dataset`  | [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) |                                               `Required`                                                |    要转换的数据集     |
| `processes` |                                                       `int`                                                        | [`os.cpu_count()`](https://docs.python.org/zh-cn/3/library/os.html?highlight=os cpu_count#os.cpu_count) | `to_svm`方法的进程数 |

##### 输出

`None`

#### method tfidf()

计算数据集的TF-IDF表示，同时更新自身的[`tfidf_matrix`](#Converter-tfidf-matrix)、[`feature_names`](#Converter-feature-names)、[`tfidf_dataset`](#Converter-tfidf-dataset)属性，返回[`tfidf_dataset`](#Converter-tfidf-dataset)

##### 输入

`None`

##### 输出

一个[`dataset.TfIdfDataset`](#class-TfIdfDataset)实例，其实就是[`self.tfidf_dataset`](#Converter-tfidf-dataset)

#### method word2vec()

**TODO**

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

[`IMDBDataset`](#class-IMDBDataset)继承了[`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)，它从`csv`文件中读取数据集，转化为特定的数据结构，便于开展后续的数据处理工作。

`__getitem__()`方法返回的数据结构取决于`get_item_by_tuple`属性。

`__len__()`方法返回数据集的长度。

> Tips: `IMDBDataset`是可迭代的

|         属性          |                                           类型                                           |    初始值     |                                              描述                                               |
|:-------------------:|:--------------------------------------------------------------------------------------:|:----------:|:---------------------------------------------------------------------------------------------:|
|    `save_memory`    |                                         `bool`                                         |  `False`   |                                         **只读**，是否节省内存                                         |
| `get_item_by_tuple` |                                         `bool`                                         |  `False`   | `True`时，`__getitem__()`方法返回一个元组<br/>`False`时，`__getitem__()`方法返回一个[`Review`](#class-Review)实例 |
| `dataset_pathname`  |                                         `str`                                          | `Required` |                                      `csv`文件的路径，**必须**存在                                      |
|   `dataset_title`   |                                         `str`                                          |     /      |                                    **只读**，`csv`文件的文件名，自动更新                                    |
|       `item`        |                            [`models.Review`](#class-Review)                            |     /      |                         **只读**，数据集中的当前项<br/>`save_memory = False`时无效                          |
|       `items`       | [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) |     /      |                       **只读**，数据集中的所有项<br/>`save_memory = True`时为`None`                        |

#### method \_\_init__()

初始化一个[`IMDBDataset`](#class-IMDBDataset)实例

##### 输入

|         参数          |   类型   |    初始值     |                                              描述                                               |
|:-------------------:|:------:|:----------:|:---------------------------------------------------------------------------------------------:|
| `dataset_pathname`  | `str`  | `Required` |                                        数据集路径，**必须**存在                                         |
|    `save_memory`    | `bool` |  `False`   |                                            是否节省内存                                             |
| `get_item_by_tuple` | `bool` |  `False`   | `True`时，`__getitem__()`方法返回一个元组<br/>`False`时，`__getitem__()`方法返回一个[`Review`](#class-Review)实例 |

##### 输出

`None`

### class TfIdfDataset

[`TfIdfDataset`](#class-TfIdfDataset)继承了[`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)，用于神经网络训练。

一般由[`Converter`](#class-Converter)自动生成，用户**不需要**自行创建该类的实例。

|    属性    |                                                       类型                                                       |    初始值     |   描述    |
|:--------:|:--------------------------------------------------------------------------------------------------------------:|:----------:|:-------:|
| `values` | [`scipy.sparse.csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) | `Required` | TF-IDF值 |
| `labels` |             [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)             | `Required` |  情感标签   |

#### method \_\_init__()

初始化一个[`TfIdfDataset`](#class-TfIdfDataset)实例

##### 输入

|   参数   |                             类型                             |   初始值   |        描述        |
| :------: | :----------------------------------------------------------: | :--------: | :----------------: |
| `values` | [`scipy.sparse.csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) | `Required` | **只读**，TF-IDF值 |
| `labels` | [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) | `Required` | **只读**，情感标签 |

##### 输出

`None`

---

## lstm.py

[源代码](./src/lstm.py)

### class LSTMModel

[`LSTMModel`](#class-LSTMModel)继承了[`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module#torch.nn.Module)，是一个`LSTM`模型，用于神经网络训练。

|       属性       |                                                                类型                                                                 |                                                                          初始值                                                                          |                                  描述                                  |
|:--------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|    `device`    |              [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html?highlight=device#torch.device)               |                                                                 `torch.device('cpu')`                                                                 |                          **只读**，训练模型时使用的设备                           |
|  `input_dim`   |                                                               `int`                                                               |                                                                      `Required`                                                                       |                             **只读**，输入维数                              |
|  `hidden_dim`  |                                                               `int`                                                               |                                                                      `Required`                                                                       |                             **只读**，隐藏层维数                             |
|  `output_dim`  |                                                               `int`                                                               |                                                                      `Required`                                                                       |                             **只读**，输出维数                              |
|  `num_layers`  |                                                               `int`                                                               |                                                                      `Required`                                                                       |                            **只读**，LSTM层数                             |
|     `lstm`     |       [`torch.nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=torch+nn+lstm#torch.nn.LSTM)       |                                                                           /                                                                           |                            **只读**，LSTM模型                             |
|      `fc`      |   [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module#torch.nn.Module)   | [`torch.nn.Linear(hidden_dim, output_dim)`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=torch+nn+linear#torch.nn.Linear) |                            **只读**，全连接层模型                             |
| `dropout_rate` |                                                              `float`                                                              |                                                                          `0`                                                                          | `dropout_rate∈[0,1]`<br/>`dropout`层要丢弃的神经元的比例<br/>等于`0`时禁用`dropout`层 |
|   `dropout`    | [`torch.nn.Dropout`](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html?highlight=torch+nn+dropout#torch.nn.Dropout) |                                                                           /                                                                           |              **只读**，`dropout`层模型，与`dropout_rate`属性自动同步               |
|   `sigmoid`    | [`torch.nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html?highlight=torch+nn+sigmoid#torch.nn.Sigmoid) |                                                                           /                                                                           |                         **只读**，`Sigmoid`层模型                          |

#### method \_\_init__()

初始化一个`LSTMModel`实例

##### 输入

|       参数       |                                                              类型                                                               |          初始值          |                                  描述                                  |
|:--------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:---------------------:|:--------------------------------------------------------------------:|
|  `input_dim`   |                                                             `int`                                                             |      `Required`       |                                 输入维数                                 |
|  `hidden_dim`  |                                                             `int`                                                             |      `Required`       |                                隐藏层维数                                 |
|  `output_dim`  |                                                             `int`                                                             |      `Required`       |                                 输出维数                                 |
|  `num_layers`  |                                                             `int`                                                             |      `Required`       |                                LSTM层数                                |
|      `fc`      | [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module#torch.nn.Module) |        `None`         |                                全连接层模型                                |
| `dropout_rate` |                                                            `float`                                                            |          `0`          | `dropout_rate∈[0,1]`<br/>`dropout`层要丢弃的神经元的比例<br/>等于`0`时禁用`dropout`层 |
|    `device`    |            [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html?highlight=device#torch.device)             | `torch.device('cpu')` |                              训练模型时使用的设备                              |

##### 输出

`None`

---

## models.py

[源代码](./src/models.py)

### class Review

[`Review`](#class-Review)继承了[`pydantic.BaseModel`](https://docs.pydantic.dev/1.10/usage/models/)，一个[`Review`](#class-Review)实例描述了一个电影评论

|     属性      |   类型   |    初始值     |  描述  |
|:-----------:|:------:|:----------:|:----:|
|  `review`   | `str`  | `Required` | 评论内容 |
| `sentiment` | `bool` | `Required` | 情感标签 |

### class GridResult

[`GridResult`](#class-GridResult)继承了[`pydantic.BaseModel`](https://docs.pydantic.dev/1.10/usage/models/)，一个[`GridResult`](#class-GridResult)实例描述了一个网格搜索的结果

|   属性    |   类型    |    初始值     |    描述     |
|:-------:|:-------:|:----------:|:---------:|
| `c_min` | `float` | `Required` |   惩罚系数    |
| `c_max` | `float` | `Required` | 惩罚系数乘以步进  |
| `g_min` | `float` | `Required` |   核函数参数   |
| `g_max` | `float` | `Required` | 核函数参数乘以步进 |
| `rate`  | `float` | `Required` |    准确率    |

### class NNTrainingState

[`NNTrainingState`](#class-NNTrainingState)继承了[`pydantic.BaseModel`](https://docs.pydantic.dev/1.10/usage/models/)，一个[`NNTrainingState`](#class-NNTrainingState)实例描述了一个神经网络训练的状态

|           属性           |   类型   | 初始值  |   描述   |
|:----------------------:|:------:|:----:|:------:|
|    `current_epoch`     | `int`  | `0`  | 当前训练轮次 |
|     `total_epoch`      | `int`  | `0`  | 总训练轮次  |
|   `model_state_dict`   | `dict` | `{}` |  模型参数  |
| `optimizer_state_dict` | `dict` | `{}` | 优化器参数  |


---

## svm.py

[源代码](./src/svm.py)

### class SymType

[`SymType`](#class-SymType)是`int`类型的枚举，提供了5种SVM类型

|       类型        |  值  |
|:---------------:|:---:|
|     `C_SVC`     | `0` |
|    `NU_SVC`     | `1` |
| `ONE_CLASS_SVM` | `2` |
|  `EPSILON_SVR`  | `3` |
|    `NU_SVR`     | `4` |

### class KernelType

[`KernelType`](#class-KernelType)是`int`类型的枚举，提供了5种核函数类型

|           类型            |  值  |
|:-----------------------:|:---:|
|        `LINEAR`         | `0` |
|      `POLYNOMIAL`       | `1` |
| `RADIAL_BASIS_FUNCTION` | `2` |
|        `SIGMOID`        | `3` |
|  `PRECOMPUTED_KERNEL`   | `4` |
### class SVM

[`SVM`](#class-SVM)提供了对SVM模型的封装，可以进行训练、预测、保存和加载等操作

|       属性       |                                                                   类型                                                                   |  初始值   |    描述     |
|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:------:|:---------:|
|    `model`     | [`libsvm.svmutil.svm_model`](https://github.com/cjlin1/libsvm/blob/aed66346593ec0e075f38eda10fef0c1fb132692/python/libsvm/svm.py#L352) | `None` |   SVM模型   |
| `grid_results` |                                             [`list[models.GridResult]`](#class-GridResult)                                             | `None` | 网格搜索的结果列表 |


#### method \_\_init__()

初始化一个`SVM`实例

##### 输入

`None`

##### 输出

`None`

#### method load()

加载一个SVM模型，`model`和`model_path`至少有一个不为`None`，若两者都不为`None`，则`model`优先

##### 输入

|      参数      |                                                                   类型                                                                   |  初始值   |  描述   |
|:------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:------:|:-----:|
|   `model`    | [`libsvm.svmutil.svm_model`](https://github.com/cjlin1/libsvm/blob/aed66346593ec0e075f38eda10fef0c1fb132692/python/libsvm/svm.py#L352) | `None` | SVM模型 |
| `model_path` |                                                                 `str`                                                                  | `None` | 模型路径  |

##### 输出

这个实例自身，即`self`

#### method save()

保存一个SVM模型

##### 输入

|   参数   |  类型   |    初始值     |   描述   |
|:------:|:-----:|:----------:|:------:|
| `path` | `str` | `Required` | 模型保存路径 |

##### 输出

保存路径的绝对路径

#### <a name="svm-train">method train()</a>

训练一个SVM模型

##### 输入

|           参数            |                类型                 |    初始值     |                                        描述                                         |
|:-----------------------:|:---------------------------------:|:----------:|:---------------------------------------------------------------------------------:|
|     `problem_path`      |               `str`               | `Required` |                                  标准libsvm格式训练集路径                                  |
|       `sym_type`        |    [`SymType`](#class-SymType)    |   `None`   |                            set type of SVM (default 0)                            |
|      `kernel_type`      | [`KernelType`](#class-KernelType) |   `None`   |                      set type of kernel function (default 2)                      |
|        `degree`         |               `int`               |   `None`   |                       degree in kernel function (default 3)                       |
|         `gamma`         |              `float`              |   `None`   |                 gamma in kernel function (default 1/num_features)                 |
|         `coef0`         |              `float`              |   `None`   |                       coef0 in kernel function (default 0)                        |
|         `cost`          |              `float`              |   `None`   |                cost in C-SVC, epsilon-SVR, and nu-SVR (default 1)                 |
|          `nu`           |              `float`              |   `None`   |               nu in nu-SVC, one-class SVM, and nu-SVR (default 0.5)               |
|        `epsilon`        |              `float`              |   `None`   |             the epsilon in loss function of epsilon-SVR (default 0.1)             |
|      `cache_size`       |              `float`              |   `None`   |                     set cache memory size in MB (default 100)                     |
|       `tolerance`       |              `float`              |   `None`   |              set tolerance of termination criterion (default 0.001)               |
|       `shrinking`       |               `int`               |   `None`   |            whether to use the shrinking heuristics, 0 or 1 (default 1)            |
| `probability_estimates` |               `int`               |   `None`   | whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0) |
|        `weight`         |              `float`              |   `None`   |           the parameter C of class i to weight*C, for C-SVC (default 1)           |
|        `n_fold`         |               `int`               |   `None`   |                           n-fold cross validation mode                            |

##### 输出

训练结果，当`n_fold`不为`None`时，返回交叉验证的结果，否则返回训练的模型

#### method predict()

预测一个SVM模型。

当`self.model is None`时，`model`和`model_path`至少有一个不为`None`，若两者都不为`None`，则`model`优先。
当`self.model is not None`时，忽略`model`和`model_path`参数。

##### 输入

|           参数            |                                                                   类型                                                                   |    初始值     |                                        描述                                         |
|:-----------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:----------:|:---------------------------------------------------------------------------------:|
|     `problem_path`      |                                                                 `str`                                                                  | `Required` |                                  标准libsvm格式测试集路径                                  |
|         `model`         | [`libsvm.svmutil.svm_model`](https://github.com/cjlin1/libsvm/blob/aed66346593ec0e075f38eda10fef0c1fb132692/python/libsvm/svm.py#L352) |   `None`   |                                      训练好的模型                                       |
|      `model_path`       |                                                                 `str`                                                                  |   `None`   |                                     训练好的模型的路径                                     |
| `probability_estimates` |                                                                 `int`                                                                  |   `None`   | whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0) |

##### 输出

- `tuple[list, tuple[float, float, float], list]`
  - `list` p_labels: A list of predicted labels
  - `tuple[float, float, float]` p_acc
    - `float` accuracy (for classification)
    - `float` mean-squared error
    - `float` squared correlation coefficient (for regression)
  - `list` p_vals: A list of decision values or probability estimates (if `probability_estimates == 1`). If `k` is the number of classes, for decision values, each element includes results of predicting `k(k-1)/2` binary-class SVMs. For probabilities, each element contains `k` values indicating the probability that the testing instance is in each class. Note that the order of classes here is the same as 'model.label' field in the model structure.

#### method grid()

网格搜索，寻找最优参数。

因为筛选过程耗时很久，所以增加了程序终止时自动保存功能（使用[`Trainer`](#class-Trainer)），并且可以从中断处继续筛选。

如果要从中断处恢复筛选，应设置`from_record = True`，`record_path = {your_file_path}`，`your_file_path`为自动保存的训练状态文件，保持训练参数和原来的训练参数一致。（具体来说，训练参数是指`problem_path`、`c_min`、`c_max`、`c_step`、`g_min`、`g_max`、`g_step`）
此处的`problem_path`与原来一致是指该路径指定的训练集文件内容不变。

> 后续版本将会改进自动保存机制，恢复筛选将不需要输入训练参数，而是从保存文件中自动读取。

##### 输入

|        参数        |   类型    |                初始值                |                                           描述                                            |
|:----------------:|:-------:|:---------------------------------:|:---------------------------------------------------------------------------------------:|
|  `problem_path`  |  `str`  |              `None`               |                                     标准libsvm格式训练集路径                                     |
|     `n_fold`     |  `int`  |                `5`                |                                         交叉验证折数                                          |
| `enable_logging` | `bool`  |              `False`              |                                        是否打印搜索进度                                         |
|     `c_min`      | `float` |              `1e-8`               |                                        Cost的最小值                                         |
|     `c_max`      | `float` |               `1e8`               |                                        Cost的最大值                                         |
|     `c_step`     | `float` |               `10`                |                                         Cost的步长                                         |
|     `g_min`      | `float` |              `1e-8`               |                                        gamma的最小值                                        |
|     `g_max`      | `float` |               `1e8`               |                                        gamma的最大值                                        |
|     `g_step`     | `float` |               `10`                |                                        gamma的步长                                         |
|    `detailed`    | `bool`  |              `False`              | 是否返回详细信息<br/>`detailed = True`时，返回所有搜索结果的列表，并绘制结果图像<br/>`detailed = False`时，只返回准确度最高的结果 |
|    `img_name`    |  `str`  | `r'..\svm\train\grid_result.png'` |                                         保存的图片名                                          |
|      `dpi`       |  `int`  |              `1000`               |                                         图片的dpi                                          |
|  `from_record`   | `bool`  |              `False`              |                                       是否从记录文件中读取                                        |
|  `record_path`   |  `str`  |              `None`               |                                         记录文件路径                                          |

##### 输出

`detailed = True`时，返回所有搜索结果的列表，并绘制结果图像
`detailed = False`时，只返回准确度最高的结果


---

## train.py

[源代码](./src/train.py)

### class Trainer

[`Trainer`](#class-Trainer)封装了一些用于训练SVM和神经网络的API。它提供了程序终止时自动保存训练结果等有用的功能。

|         属性         |                                                              类型                                                               |          初始值          |                                                   描述                                                    |
|:------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:---------------------:|:-------------------------------------------------------------------------------------------------------:|
|     `autosave`     |                                                            `bool`                                                             |        `True`         |                                               是否自动保存训练结果                                                |
|   `autosave_dir`   |                                                             `str`                                                             |   `r'..\autosave'`    |                                             自动保存的**文件夹**路径                                              |
|  `tfidf_dataset`   |      [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)       |        `None`         | 存有`tfidf`和`label`的数据集，一般使用[`Converter`](#class-Converter)的[`tfidf_dataset`](#Converter-tfidf-dataset)属性 |
| `word2vec_dataset` |      [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)       |        `None`         |                                                **TODO**                                                 |
|  `svm_train_path`  |                                                             `str`                                                             |        `None`         |                                          SVM的训练集文件，**必须**实际存在                                           |
|  `svm_model_path`  |                                                             `str`                                                             |        `None`         |                                           SVM的模型文件，**必须**实际存在                                           |
|       `svm`        |                                                    [`svm.SVM`](#class-SVM)                                                    | [`SVM()`](#class-SVM) |                               svm实例，训练SVM时使用该属性的[`train()`](#svm-train)方法                               |
|      `model`       | [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module#torch.nn.Module) |        `None`         |                   要训练的神经网络模型，一般使用[`lstm.LSTMModel`](#class-LSTMModel)，也可以传入其它`Module`                   |
|    `optimizer`     |        [`torch.optim.Optimizer`](https://pytorch.org/docs/stable/optim.html?highlight=optimizer#torch.optim.Optimizer)        |        `None`         |                                              训练神经网络时使用的优化器                                              |
|    `criterion`     | [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module#torch.nn.Module) |        `None`         |                                             训练神经网络时使用的损失函数                                              |
|      `device`      |            [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html?highlight=device#torch.device)             |        `None`         |                                              训练神经网络时使用的设备                                               |

#### method \_\_init\_\_()

初始化[`Trainer`](#class-Trainer)实例

##### 输入

|         参数         |                                                              类型                                                               |       初始值        |                                                   描述                                                    |
|:------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:----------------:|:-------------------------------------------------------------------------------------------------------:|
|     `autosave`     |                                                            `bool`                                                             |      `True`      |                                               是否自动保存训练结果                                                |
|   `autosave_dir`   |                                                             `str`                                                             | `r'..\autosave'` |                                             自动保存的**文件夹**路径                                              |
|  `tfidf_dataset`   |      [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)       |      `None`      | 存有`tfidf`和`label`的数据集，一般使用[`Converter`](#class-Converter)的[`tfidf_dataset`](#Converter-tfidf-dataset)属性 |
| `word2vec_dataset` |      [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)       |      `None`      |                                                **TODO**                                                 |
|  `svm_train_path`  |                                                             `str`                                                             |      `None`      |                                          SVM的训练集文件，**必须**实际存在                                           |
|  `svm_model_path`  |                                                             `str`                                                             |      `None`      |                                           SVM的模型文件，**必须**实际存在                                           |
|      `model`       | [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module#torch.nn.Module) |      `None`      |                   要训练的神经网络模型，一般使用[`lstm.LSTMModel`](#class-LSTMModel)，也可以传入其它`Module`                   |
|    `optimizer`     |        [`torch.optim.Optimizer`](https://pytorch.org/docs/stable/optim.html?highlight=optimizer#torch.optim.Optimizer)        |      `None`      |                                              训练神经网络时使用的优化器                                              |
|    `criterion`     | [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module#torch.nn.Module) |      `None`      |                                             训练神经网络时使用的损失函数                                              |
|      `device`      |            [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html?highlight=device#torch.device)             |      `None`      |                                              训练神经网络时使用的设备                                               |

##### 输出

`None`

#### method train()

训练神经网络

因为训练过程耗时很久，所以增加了程序终止时自动保存功能，并且可以从中断处继续训练。

如果要从中断处恢复训练，应设置`from_record = True`，`record_path = {your_file_path}`，`your_file_path`为自动保存的训练状态文件，保持训练参数和原来的训练参数一致。（具体来说，训练参数是指`train_loader`、`num_epochs`）
此处的`train_loader`与原来一致是指该`DataLoader`存储的训练集内容不变。

> 后续版本将会改进自动保存机制，恢复训练将不需要输入训练参数，而是从保存文件中自动读取。

##### 输入

|       参数        |  类型  | 初始值  |                             描述                             |
| :---------------: | :----: | :-----: | :----------------------------------------------------------: |
|     `epochs`      | `int`  |   `1`   |                          训练的轮数                          |
|    `svm_mode`     | `bool` | `False` |   **关键字参数**，以`self.tfidf_dataset`作为训练集进行训练   |
|  `word2vec_mode`  | `bool` | `False` | **关键字参数**，以`self.word2vec_dataset`作为训练集进行训练  |
| `enable_logging`  | `bool` | `False` |               **关键字参数**，是否打印训练进度               |
|   `from_record`   | `bool` | `False` |         **关键字参数**，是否从记录文件中读取训练结果         |
|   `record_path`   | `str`  | `None`  |                 **关键字参数**，记录文件路径                 |
| `**loader_kwargs` | `dict` | `None`  | 见[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)，不能传入`dataset`参数 |

##### 输出

这个实例自身，即`self`

#### method evaluate()

评估神经网络

##### 输入

|      参数       |                                                             类型                                                              |    初始值     |         描述         |
|:-------------:|:---------------------------------------------------------------------------------------------------------------------------:|:----------:|:------------------:|
| `test_loader` | [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader) | `Required` | 存放测试集的`DataLoader` |

##### 输出

模型预测准确度

#### method predict()

使用神经网络预测

##### 输入

|   参数    |                                              类型                                              |    初始值     |    描述    |
|:-------:|:--------------------------------------------------------------------------------------------:|:----------:|:--------:|
| `texts` | [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html?highlight=tensor#torch.Tensor) | `Required` | 要预测的文本张量 |

##### 输出

预测结果，类型为[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html?highlight=tensor#torch.Tensor)

#### method save()

保存训练结果

##### 输入

|     参数      |  类型   |             初始值             |    描述     |
|:-----------:|:-----:|:---------------------------:|:---------:|
| `save_path` | `str` | `r'..\lstm\model\lstm.pth'` | 神经网络的保存路径 |

##### 输出

保存路径的绝对路径

#### method load()

加载神经网络

##### 输入

|     参数      |  类型   |             初始值             |     描述      |
|:-----------:|:-----:|:---------------------------:|:-----------:|
| `load_path` | `str` | `r'..\lstm\model\lstm.pth'` | 要加载的神经网络的路径 |

##### 输出

这个实例自身，即`self`