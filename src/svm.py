import os
import sys
import time
from math import log
from torch import load
from enum import IntEnum
from .models import GridResult
import matplotlib.pyplot as plt
from libsvm.svmutil import (
    svm_model,
    svm_train, svm_predict,
    svm_save_model, svm_load_model,
    svm_read_problem
)


class SymType(IntEnum):
    C_SVC = 0
    NU_SVC = 1
    ONE_CLASS_SVM = 2
    EPSILON_SVR = 3
    NU_SVR = 4


class KernelType(IntEnum):
    LINEAR = 0
    POLYNOMIAL = 1
    RADIAL_BASIS_FUNCTION = 2
    SIGMOID = 3
    PRECOMPUTED_KERNEL = 4


class SVM(object):
    """SVM模型"""
    __svm_train_options = {
        'sym_type': '-s',
        'kernel_type': '-t',
        'degree': '-d',
        'gamma': '-g',
        'coef0': '-r',
        'cost': '-c',
        'nu': '-n',
        'epsilon': '-p',
        'cache_size': '-m',
        'tolerance': '-e',
        'shrinking': '-h',
        'probability_estimates': '-b',
        'weight': '-wi',
        'n_fold': '-v'
    }

    def __init__(self):
        self.__model: svm_model | None = None
        self.__grid_results: list[GridResult] = []

    @property
    def model(self):
        return self.__model

    @property
    def grid_results(self) -> list[GridResult]:
        return self.__grid_results

    def load(self, model: svm_model = None, model_path: str = None) -> 'SVM':
        """加载模型，model和model_path必须至少指定一个"""
        if model is not None:
            if not isinstance(model, svm_model):
                raise TypeError('model必须是svm_model类型')
            self.__model = model
        elif model_path is not None:
            if not isinstance(model_path, str):
                raise TypeError('model_path必须是str类型')
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f'模型文件{model_path}不存在')
            self.__model = svm_load_model(model_path)
        else:
            raise ValueError('model和model_path必须至少指定一个')
        return self

    def save(self, path: str) -> str:
        """保存模型"""
        svm_save_model(path, self.__model)
        return os.path.abspath(path)

    def train(
            self, problem_path: str,
            sym_type: SymType = None, kernel_type: KernelType = None,
            degree: int = None, gamma: float = None, coef0: float = None, cost: float = None,
            nu: float = None, epsilon: float = None, cache_size: float = None, tolerance: float = None,
            shrinking: int = None, probability_estimates: int = None, weight: float = None, n_fold: int = None,
    ) -> svm_model | float:
        """
        使用libsvm训练svm模型
        :param problem_path: 标准libsvm格式训练集路径
        :param sym_type: set type of SVM (default 0)
        :param kernel_type: set type of kernel function (default 2)
        :param degree: set degree in kernel function (default 3)
        :param gamma: set gamma in kernel function (default 1/num_features)
        :param coef0: set coef0 in kernel function (default 0)
        :param cost: set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        :param nu: set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
        :param epsilon: set the epsilon in loss function of epsilon-SVR (default 0.1)
        :param cache_size: set cache memory size in MB (default 100)
        :param tolerance: set tolerance of termination criterion (default 0.001)
        :param shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
        :param probability_estimates: whether to train an SVC or SVR model for probability estimates, 0 or 1 (default 0)
        :param weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
        :param n_fold: n-fold cross validation mode
        :return: 训练好的模型
        """
        if not os.path.isfile(problem_path):
            raise FileNotFoundError(f'数据文件{problem_path}不存在')
        if shrinking is not None and shrinking not in (0, 1):
            raise ValueError('shrinking必须是0或1')
        if probability_estimates is not None and probability_estimates not in (0, 1):
            raise ValueError('probability_estimates必须是0或1')
        # 生成参数字符串
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('problem_path')
        param_str = ''
        for key, value in kwargs.items():
            if value is not None:
                param_str += f'{self.__svm_train_options[key]} {value} '

        # 读取数据，训练模型
        prob = svm_read_problem(problem_path)
        if param_str == '':
            result = svm_train(*prob)
        else:
            result = svm_train(*prob, param_str)
        if n_fold is None:
            self.__model = result
        return result

    def predict(
            self,
            problem_path: str,
            model: svm_model = None,
            model_path: str = None,
            probability_estimates: int = None,
    ) -> tuple[list, tuple[float, float, float], list]:
        """
        使用libsvm预测
        :param problem_path: 标准libsvm格式测试集路径
        :param model: 训练好的模型
        :param model_path: 训练好的模型保存路径
        :param probability_estimates: whether to predict probability estimates, 0 or 1 (default 0)
        :return: 预测结果
        """
        # 检查参数
        if self.__model is None:
            if model is None:
                if model_path is None:
                    raise ValueError('模型未加载')
                if not os.path.exists(model_path):
                    raise FileNotFoundError('模型文件不存在')
                model = svm_load_model(model_path)
            else:
                if model_path is not None and os.path.exists(model_path):
                    print('Warning: 因为model和model_path同时存在，model_path参数已被忽略', file=sys.stderr)
        else:
            model = self.__model
            if model is not None:
                print('Warning: 因为已加载模型，model参数已被忽略', file=sys.stderr)
            if model_path is not None:
                print('Warning: 因为已加载模型，model_path参数已被忽略', file=sys.stderr)
        if probability_estimates is not None and probability_estimates not in (0, 1):
            raise ValueError('probability_estimates必须是0或1')

        # 生成参数字符串
        param_str = ''
        if probability_estimates is not None:
            param_str = '-b ' + str(probability_estimates)
        # 测试模型
        y, x = svm_read_problem(problem_path)
        return svm_predict(y, x, model, param_str)

    def grid(
            self,
            problem_path: str, n_fold: int = 5, enable_logging: bool = False,
            c_min: float = 1e-8, c_max: float = 1e8, c_step: float = 10,
            g_min: float = 1e-8, g_max: float = 1e8, g_step: float = 10,
            detailed: bool = False, img_name: str = r'.\svm\train\grid_result.png', dpi: int = 1000,
            from_record: bool = False, record_path: str = None
    ) -> list[GridResult] | GridResult:
        """
        网格搜索
        :param problem_path: 训练集文件路径
        :param n_fold: n折交叉验证
        :param enable_logging: 是否打印搜索进度
        :param c_min: C的最小值
        :param c_max: C的最大值
        :param c_step: C的步长
        :param g_min: Gamma的最小值
        :param g_max: Gamma的最大值
        :param g_step: Gamma的步长
        :param detailed: 是否返回详细信息
        :param img_name: 图片名称
        :param dpi: 图片dpi
        :param from_record: 是否从记录中读取结果继续训练
        :param record_path: 记录文件路径
        """

        def _mul_range(start: float, end: float, step: float) -> list[float]:
            return [start * step ** i for i in range(int(log(end / start, step)) + 1)]

        def _draw_result():
            # 数据处理
            c_values = [log(result.c_min, 10) for result in self.__grid_results]
            g_values = [log(result.g_min, 10) for result in self.__grid_results]
            accuracy_values = [result.rate for result in self.__grid_results]
            # 绘图
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(c_values, g_values, accuracy_values, c=accuracy_values, cmap='viridis')
            ax.set_xlabel('lg(Cost)')
            ax.set_ylabel('lg(Gamma)')
            ax.set_zlabel('Accuracy')
            # 保存图片
            plt.savefig(img_name, dpi=dpi)
            plt.close()

        # 检查参数
        if not os.path.isfile(problem_path):
            raise FileNotFoundError('训练集文件不存在')
        if from_record:
            if record_path is None:
                raise ValueError('record_path参数不能为None')
            if not os.path.isfile(record_path):
                raise FileNotFoundError('记录文件不存在')
            try:
                self.__grid_results = [GridResult(**result) for result in load(record_path)]
            except Exception as error:
                raise ValueError('记录文件格式错误') from error
            if len(self.__grid_results) == 0:
                raise ValueError('记录文件为空')
            last_max_c_min = self.__grid_results[0].c_min
            last_max_g_min = self.__grid_results[0].g_min
            for r in self.__grid_results:
                if r.c_min > last_max_c_min:
                    last_max_c_min = r.c_min
                if r.g_min > last_max_g_min:
                    last_max_g_min = r.g_min
            c_range = _mul_range(last_max_c_min, c_max, c_step)
            g_range = _mul_range(g_min, g_max, g_step)
        else:
            self.__grid_results = []
            last_max_c_min = c_min
            last_max_g_min = g_min - 1
            c_range = _mul_range(c_min, c_max, c_step)
            g_range = _mul_range(g_min, g_max, g_step)
        # 计算总epoch数
        total_epochs = len(c_range) * len(g_range)
        hour_per_epoch = 0.25
        if enable_logging:
            print('total epochs:', total_epochs, 'epochs')
            print('expected time:', total_epochs * hour_per_epoch, 'hours')
        # 开始搜索
        start_time = time.time()
        for c in c_range:
            for g in g_range:
                if from_record and c == last_max_c_min and g <= last_max_g_min:
                    continue
                if enable_logging:
                    print('-' * 50)
                    print(f'epoch {len(self.__grid_results) + 1} / {total_epochs}\ncurrent c: {c}, current g: {g}\n'
                          f'current time: {time.time() - start_time} seconds\n'
                          f'current hour per epoch: {hour_per_epoch} hours\n'
                          f'remaining time: {(total_epochs - len(self.__grid_results) - 1) * hour_per_epoch} hours')
                accuracy = self.train(problem_path, n_fold=n_fold, gamma=g, cost=c)
                self.__grid_results.append(
                    GridResult(c_min=c, c_max=c * c_step, g_min=g, g_max=g * g_step, rate=accuracy)
                )
                if enable_logging:
                    current = time.time()
                    print(f'epoch {len(self.__grid_results)} finished, time elapsed: {current - start_time} seconds')
                    hour_per_epoch = (current - start_time) / len(self.__grid_results) / 3600
        # 结束搜索
        if enable_logging:
            print('-' * 50)
            print('grid search finished')
            print('total time elapsed:', time.time() - start_time, 'seconds')
            print('total hour per epoch:', (time.time() - start_time) / len(self.__grid_results) / 3600, 'hours')
        if detailed:
            if enable_logging:
                print('drawing result...')
            _draw_result()
            if enable_logging:
                print('drawing finished')
            return self.__grid_results
        else:
            return max(self.__grid_results, key=lambda x: x.rate)
