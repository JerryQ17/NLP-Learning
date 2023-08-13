import os
import sys
import time
import logging
from math import log
from torch import load
from enum import IntEnum
import matplotlib.pyplot as plt
from libsvm.svmutil import (
    svm_model,
    svm_train, svm_predict,
    svm_save_model, svm_load_model,
    svm_read_problem
)

from src.utils import tools
from src.utils.models import GridResult, SVMTrainingState


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


class SVM:
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

    def __init__(self, problem_path: str = None, model_savepath: str = None,
                 logger: logging.Logger = logging.getLogger(__name__)):
        self.__logger: logging.Logger | None = None
        self.logger = logger

        self.__model: svm_model | None = None

        self.__problem_path: str | None = None
        self.problem_path = problem_path

        self.__model_savepath: str | None = None
        self.model_savepath = model_savepath

        self.__state: SVMTrainingState = SVMTrainingState()

    @property
    def logger(self):
        return self.__logger

    @logger.setter
    def logger(self, logger):
        self.__logger = tools.TypeCheck(logging.Logger)(logger, default=logging.getLogger(__name__))

    @property
    def model(self):
        return self.__model

    @property
    def problem_path(self) -> str:
        return self.__problem_path

    @problem_path.setter
    def problem_path(self, path: str):
        self.__problem_path = tools.check_file(path)

    @property
    def model_savepath(self) -> str:
        return self.__model_savepath

    @model_savepath.setter
    def model_savepath(self, path: str):
        self.__model_savepath = tools.check_str(path)

    @property
    def state(self):
        return self.__state

    def load(self, model: svm_model = None, model_path: str = None) -> 'SVM':
        """加载模型，model和model_path必须至少指定一个"""
        if model is not None:
            self.__model = tools.TypeCheck(svm_model)(model)
        elif model_path is not None:
            self.__model = svm_load_model(tools.check_file(model_path))
        else:
            raise ValueError('model和model_path必须至少指定一个')
        return self

    def save(self, path: str = None) -> str:
        """保存模型"""
        if path is None:
            if self.__model_savepath is None:
                raise ValueError('未指定模型保存路径')
            path = self.__model_savepath
        else:
            path = tools.check_str(path)
        svm_save_model(path, self.__model)
        return os.path.abspath(path)

    def train(
            self, problem_path: str = None,
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
        tools.check_file(problem_path)
        tools.TypeCheck(int)(shrinking, probability_estimates, include_none=True,
                             extra_checks=[(lambda x: x in (0, 1), ValueError('shrinking必须是0或1'))])
        # 生成参数字符串
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('problem_path')
        param_str = ''
        for key, value in kwargs.items():
            if value is not None:
                param_str += f'{self.__svm_train_options[key]} {value} '
        # 读取数据
        if problem_path is None:
            if self.__problem_path is None:
                raise ValueError('未指定训练集路径')
            problem_path = self.__problem_path
        else:
            if not isinstance(problem_path, str):
                raise TypeError('problem_path必须是str类型')
            if not os.path.isfile(problem_path):
                raise FileNotFoundError(f'文件{problem_path}不存在')
        problem = svm_read_problem(problem_path)
        # 训练模型
        if param_str == '':
            result = svm_train(*problem)
        else:
            result = svm_train(*problem, param_str)
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
                model = svm_load_model(tools.check_file(model_path))
            else:
                if model_path is not None and os.path.isfile(model_path):
                    print('Warning: 因为model和model_path同时存在，model_path参数已被忽略', file=sys.stderr)
        else:
            model = self.__model
            if model is not None:
                self.__logger.warning('因为已加载模型，model参数已被忽略')
            if model_path is not None:
                self.__logger.warning('因为已加载模型，model_path参数已被忽略')
        tools.TypeCheck(int)(probability_estimates, include_none=True,
                             extra_checks=[(lambda i: i in (0, 1), ValueError('probability_estimates必须是0或1'))])

        # 生成参数字符串
        param_str = ''
        if probability_estimates is not None:
            param_str = '-b ' + str(probability_estimates)
        # 测试模型
        y, x = svm_read_problem(problem_path)
        return svm_predict(y, x, model, param_str)

    @staticmethod
    def __mul_range(start: float, end: float, step: float) -> tuple[float]:
        return tuple(start * step ** i for i in range(int(log(end / start, step)) + 1))

    def __draw_result(self, img_name: str = None, dpi: int = None):
        tools.check_str(img_name, default=r'.\svm\data\grid_result.png')
        tools.TypeCheck(int)(dpi, default=1000)
        self.__logger.info('drawing result...')
        # 数据处理
        c_values = [log(result.c, 10) for result in self.__state.results]
        g_values = [log(result.g, 10) for result in self.__state.results]
        accuracy_values = [result.accuracy for result in self.__state.results]
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
        self.__logger.info('drawing finished, saved to %s', img_name)

    def __single_grid(self, c: float, g: float, problem_path: str, n_fold: int):
        start_time = time.time()
        self.__logger.info(f'epoch {len(self.__state.results) + 1}\ncurrent c: {c}, current g: {g}')
        accuracy = self.train(problem_path, n_fold=n_fold, gamma=g, cost=c)
        self.__state.results = tuple([_ for _ in self.__state.results] + [GridResult(c=c, g=g, accuracy=accuracy)])

        cost = time.time() - start_time
        self.__logger.info(
            f'epoch {len(self.__state.results)} finished, time cost: {cost} seconds'
        )

    def grid(
            self, problem_path: str = None, n_fold: int = 5,
            c_min: float = 1e-8, c_max: float = 1e8, c_step: float = 10,
            g_min: float = 1e-8, g_max: float = 1e8, g_step: float = 10,
            detailed: bool = False, img_name: str = r'.\svm\data\grid_result.png', dpi: int = 1000
    ) -> tuple[GridResult] | GridResult:
        """
        网格搜索
        :param problem_path: 训练集文件路径
        :param n_fold: n折交叉验证
        :param c_min: C的最小值
        :param c_max: C的最大值
        :param c_step: C的步长
        :param g_min: Gamma的最小值
        :param g_max: Gamma的最大值
        :param g_step: Gamma的步长
        :param detailed: 是否返回详细信息
        :param img_name: 图片名称
        :param dpi: 图片dpi
        """

        # 检查参数
        if problem_path is None:
            if self.__problem_path is None:
                raise ValueError('problem_path参数不能为None')
            problem_path = self.__problem_path
        else:
            tools.check_file(problem_path)

        self.__state = SVMTrainingState()
        c_range = self.__mul_range(c_min, c_max, c_step)
        g_range = self.__mul_range(g_min, g_max, g_step)

        # 计算总epoch数
        total_epochs = len(c_range) * len(g_range)
        hour_per_epoch = 0.25
        self.__logger.info(f'total epochs: {total_epochs} epochs\n'
                           f'expected time: {total_epochs * hour_per_epoch} hours')
        # 开始搜索
        start_time = time.time()
        for c in c_range:
            for g in g_range:
                self.__single_grid(c, g, problem_path, n_fold)

        # 结束搜索
        self.__logger.info(
            'grid search finished'
            f'total time elapsed: {time.time() - start_time} seconds'
            f'total hour per epoch: {(time.time() - start_time) / len(self.__state.results) / 3600} hours'
        )
        if detailed:
            self.__draw_result(img_name, dpi)
            return self.__state.results
        else:
            return max(self.__state.results, key=lambda x: x.accuracy)

    def grid_from_state(
            self, state_path: str, problem_path: str = None, n_fold: int = 5,
            detailed: bool = False, img_name: str = r'.\svm\data\grid_result.png', dpi: int = 1000
    ) -> tuple[GridResult] | GridResult:
        tools.check_file(state_path)
        try:
            self.__state = SVMTrainingState(**load(state_path))
        except Exception as error:
            raise RuntimeError('读取文件时发生错误') from error
        if len(self.__state.results) == 0:
            raise ValueError('记录文件为空')
        for c, g in self.__state.current_range:
            self.__single_grid(c, g, problem_path, n_fold)

        if detailed:
            self.__draw_result(img_name, dpi)
            return self.__state.results
        else:
            return max(self.__state.results, key=lambda x: x.accuracy)
