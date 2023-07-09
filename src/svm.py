import time
from math import log
from enum import Enum
from libsvm.svmutil import *
import matplotlib.pyplot as plt
from src.models import GridResult


class SymType(Enum):
    C_SVC = 0
    NU_SVC = 1
    ONE_CLASS_SVM = 2
    EPSILON_SVR = 3
    NU_SVR = 4


class KernelType(Enum):
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

    def __init__(self, problem_path: str = None, model_path: str = None):
        """
        :param problem_path: 训练集文件路径
        :param model_path: 模型文件路径
        """
        self.__problem_path: str | None = None
        self.problem_path = problem_path
        self.model_path: str = model_path

    @property
    def problem_path(self):
        """训练集文件路径"""
        return self.__problem_path

    @problem_path.setter
    def problem_path(self, problem_path: str):
        """训练集文件路径必须实际存在"""
        if os.path.exists(problem_path):
            self.__problem_path = problem_path
        else:
            raise FileNotFoundError(f'文件{problem_path}不存在')

    def grid(
            self,
            problem_path: str = None, n_fold: int = 5, enable_logging: bool = False,
            c_min: float = 1e-8, c_max: float = 1e8, c_step: float = 10,
            g_min: float = 1e-8, g_max: float = 1e8, g_step: float = 10,
            detailed: bool = False, img_name: str = 'grid_result.png', dpi: int = 1000
    ) -> list[GridResult] | GridResult:
        """
        网格搜索
        :param problem_path: 训练集文件路径
        :param n_fold: n折交叉验证
        :param enable_logging: 是否启用运行信息记录
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

        def _mul_range(start: float, end: float, step: float) -> list[float]:
            return [start * step ** i for i in range(int(log(end / start, step)) + 1)]

        def _draw_result():
            # 数据处理
            c_values = [log(result.c_min, 10) for result in results]
            g_values = [log(result.g_min, 10) for result in results]
            accuracy_values = [result.rate for result in results]
            # 绘图
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(c_values, g_values, accuracy_values, c=accuracy_values, cmap='viridis')
            ax.set_xlabel('lg(Cost)')
            ax.set_ylabel('lg(Gamma)')
            ax.set_zlabel('Accuracy')
            # 保存图片
            plt.savefig(img_name, dpi=dpi)

        results = []
        c_range = _mul_range(c_min, c_max, c_step)
        g_range = _mul_range(g_min, g_max, g_step)
        total_epochs = len(c_range) * len(g_range)
        hour_per_epoch = 0.25
        if enable_logging:
            print('total epochs:', total_epochs, 'epochs')
            print('expected time:', total_epochs * hour_per_epoch, 'hours')
        start_time = time.time()
        for c in c_range:
            for g in g_range:
                if enable_logging:
                    print('-' * 50)
                    print(f'epoch {len(results) + 1} / {total_epochs}')
                    print(f'current c: {c}, current g: {g}')
                    print(f'current time: {time.time() - start_time} seconds')
                    print(f'current hour per epoch: {hour_per_epoch} hours')
                    print(f'remaining time: {(total_epochs - len(results) - 1) * hour_per_epoch} hours')
                accuracy = self.train(
                    self.__problem_path if problem_path is None else problem_path,
                    n_fold=n_fold, gamma=g, cost=c
                )
                results.append(GridResult(c_min=c, c_max=c * c_step, g_min=g, g_max=g * g_step, rate=accuracy))
                if enable_logging:
                    current = time.time()
                    print(f'epoch {len(results)} finished, time elapsed: {current - start_time} seconds')
                    hour_per_epoch = (current - start_time) / len(results) / 3600
        if enable_logging:
            print('-' * 50)
            print('grid search finished')
            print('total time elapsed:', time.time() - start_time, 'seconds')
            print('total hour per epoch:', (time.time() - start_time) / len(results) / 3600, 'hours')
        if detailed:
            if enable_logging:
                print('drawing result...')
            _draw_result()
            if enable_logging:
                print('drawing finished')
            return results
        else:
            return max(results, key=lambda x: x.rate)

    def train(
            self,
            problem_path: str = None,
            model_path: str = None,
            sym_type: SymType = None,
            kernel_type: KernelType = None,
            degree: int = None,
            gamma: float = None,
            coef0: float = None,
            cost: float = None,
            nu: float = None,
            epsilon: float = None,
            cache_size: float = None,
            tolerance: float = None,
            shrinking: int = None,
            probability_estimates: int = None,
            weight: float = None,
            n_fold: int = None,
    ) -> float:
        """
        使用libsvm训练svm模型
        :param problem_path: libsvm可以读取的文件路径
        :param model_path: 模型保存路径
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
        # 生成参数字符串
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('problem_path')
        kwargs.pop('model_path')
        param_str = ''
        for key, value in kwargs.items():
            if value is not None:
                param_str += f'{SVM.__svm_train_options[key]} {value} '

        # 读取数据，训练模型
        prob = svm_read_problem(self.__problem_path if problem_path is None else problem_path)
        if param_str == '':
            model = svm_train(*prob)
        else:
            model = svm_train(*prob, param_str)
        # 保存模型
        if n_fold is None:
            svm_save_model(self.model_path if model_path is None else model_path + param_str + '.model', model)
        return model

    def predict(
            self,
            problem_path: str = None,
            model: float = None,
            model_path: str = None,
            probability_estimates: int = None,
    ) -> tuple[list, tuple[float, float, float], list]:
        """
        使用libsvm预测
        :param problem_path: libsvm可以读取的文件路径
        :param model: 训练好的模型
        :param model_path: 训练好的模型保存路径
        :param probability_estimates: whether to predict probability estimates, 0 or 1 (default 0)
        :return: 预测结果
        """
        # 检查参数
        if self.model_path is None and model is None and model_path is None:
            raise ValueError('model_path和model不能同时为None')
        if model_path and not os.path.exists(model_path):
            raise FileNotFoundError('模型文件不存在')
        # 生成参数字符串
        param_str = ''
        if probability_estimates is not None:
            param_str += '-b ' + str(probability_estimates)
        # 测试模型
        y, x = svm_read_problem(self.__problem_path if problem_path is None else problem_path)
        if model is None:
            model = svm_load_model(self.model_path if model_path is None else model_path)
        return svm_predict(y, x, model, param_str)
