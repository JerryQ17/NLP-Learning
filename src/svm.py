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


class SVM:
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
        pass

    @staticmethod
    def grid(
            problem_path: str, n_fold: int = 5,
            c_min: float = 1e-8, c_max: float = 1e8, c_step: float = 10,
            g_min: float = 1e-8, g_max: float = 1e8, g_step: float = 10,
            detailed: bool = False, img_name: str = 'grid_result.png', dpi: int = 1000
    ) -> list[GridResult] | GridResult:

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
        print('total epochs:', total_epochs, 'epochs')
        print('expected time:', total_epochs / 4, 'hours')
        for c in c_range:
            for g in g_range:
                print(f'epoch {len(results) + 1} / {total_epochs}')
                ac = SVM.train(problem_path, n_fold=n_fold, gamma=g, cost=c)
                results.append(GridResult(c_min=c_min, c_max=c_max, g_min=g_min, g_max=g_max, rate=ac))
        if detailed:
            _draw_result()
            return results
        else:
            return max(results, key=lambda x: x.rate)

    @staticmethod
    def train(
            problem_path: str,
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
        kwargs.pop('problem_path')
        kwargs.pop('model_path')
        param_str = ''
        for key, value in kwargs.items():
            if value is not None:
                param_str += f'{SVM.__svm_train_options[key]} {value} '

        # 读取数据，训练模型
        prob = svm_read_problem(problem_path)
        if param_str == '':
            model = svm_train(*prob)
        else:
            model = svm_train(*prob, param_str)
        # 保存模型
        if n_fold is None:
            svm_save_model(model_path + param_str + '.model', model)
        return model

    @staticmethod
    def svm_predict(
            problem_path: str,
            model: float = None,
            model_path: str = None,
            probability_estimates: int = None,
    ) -> tuple[list, tuple[float, float, float], list]:
        """
        使用libsvm预测
        :return: None
        """
        # 检查参数
        if model is None and model_path is None:
            raise ValueError('model和model_path不能同时为None')
        # 生成参数字符串
        param_str = ''
        if probability_estimates is not None:
            param_str += '-b ' + str(probability_estimates)
        # 测试模型
        y, x = svm_read_problem(problem_path)
        if model is None:
            model = svm_load_model(model_path)
        return svm_predict(y, x, model, param_str)
