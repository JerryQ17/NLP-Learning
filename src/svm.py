import time
from math import log
from enum import IntEnum
from libsvm.svmutil import *
import matplotlib.pyplot as plt
from src.models import GridResult


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
        self.__model = None

    def load(self, model_path: str):
        """加载模型"""
        self.__model = svm_load_model(model_path)

    def save(self, model_path: str):
        """保存模型"""
        svm_save_model(model_path, self.__model)

    def train(
            self, problem_path: str,
            sym_type: SymType = None, kernel_type: KernelType = None,
            degree: int = None, gamma: float = None, coef0: float = None, cost: float = None,
            nu: float = None, epsilon: float = None, cache_size: float = None, tolerance: float = None,
            shrinking: int = None, probability_estimates: int = None, weight: float = None, n_fold: int = None,
    ) -> float:
        # 生成参数字符串
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('problem_path')
        kwargs.pop('model_path')
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
            model=None,
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

        # 生成参数字符串
        param_str = ''
        if probability_estimates is not None:
            param_str = '-b ' + str(probability_estimates)
        # 测试模型
        y, x = svm_read_problem(problem_path)
        return svm_predict(y, x, model, param_str)

    def grid(
            self,
            problem_path: str = None, n_fold: int = 5, enable_logging: bool = False,
            c_min: float = 1e-8, c_max: float = 1e8, c_step: float = 10,
            g_min: float = 1e-8, g_max: float = 1e8, g_step: float = 10,
            detailed: bool = False, img_name: str = r'..\svm\train\grid_result.png', dpi: int = 1000
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
                    print(f'epoch {len(results) + 1} / {total_epochs}\ncurrent c: {c}, current g: {g}\n'
                          f'current time: {time.time() - start_time} seconds\n'
                          f'current hour per epoch: {hour_per_epoch} hours\n'
                          f'remaining time: {(total_epochs - len(results) - 1) * hour_per_epoch} hours')
                accuracy = self.train(problem_path, n_fold=n_fold, gamma=g, cost=c)
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
