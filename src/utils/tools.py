import os
import sys
from numpy import ndarray
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import TypeVar, Protocol, final, runtime_checkable
from collections.abc import Sized, Iterable, Callable, Hashable

if sys.version_info <= (3, 9):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec

__all__ = [
    # class
    'TypeCheck', 'StrictTypeCheck',
    # instance
    'check_str', 'check_iterable', 'check_callable', 'check_ndarray', 'check_dataset',
    # function
    'check_file', 'check_dir',
]

_T = TypeVar('_T')
_P = ParamSpec('_P')


def _copy_docstring_from(source_method: Callable[_P, _T]):
    """
    复制源函数的文档字符串到目标函数

    Args:
        source_method: 源函数

    Returns:
        装饰器，应用于目标函数
    """

    def decorator(target_method: Callable[_P, _T]) -> Callable[_P, _T]:
        target_method.__doc__ = source_method.__doc__
        return target_method

    return decorator


class _BaseTypeCheck(ABC):
    """
    类型检查基类，是一个抽象类，可被序列化

    Abstractmethods:
        _types_to_key: 该方法将类型检查器的检查类型转换为__instances字典的键\n
        __call__: 该方法将检查对象是否符合预期类型，如果不符合则引发TypeError

    Attributes:
        __TC: 一个类型变量，被绑定于此类，用于表示类型检查器的类型
        __instances: 字典，键为类型检查器的检查类型，值为类型检查器的实例，用于缓存类型检查器的实例

    Properties:
        types: 类型检查器的检查类型

    Notes:
        _BaseTypeCheck类维护了类属性__instances，该字典的键为类型检查器的检查类型，值为类型检查器的实例。
        创建实例时，如果检查类型已经存在于__instances字典中，则直接返回该类型检查器的实例，否则创建一个新的类型检查器实例并返回。

    Notes:
        可以通过加减法运算符对类型检查器进行组合。
    """

    @runtime_checkable
    class __K(Protocol, Hashable, Iterable):
        """
        键类型抽象类，用于作为__instances字典的键，必须是可哈希且可迭代的对象

        Notes:
            虽然字典的键只要求Hashable，但是由于只要求实现了_types_to_key方法，不要求实现_key_to_types方法，缺少了将键转换为类型的方法，
            而在TypeCheck和AllTypeCheck的实现中，用迭代器间接实现了_key_to_types方法，因此__K选择继承了Iterable类。
            如果自定义子类自行实现了_key_to_types方法，可以不继承Iterable类。但是需要重写(override)所有与__key相关的方法。
        """

    __TC = TypeVar('__TC', bound='_BaseTypeCheck')
    __instances: dict[__K, __TC] = {}

    @classmethod
    @abstractmethod
    def _types_to_key(cls, *types: type) -> __K:
        """
        将类型检查器的检查类型转换为__instances字典的键

        Args:
            types: 类型

        Returns:
            一个可哈希且可迭代的对象

        Notes:
            该方法的耗时不应过长，因为该方法初始化时会被调用两次。
        """

    def __new__(cls, *types: type) -> __TC:
        """
        创建类型检查器实例

        Args:
            types: 类型

        Returns:
            如果types相同，且实例类型相同，则返回同一个实例，否则创建一个新的实例。
        """
        key = cls._types_to_key(*types)
        if key not in cls.__instances:
            cls.__instances[key] = super().__new__(cls)
        return cls.__instances[key]

    def __init__(self, *types: type):
        """
        初始化类型检查器

        Args:
            types: 预期类型

        Notes:
            如果types为空，则传入__call__方法的任何obj都将引发TypeError，除非obj和include_none都为None
        """
        self.__key = self._types_to_key(*types)

    def __del__(self):
        self.__instances.pop(self.__key)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.__key})'

    def __str__(self):
        sep = ' | ' if isinstance(self, TypeCheck) else ' & ' if isinstance(self, StrictTypeCheck) else ','
        return 'TypeChecker for ' + sep.join(map(lambda x: x.__name__, self.__key))

    @abstractmethod
    def __call__(self, *obj: _T,
                 default: _T | None = None, include_none: bool = False,
                 extra_checks: Iterable[tuple[Callable[[_T], bool], Exception]] = None) -> tuple[_T | None] | _T | None:
        """
        检查obj是否符合预期类型并能通过额外检查，返回检查后的obj

        Args:
            *obj: 要检查的对象

        Keyword Args:
            default: 可选，当检查失败时返回的默认值，如果不为None，不符合预期类型或额外检查时不会抛出异常，而是返回default
            include_none: 可选，是否包含None，如果为True，obj为None时不进行所有检查，直接返回None
            extra_checks: 可选，额外检查，为一个可迭代对象，
                可迭代对象的每个元素为一个元组，元组的第一项是一个可调用对象，第二项是一个异常，
                可调用对象只有一个参数，为要检查的对象obj，检查通过时返回值的求值结果为True，检查失败返回值的求值结果为False，
                如果调用该可调用对象的求值结果为False，则根据default的值决定抛出元组第二项对应的异常或返回default

        Returns:
            当传入的obj数量大于一时，返回检查通过的对象元组
            当传入的obj数量等于一时，直接返回该对象
            当传入的obj数量等于零时，返回None

        Raises:
            TypeError: obj不符合预期类型
            Exception: 额外检查失败时抛出的异常

        Notes:
            检查顺序为：
                1. include_none为True时，检查obj是否为None
                2. obj是否符合预期类型，不改变obj的值
                3. extra_checks不为None时，进行额外检查，额外检查函数可能会改变obj的值

        Notes:
            当default不为None时，如果obj == default，不会进行类型检查和额外检查，直接返回obj

        Notes:
            返回的obj可能与传入的obj不同，取决于额外检查函数的实现
        """

    def __eq__(self: __TC, other: __TC):
        if isinstance(self, _BaseTypeCheck) and isinstance(other, _BaseTypeCheck):
            return self.__key == other.__key
        raise NotImplemented(f'Cannot compare {type(self)} with {type(other)}')

    def __lt__(self: __TC, other: __TC):
        if isinstance(other, self.__class__):
            return frozenset(self.types).issubset(frozenset(other.types))
        raise NotImplemented(f'Cannot compare {type(self)} with {type(other)}')

    def __le__(self: __TC, other: __TC):
        if isinstance(other, self.__class__):
            return frozenset(self.types).issubset(frozenset(other.types)) or self.types == other.types
        raise NotImplemented(f'Cannot compare {type(self)} with {type(other)}')

    def __gt__(self: __TC, other: __TC):
        if isinstance(other, self.__class__):
            return frozenset(self.types).issuperset(frozenset(other.types))
        raise NotImplemented(f'Cannot compare {type(self)} with {type(other)}')

    def __ge__(self: __TC, other: __TC):
        if isinstance(other, self.__class__):
            return frozenset(self.types).issuperset(frozenset(other.types)) or self.types == other.types
        raise NotImplemented(f'Cannot compare {type(self)} with {type(other)}')

    def __add__(self: __TC, other: __TC) -> __TC:
        if isinstance(other, self.__class__):
            return self.__class__(*set(self.types + other.types))
        raise NotImplemented(f'Cannot add {type(other)} to {type(self)}')

    def __sub__(self: __TC, other: __TC) -> __TC:
        if isinstance(other, self.__class__):
            return self.__class__(*(set(self.types) - set(other.types)))
        raise NotImplemented(f'Cannot subtract {type(other)} from {type(self)}')

    def __hash__(self):
        return hash(self.__key)

    def __reduce_ex__(self, protocol):
        return self.__class__, self.types

    @property
    def types(self) -> tuple[type]:
        """检查的类型元组"""
        return tuple(self.__key)

    @_copy_docstring_from(__call__)
    def check(self, *args, **kwargs):
        return self(*args, **kwargs)

    @staticmethod
    def _check_type(obj: type) -> type:
        if isinstance(obj, type):
            return obj
        raise TypeError(f'The parameter must be a type, got {obj} , did you mean {type(obj)}?')

    @staticmethod
    def _extra_checks(obj: _T, default: _T | None,
                      extra_checks: Iterable[tuple[Callable[[_T], bool], Exception]]) -> _T:
        if extra_checks:
            for check, exception in extra_checks:
                if not check(obj):
                    if default is not None:
                        return default
                    raise exception
        return obj


@final
class TypeCheck(_BaseTypeCheck):
    """类型检查，要求obj符合self.types中的任意一个类型，不能被继承"""

    @classmethod
    def _types_to_key(cls, *types: type) -> tuple:
        return tuple(sorted(set(types), key=lambda x: f'{x.__module__}.{x.__name__}'))

    def __call__(self, *obj: _T,
                 default: _T | None = None, include_none: bool = False,
                 extra_checks: Iterable[tuple[Callable[[_T], bool], Exception]] = None) -> tuple[_T | None] | _T | None:
        if len(obj) == 0:
            return None

        rev = []

        for o in obj:
            if default is not None and o == default:
                rev.append(o)
            elif include_none and o is None:
                rev.append(None)
            elif isinstance(o, self.types):
                rev.append(self._extra_checks(o, default, extra_checks))
            elif default is not None:
                rev.append(default)
            else:
                o_repr = repr(o)
                if len(o_repr) > 25:
                    o_repr = o_repr[:25] + '...'
                raise TypeError(f'The parameter type must be {self.types}, got {o_repr} , type {type(o)}')

        return tuple(rev) if len(rev) > 1 else rev[0]


@final
class StrictTypeCheck(_BaseTypeCheck):
    """严格类型检查，要求obj符合self.types中的所有类型，不能被继承"""

    @classmethod
    def _types_to_key(cls, *types: type) -> frozenset:
        return frozenset(map(cls._check_type, types))

    def __call__(self, *obj: _T,
                 default: _T | None = None, include_none: bool = False,
                 extra_checks: Iterable[tuple[Callable[[_T], bool], Exception]] = None) -> tuple[_T | None] | _T | None:
        objlen = len(obj)
        if objlen == 0:
            return None
        if default is not None and obj == tuple(default for _ in range(len(obj))):
            return obj
        for checker in map(TypeCheck, self.types):
            obj = checker(*obj, default=default, include_none=include_none)
            if objlen == 1:
                obj = (obj,)
        rev = tuple(map(lambda x: self._extra_checks(x, default, extra_checks), obj))
        return rev if objlen > 1 else rev[0]


check_str = TypeCheck(str)
check_iterable = TypeCheck(Iterable)
check_callable = TypeCheck(Callable)
check_ndarray = TypeCheck(ndarray)
check_dataset = StrictTypeCheck(Dataset, Sized)


def check_file(file_path: str, include_none: bool = False, default: str = None) -> str:
    """
    检查文件路径是否存在，如果存在，返回绝对路径，否则抛出FileNotFoundError

    Args:
        file_path: 文件路径
        include_none: 是否允许file_path为None
        default: 如果file_path不存在，返回的默认值

    Returns:
        file_path的绝对路径
    """
    return os.path.abspath(check_str(
        file_path, include_none=include_none, default=default,
        extra_checks=[(os.path.isfile, FileNotFoundError(f'file {file_path} not found'))]
    ))


def check_dir(dir_path: str, include_none: bool = False, default: str = None) -> str:
    """
    检查文件夹路径是否存在，如果存在，返回绝对路径，否则抛出FileNotFoundError

    Args:
        dir_path: 文件夹路径
        include_none: 是否允许dir_path为None
        default: 如果dir_path不存在，返回的默认值

    Returns:
        dir_path的绝对路径
    """
    return os.path.abspath(check_str(
        dir_path, include_none=include_none, default=default,
        extra_checks=[(os.path.isdir, FileNotFoundError(f'directory {dir_path} not found'))]
    ))
