from pydantic import BaseModel, Field


class GridResult(BaseModel):
    c: float = Field(..., title='惩罚系数')
    g: float = Field(..., title='核函数参数')
    accuracy: float = Field(..., title='准确率')


class SVMTrainingState(BaseModel):
    current_range: tuple[tuple[float, float], ...] = Field(tuple(), title='当前训练范围')
    results: tuple[GridResult] = Field([], title='网格搜索结果')


class NNTrainingState(BaseModel):
    current_epoch: int = Field(0, title='当前训练轮次')
    total_epoch: int = Field(0, title='总训练轮次')
    model_state_dict: dict = Field({}, title='模型参数')
    optimizer_state_dict: dict = Field({}, title='优化器参数')


a = SVMTrainingState(**{'current_range': ((1, 2,), (2, 3,)),
                        'results': [GridResult(**{'c_min': 1, 'c_max': 2, 'g_min': 3, 'g_max': 4, 'rate': 5})]})
