from pydantic import BaseModel, Field


class Review(BaseModel):
    review: str = Field(..., title='评论内容')
    sentiment: bool = Field(..., title='情感标签')


class GridResult(BaseModel):
    c_min: float = Field(..., title='惩罚系数')
    c_max: float = Field(..., title='惩罚系数乘以步进')
    g_min: float = Field(..., title='核函数参数')
    g_max: float = Field(..., title='核函数参数乘以步进')
    rate: float = Field(..., title='准确率')


class NNTrainingState(BaseModel):
    current_epoch: int = Field(0, title='当前训练轮次')
    total_epoch: int = Field(0, title='总训练轮次')
    model_state_dict: dict = Field({}, title='模型参数')
    optimizer_state_dict: dict = Field({}, title='优化器参数')
