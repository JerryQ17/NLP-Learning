from pydantic import BaseModel, Field


class Review(BaseModel):
    review: str = Field(..., title='评论内容')
    sentiment: bool = Field(..., title='情感标签')


class GridResult(BaseModel):
    c_min: float = Field(..., title='惩罚系数')
    c_max: float = Field(..., title='惩罚系数')
    g_min: float = Field(..., title='核函数参数')
    g_max: float = Field(..., title='核函数参数')
    rate: float = Field(..., title='准确率')
