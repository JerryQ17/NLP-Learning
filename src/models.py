from pydantic import BaseModel, Field


class Review(BaseModel):
    review: str = Field(..., title='评论内容')
    sentiment: bool = Field(..., title='情感标签')
