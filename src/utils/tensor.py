import torch
from torch import nn, Tensor


def randn_tensor_within_norm(*size: int, norm: Tensor, p: float = 2) -> Tensor:
    """
    生成一个范数不超过norm的随机张量

    Args:
        *size: 张量形状

    Keyword Args:
        norm: 张量范数
        p: 可选，计算张量范数的p值
    """
    pdist = nn.PairwiseDistance(p=p)
    random_tensor = torch.randn(*size)
    return random_tensor / pdist(random_tensor, torch.zeros(*size)) * norm


def random_tensors_outside_existed_tensors(
        *tensor: Tensor,
        num: int = 1, p: float = 2,
        dist_ratio: float = 2, rand_ratio: float = 0.5
) -> Tensor:
    """
    生成一个或多个与已有张量距离足够远的张量

    Notes:
        dist_ratio - rand_ratio ≤ random_tensor / average_tensor ≤ dist_ratio + rand_ratio\n
        min_radius为一个能包裹所有已有张量的最小半径\n
        average_tensor为所有已有张量的平均张量

    Args:
        *tensor: 已有张量

    Keyword Args:
        num: 可选，生成张量的个数，当num为1时返回一个张量，否则返回一个堆叠张量
        p: 可选，计算张量范数的p值
        dist_ratio: 可选，生成张量的生成原点距离已有张量的平均值与已有张量的最大距离与已有张量的平均值的距离的比值
        rand_ratio: 可选，生成张量的随机部分与已有张量的最大距离的比值

    Returns:
        生成的张量，当num为1时返回一个张量，否则返回一个堆叠张量，形状为[num, *tensor.shape]
    """
    # 计算平均张量
    stack_tensor = torch.stack(tensor)
    average_tensor = torch.mean(stack_tensor, dim=0)
    # 计算离平均值最远的张量
    pdist = nn.PairwiseDistance(p=p)
    min_radius, indice = torch.max(torch.stack([pdist(tensor, average_tensor) for tensor in stack_tensor]), dim=0)
    farthest: Tensor = stack_tensor[indice]
    # 计算生成张量的原点，返回的随机张量与该原点的范数不超过min_radius * rand_ratio
    unit_tensor: Tensor = (farthest - average_tensor) / min_radius
    generate_origin: Tensor = average_tensor + min_radius * dist_ratio * unit_tensor
    # 生成随机张量
    return_tensors = []
    norm = min_radius * rand_ratio
    for _ in range(num):
        random_direction = randn_tensor_within_norm(farthest.shape, norm=norm, p=p)
        return_tensors.append(generate_origin + random_direction)
    if num == 1:
        return return_tensors[0]
    return torch.stack(return_tensors)


def pad_tensor_with_tensor(tensor: Tensor, pad_tensor: Tensor, length: int, dim: int = 0) -> Tensor:
    """
    使用指定张量填充张量

    Args:
        tensor: 被填充的张量
        pad_tensor: 用于填充的张量
        length: 需填充的长度
        dim: 可选，填充的维度

    Returns:
        填充后的张量
    """
    return torch.cat([tensor, pad_tensor.repeat(length, 1)], dim=dim)
