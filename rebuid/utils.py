import torch


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm'. i.e. downsample by keeping every 'm'th value.

    Parameters:
        tensor (Tensor): tensor to be decimated
        m (list): list fo decimation factor for each dimension. set None if not to be decimated along a dimension

    Returns:
        tensor (Tensor): tensor decimated
    """

    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = torch.index_select(
                tensor,
                dim=d,
                index=torch.arange(
                    start=0, end=tensor.size(d), step=m[d]
                ).long(),
            )

    return tensor
