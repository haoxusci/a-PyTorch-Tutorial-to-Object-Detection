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


def cxcy_toxy(cxcy):
    """Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to (x_min, y_min, x_max, y_max)"""
    return torch.cat(
        [cxcy[:, :2] - cxcy[:, 2:] / 2, cxcy[:, :2] + cxcy[:, 2:] / 2], dim=1
    )


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decoded the predicted gcxgcy bounding box to center-size coordinates

    Paramters:
    gcxgcy (Tensor): gcxgcy coordinates. (8732, 4)
    priors_cxcy (Tensor): center-size priors' coordinates. (8732, 4)

    Returns:
    Tensor: (8732, 4)
    """
    # The 10 and 5 below are referred to as 'variances' in the original caffe repo, completely empirical
    # They are for tsome sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    # ??hao?? I dont get it. or totally meaningless, more likely, a variation of definiation of gcx, gcy (10), gw, gh (5)
    return torch.cat(
        [
            gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],
            torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:],
        ],
        dim=1,
    )


def find_intersection(set_1, set_2):
    """
    find the overlpa

    Parameters:
    set_1 (tensor): a tensor of dimensions (n1, 4)
    set_2 (tensor): a tensor of dimensions (n2, 4)

    Returns:
    (tensor): Jaccard Overlap of each of the bxoes in set 1 w.r.t. each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # Pytorch auto-broadcasts
    lower_bounds = torch.max(
        set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)
    )  # (n1, n2, 2)
    upper_bounds = torch.min(
        set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0)
    )  # (n1, n2, 2)

    intersection_dims = torch.clamp(
        upper_bounds - lower_bounds, min=0
    )  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """Find the Jaccard Overlp (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    Parameters:
    set_1 (tensor): a tensor of dimensions (n1, 4)
    set_2 (tensor): a tensor of dimensions (n2, 4)

    Returns:
    (tensor): Jaccard Overlap of each of the bxoes in set 1 w.r.t. each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (
        set_1[:, 3] - set_1[:, 1]
    )  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (
        set_2[:, 3] - set_2[:, 1]
    )  # (n2)

    union = (
        areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection
    )  # (n1, n2)

    return intersection / union
