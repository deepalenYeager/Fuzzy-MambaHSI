import warnings
import torch
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def head_loss(loss_func,logits,label,align_corners=True):
    seg_logits = resize(
        input=logits,
        size=label.shape[1:],
        mode='bilinear',
        align_corners=align_corners)

    loss = loss_func(seg_logits,label)
    return loss


def diversity_loss(centers_list, tau=0.1, eps=1e-8):
    """Hinge-style, scale-normalized rule-center separation penalty (Eq. 22).

    centers_list: iterable of tensors shaped [R, d]. Tensors with R < 2 are skipped.
    """
    terms = []
    for M in centers_list:
        if M is None or M.dim() < 2 or M.size(0) < 2:
            continue
        norms = M.norm(dim=-1)
        diff = M.unsqueeze(0) - M.unsqueeze(1)
        pair_dist = diff.norm(dim=-1)
        pair_scale = norms.unsqueeze(0) + norms.unsqueeze(1) + eps
        rel = pair_dist / pair_scale
        R = M.size(0)
        mask = 1.0 - torch.eye(R, device=M.device, dtype=M.dtype)
        hinge = torch.clamp(tau - rel, min=0.0) ** 2
        terms.append((hinge * mask).sum() / (R * (R - 1)))
    if not terms:
        return torch.zeros((), device=centers_list[0].device if centers_list else 'cpu')
    return torch.stack(terms).mean()