import torch


def compute_pos_weight(df, label_cols, device=None):
    """Computes BCEWithLogitsLoss's per-label pos_weight: num_negative / num_positive for
    each label, from df's actual class counts. This is PyTorch's documented closed-form
    correction for class imbalance -- it scales the gradient of each label's POSITIVE
    examples by that label's own ratio (negatives untouched), so a rare label's few
    positives push as hard in aggregate as its many negatives.

    Note each label's ratio is computed independently, down its own column of the label
    matrix -- "negative" means that one disease is absent (which includes images carrying
    other diseases), NOT "the other labels". This matches the multi-label setup, where the
    model emits one independent logit per label rather than one distribution over labels.

    Returns a 1-D tensor of len(label_cols), ordered to match label_cols -- which must be
    the same order used to build the model's output logits."""
    pos_counts = torch.tensor(df[label_cols].sum(axis=0).values, dtype=torch.float32)
    neg_counts = len(df) - pos_counts
    pos_weight = neg_counts / pos_counts.clamp(min=1.0)   # clamp guards a 0-positive label
    return pos_weight.to(device) if device is not None else pos_weight


class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg                     = gamma_neg
        self.gamma_pos                     = gamma_pos
        self.clip                          = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps                           = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos    = x_sigmoid
        xs_neg    = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss    = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0             = xs_pos * y
            pt1             = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt              = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w     = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()