import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def normalize_logit(logit):
    """
    Z-score logit standardization from LSKD
    Args:
        logit: input logit tensor
    Returns:
        standardized logit tensor (temperature applied later in softmax)
    """
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def dkd_loss_with_lskd(logits_student, logits_teacher, target, alpha, beta, temperature):
    """
    DKD loss with LSKD logit standardization (always applied)
    Following the reference implementation exactly
    """
    # Apply Z-score standardization (no temperature scaling here)
    logits_student = normalize_logit(logits_student)
    logits_teacher = normalize_logit(logits_teacher)
    
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    # Apply temperature in softmax (consistent with reference)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    
    # TCKD loss with temperature**2 scaling (consistent with reference)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    
    # NCKD loss with temperature scaling (consistent with reference)
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    
    return alpha * tckd_loss + beta * nckd_loss


def dkd_loss_per_sample_lskd(logits_student, logits_teacher, target, alpha, beta, temperature):
    """
    DKD loss with LSKD logit standardization - returns per-sample losses for SDD weighting
    Following the reference implementation exactly
    """
    # Apply Z-score standardization (no temperature scaling here)
    logits_student = normalize_logit(logits_student)
    logits_teacher = normalize_logit(logits_teacher)
    
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    # Apply temperature in softmax (consistent with reference)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    
    # TCKD loss per sample
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(dim=1)
        * (temperature**2)
    )
    
    # NCKD loss per sample
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none').sum(dim=1)
        * (temperature**2)
    )
    
    return alpha * tckd_loss + beta * nckd_loss


def multi_scale_distillation_with_lskd(out_s_multi, out_t_multi, target, alpha, beta, temperature):
    """
    Multi-scale distillation from SDD combined with LSKD logit standardization
    Args:
        out_s_multi: student multi-scale logits [B, C, N]
        out_t_multi: teacher multi-scale logits [B, C, N] 
        target: ground truth labels
        alpha, beta: DKD loss weights
        temperature: base temperature (LSKD standardization always applied)
    """
    # Convert shape from B x C x N to N*B x C
    out_s_multi_t = out_s_multi.permute(2, 0, 1)
    out_t_multi_t = out_t_multi.permute(2, 0, 1)

    out_t = torch.reshape(out_t_multi_t, (out_t_multi_t.shape[0] * out_t_multi_t.shape[1], out_t_multi_t.shape[2]))
    out_s = torch.reshape(out_s_multi_t, (out_s_multi_t.shape[0] * out_s_multi_t.shape[1], out_s_multi_t.shape[2]))
    target_r = target.repeat(out_t_multi.shape[2])

    # Calculate distillation loss with LSKD standardization - per sample for SDD weighting
    loss = dkd_loss_per_sample_lskd(out_s, out_t, target_r, alpha, beta, temperature)

    # Find complementary and consistent local distillation loss (SDD mechanism)
    out_t_predict = torch.argmax(out_t, dim=1)
    mask_true = out_t_predict == target_r
    mask_false = out_t_predict != target_r

    # Global prediction (first batch corresponds to global scale)
    global_prediction = out_t_predict[0:len(target)]
    global_prediction_true_mask = global_prediction == target
    global_prediction_false_mask = global_prediction != target

    global_prediction_true_mask_repeat = torch.tensor(global_prediction_true_mask).repeat(out_t_multi.shape[2])
    global_prediction_false_mask_repeat = torch.tensor(global_prediction_false_mask).repeat(out_t_multi.shape[2])

    # Global true, local wrong
    mask_false[global_prediction_false_mask_repeat] = False
    mask_false[0:len(target)] = False
    gt_lw = mask_false

    # Global wrong, local true
    mask_true[global_prediction_true_mask_repeat] = False
    mask_true[0:len(target)] = False
    gw_lt = mask_true

    # Reset masks
    mask_false = out_t_predict != target_r
    mask_true = out_t_predict == target_r

    # Global wrong, local wrong
    mask_false[global_prediction_true_mask_repeat] = False
    gw_lw = mask_false

    # Global true, local true
    mask_true[global_prediction_false_mask_repeat] = False
    gt_lt = mask_true

    assert torch.sum(gt_lt) + torch.sum(gw_lw) + torch.sum(gt_lw) + torch.sum(gw_lt) == target_r.shape[0]

    # Apply SDD weighting scheme for complementary terms
    index = torch.zeros_like(loss).float()
    index[gw_lw] = 1.0  # Global wrong, local wrong - consistent
    index[gt_lt] = 1.0  # Global true, local true - consistent  
    index[gw_lt] = 2.0  # Global wrong, local true - complementary
    index[gt_lw] = 2.0  # Global true, local wrong - complementary

    loss = torch.sum(loss * index)

    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: NaN or Inf loss detected, setting to zero")
        loss = torch.zeros(1).float().to(loss.device)

    return loss


class SDD_DKD_LSKD(Distiller):
    """
    SDD-DKD-LSKD Fusion: Scale Decoupled Distillation + DKD + Logit Standardization
    
    This class combines:
    - SDD: Multi-scale logit decoupling with consistent/complementary knowledge weighting
    - DKD: Decoupled Knowledge Distillation (TCKD + NCKD)  
    - LSKD: Z-score logit standardization to focus on logit relations rather than magnitude
    
    Note: LSKD standardization is always applied in this implementation.
    """

    def __init__(self, student, teacher, cfg):
        super(SDD_DKD_LSKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP
        self.M = getattr(cfg, 'M', 4)
        
    def forward_train(self, image, target, **kwargs):
        logits_student, patch_s = self.student(image)
        with torch.no_grad():
            logits_teacher, patch_t = self.teacher(image)

        # Standard cross-entropy loss
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # Choose between global distillation and multi-scale distillation
        if self.M == '[1]':
            # Global distillation with LSKD standardization
            loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss_with_lskd(
                logits_student,
                logits_teacher,
                target,
                self.alpha,
                self.beta,
                self.temperature,
            )
        else:
            # Multi-scale distillation with SDD + LSKD fusion
            loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * multi_scale_distillation_with_lskd(
                patch_s,
                patch_t,
                target,
                self.alpha,
                self.beta,
                self.temperature,
            )
            
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
