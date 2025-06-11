import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


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


def kd_loss_with_lskd(logits_student, logits_teacher, temperature):
    """
    KD loss with LSKD logit standardization (always applied)
    Following the reference implementation exactly
    """
    # Apply Z-score standardization (no temperature scaling here)
    logits_student = normalize_logit(logits_student)
    logits_teacher = normalize_logit(logits_teacher)
    
    # Apply temperature in softmax (consistent with reference)
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    # KD loss with temperature**2 scaling (consistent with reference)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


def kd_loss_per_sample_lskd(logits_student, logits_teacher, temperature):
    """
    KD loss with LSKD logit standardization - returns per-sample losses for SDD weighting
    Following the reference implementation exactly
    """
    # Apply Z-score standardization (no temperature scaling here)
    logits_student = normalize_logit(logits_student)
    logits_teacher = normalize_logit(logits_teacher)
    
    # Apply temperature in softmax (consistent with reference)
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    # KD loss per sample with temperature**2 scaling
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd


def multi_scale_kd_distillation_with_lskd(out_s_multi, out_t_multi, target, temperature):
    """
    Multi-scale KD distillation from SDD combined with LSKD logit standardization
    Args:
        out_s_multi: student multi-scale logits [B, C, N]
        out_t_multi: teacher multi-scale logits [B, C, N] 
        target: ground truth labels
        temperature: base temperature (LSKD standardization always applied)
    """
    # Convert shape from B x C x N to N*B x C
    out_s_multi_t = out_s_multi.permute(2, 0, 1)
    out_t_multi_t = out_t_multi.permute(2, 0, 1)

    out_t = torch.reshape(out_t_multi_t, (out_t_multi_t.shape[0] * out_t_multi_t.shape[1], out_t_multi_t.shape[2]))
    out_s = torch.reshape(out_s_multi_t, (out_s_multi_t.shape[0] * out_s_multi_t.shape[1], out_s_multi_t.shape[2]))
    target_r = target.repeat(out_t_multi.shape[2])

    # Calculate distillation loss with LSKD standardization - per sample for SDD weighting
    loss = kd_loss_per_sample_lskd(out_s, out_t, temperature)

    # Find complementary and consistent local distillation loss (SDD mechanism)
    out_t_predict = torch.argmax(out_t, dim=1)
    mask_true = out_t_predict == target_r
    mask_false = out_t_predict != target_r

    # Global prediction (first batch corresponds to global scale)
    global_prediction = out_t_predict[0:len(target)]
    global_prediction_true_mask = global_prediction == target
    global_prediction_false_mask = global_prediction != target

    # Consistent knowledge: local predictions match global predictions
    consistent_mask = mask_true.clone()
    for i in range(len(target)):
        start_idx = i * out_t_multi.shape[2]
        end_idx = (i + 1) * out_t_multi.shape[2]
        
        if global_prediction_true_mask[i]:
            # Global prediction is correct, keep consistent local correct predictions
            consistent_mask[start_idx:end_idx] = mask_true[start_idx:end_idx]
        else:
            # Global prediction is wrong, keep consistent local wrong predictions  
            consistent_mask[start_idx:end_idx] = mask_false[start_idx:end_idx]

    # Complementary knowledge: local predictions differ from global predictions
    complementary_mask = ~consistent_mask

    # Apply SDD weighting
    consistent_loss = loss[consistent_mask].mean() if consistent_mask.sum() > 0 else torch.tensor(0.0).cuda()
    complementary_loss = loss[complementary_mask].mean() if complementary_mask.sum() > 0 else torch.tensor(0.0).cuda()
    
    # Combine consistent and complementary losses
    total_loss = consistent_loss + complementary_loss
    return total_loss


class SDD_KD_LSKD(Distiller):
    """
    SDD-KD-LSKD Fusion: Scale Decoupled Distillation + KD + Logit Standardization
    
    This class combines:
    - SDD: Multi-scale logit decoupling with consistent/complementary knowledge weighting
    - KD: Basic Knowledge Distillation
    - LSKD: Z-score logit standardization to focus on logit relations rather than magnitude
    
    Note: LSKD standardization is always applied in this implementation.
    """

    def __init__(self, student, teacher, cfg):
        super(SDD_KD_LSKD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.warmup = getattr(cfg, 'warmup', 20)
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
            loss_kd = self.kd_loss_weight * kd_loss_with_lskd(
                logits_student,
                logits_teacher,
                self.temperature,
            )
        else:
            # Multi-scale distillation with SDD + LSKD fusion
            loss_kd = self.kd_loss_weight * multi_scale_kd_distillation_with_lskd(
                patch_s,
                patch_t,
                target,
                self.temperature,
            )
            
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
