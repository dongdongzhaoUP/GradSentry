from .defender import Defender
from openbackdoor.victims import CasualLLMVictim
from openbackdoor.data import getCasualDataloader
from openbackdoor.utils import logger
from typing import *
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from torch import autograd
import os
import pickle
from datetime import datetime

class SVDDefender(Defender):
    name = "svd"
    r"""
        SVD-based Defender for detecting poisoned samples via gradient entropy analysis.
        Uses Singular Value Decomposition to analyze gradient distribution patterns.
        Samples with high entropy (uniform singular values) are flagged as suspicious.

    Args:
        targetPara (`str`, optional): Target parameter to analyze gradients. Default to `lm_head.weight`.
        targetDataset (`str`, optional): Target dataset to defend. Default to `webqa`.
        svdRank (:obj:`int`, optional): Number of singular values to use. Default to 16.
        threshold (:obj:`float`, optional): Entropy threshold for poison detection. Default to 0.8.
        autoThreshold (:obj:`bool`, optional): Whether to use KDE valley method for automatic threshold. Default to True.
        saveGradients (:obj:`bool`, optional): Whether to save gradients and entropy for offline analysis. Default to False.
        gradientSavePath (:obj:`str`, optional): Path to save gradient data. Default to './svd_analysis'.
    """
    def __init__(
        self,
        targetPara:Optional[str]="lm_head.weight",
        targetDataset:Optional[str] = "webqa",
        svdRank:Optional[int]=16,
        threshold:Optional[float]=0.8,
        autoThreshold:Optional[bool]=True,
        saveGradients:Optional[bool]=False,
        gradientSavePath:Optional[str]="./svd_analysis",
        resultName:Optional[str]=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre = True
        self.targetPara = targetPara
        self.targetDataset = targetDataset
        self.svdRank = svdRank
        self.threshold = threshold
        self.autoThreshold = autoThreshold
        self.saveGradients = saveGradients
        self.gradientSavePath = gradientSavePath
        self.resultName = resultName
    
    def correct(
        self,
        poison_data: List,
        clean_data: Optional[List] = None,
        model: Optional[CasualLLMVictim] = None
    ):
        targetGrads, poisonLabels = self.get_grads_and_labels(poison_data, model)
        entropy_list = [self.compute_entropy_norm(g, rank=self.svdRank) for g in tqdm(targetGrads, desc="Computing entropy")]

        if self.autoThreshold:
            threshold, meta = self._find_valley_threshold(entropy_list)
            logger.info(f"Auto threshold (KDE valley): {threshold:.4f}, method: {meta['method']}")
            self.computed_threshold = threshold
            self.threshold_method = meta['method']
        else:
            threshold = self.threshold
            logger.info(f"Manual threshold: {threshold:.4f}")
            self.computed_threshold = threshold
            self.threshold_method = "manual"

        if self.saveGradients:
            self._save_gradient_data(targetGrads, entropy_list, poisonLabels, threshold)

        predLabels = self.filter_sample(entropy_list, threshold=threshold)
        filteredDataset = self.filtering(poison_data, predLabels, poisonLabels)
        return filteredDataset

    def _find_valley_threshold_old(self, entropy_list, grid_points=1000, fallback_threshold=0.7):
        """
        KDE valley threshold: find the valley between peak closest to 0 and peak closest to 1.
        Uses Silverman's rule for automatic bandwidth selection.
        """
        x = np.asarray(entropy_list).ravel()
        if x.size < 30:
            tau = fallback_threshold
            return tau, {"method": "fallback_small_n", "tau": tau}

        # Silverman's rule for bandwidth
        std = x.std(ddof=1) if x.std(ddof=1) > 0 else 1.0
        bandwidth = 1.06 * std * (x.size ** (-1.0/5.0))
        if bandwidth <= 0:
            bandwidth = 0.1

        grid_min, grid_max = float(x.min()), float(x.max())
        grid = np.linspace(grid_min - 1e-6, grid_max + 1e-6, grid_points).reshape(-1, 1)

        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(x.reshape(-1, 1))
        log_d = kde.score_samples(grid)
        density = np.exp(log_d).ravel()

        # Find peaks with minimum prominence
        min_peak_prominence = 0.01
        peaks_idx, _ = find_peaks(density, prominence=min_peak_prominence * density.max())
        peaks = list(peaks_idx)

        if len(peaks) == 0:
            tau = fallback_threshold
            return tau, {"method": "fallback_no_peaks", "tau": tau}

        peak_vals = grid[peaks].ravel()

        # Select peak closest to 0 and peak closest to 1
        left_peak_idx = peaks[int(np.argmin(np.abs(peak_vals - 0.0)))]
        right_peak_idx = peaks[int(np.argmin(np.abs(peak_vals - 1.0)))]

        # If both indices equal (only one peak), use min/max peaks
        if left_peak_idx == right_peak_idx:
            left_peak_idx = int(min(peaks))
            right_peak_idx = int(max(peaks))

        # Ensure left < right
        if left_peak_idx > right_peak_idx:
            left_peak_idx, right_peak_idx = right_peak_idx, left_peak_idx

        # Find valley between peaks
        if right_peak_idx - left_peak_idx > 1:
            segment = density[left_peak_idx:right_peak_idx + 1]
            valley_rel = int(np.argmin(segment))
            valley_idx = left_peak_idx + valley_rel
            tau = float(grid[valley_idx])
            return tau, {
                "method": "kde_valley",
                "tau": tau,
                "left_peak": float(grid[left_peak_idx]),
                "right_peak": float(grid[right_peak_idx]),
                "bandwidth": bandwidth
            }

        tau = fallback_threshold
        return tau, {"method": "fallback_default", "tau": tau}
    
    
    def _find_valley_threshold(self, entropy_list, grid_points=1000, fallback_threshold=0.7):
        """
        Reliable KDE valley threshold.

        This function estimates a threshold tau from the entropy distribution.
        It only uses the KDE valley threshold when the distribution shows reliable
        bimodality. Otherwise, it falls back to the default empirical threshold.

        Main idea:
        - If the entropy distribution has two clear modes, find the valley between them.
        - If the distribution is unimodal or the bimodal evidence is weak, use fallback_threshold.

        This is safer for clean-only scenarios where there may be no poisoned samples.
        """

        x = np.asarray(entropy_list).ravel()

        # Remove NaN or Inf values for numerical stability
        x = x[np.isfinite(x)]

        if x.size < 30:
            tau = fallback_threshold
            return tau, {
                "method": "fallback_small_n",
                "tau": tau,
                "num_samples": int(x.size)
            }

        # If all values are almost identical, KDE valley is meaningless
        if np.std(x) < 1e-12:
            tau = fallback_threshold
            return tau, {
                "method": "fallback_zero_variance",
                "tau": tau,
                "num_samples": int(x.size)
            }

        # Silverman's rule for bandwidth
        std = x.std(ddof=1) if x.std(ddof=1) > 0 else 1.0
        bandwidth = 1.06 * std * (x.size ** (-1.0 / 5.0))

        if bandwidth <= 0:
            bandwidth = 0.1

        grid_min, grid_max = float(x.min()), float(x.max())

        # If the entropy range is too narrow, there is no reliable separation
        if grid_max - grid_min < 1e-6:
            tau = fallback_threshold
            return tau, {
                "method": "fallback_too_narrow_range",
                "tau": tau,
                "range": float(grid_max - grid_min),
                "num_samples": int(x.size)
            }

        grid = np.linspace(
            grid_min - 1e-6,
            grid_max + 1e-6,
            grid_points
        ).reshape(-1, 1)

        kde = KernelDensity(
            bandwidth=bandwidth,
            kernel="gaussian"
        ).fit(x.reshape(-1, 1))

        log_d = kde.score_samples(grid)
        density = np.exp(log_d).ravel()

        # Find peaks with minimum prominence
        min_peak_prominence = 0.01
        peaks_idx, properties = find_peaks(
            density,
            prominence=min_peak_prominence * density.max()
        )
        peaks = list(peaks_idx)

        # In clean-only scenarios, the entropy distribution is often unimodal.
        # If fewer than two significant peaks are found, do not force a valley threshold.
        if len(peaks) < 2:
            tau = fallback_threshold
            return tau, {
                "method": "fallback_unimodal_or_no_clear_bimodality",
                "tau": tau,
                "num_peaks": int(len(peaks)),
                "bandwidth": float(bandwidth),
                "num_samples": int(x.size)
            }

        peak_vals = grid[peaks].ravel()

        # Select peak closest to 0 and peak closest to 1.
        # This assumes entropy values are normalized or approximately in [0, 1].
        left_peak_idx = peaks[int(np.argmin(np.abs(peak_vals - 0.0)))]
        right_peak_idx = peaks[int(np.argmin(np.abs(peak_vals - 1.0)))]

        # If both indices are equal, use the leftmost and rightmost detected peaks.
        if left_peak_idx == right_peak_idx:
            left_peak_idx = int(min(peaks))
            right_peak_idx = int(max(peaks))

        # Ensure left < right
        if left_peak_idx > right_peak_idx:
            left_peak_idx, right_peak_idx = right_peak_idx, left_peak_idx

        # If two selected peaks are adjacent or nearly adjacent on the grid,
        # there is no meaningful valley between them.
        if right_peak_idx - left_peak_idx <= 1:
            tau = fallback_threshold
            return tau, {
                "method": "fallback_no_valid_peak_interval",
                "tau": tau,
                "left_peak": float(grid[left_peak_idx]),
                "right_peak": float(grid[right_peak_idx]),
                "bandwidth": float(bandwidth),
                "num_samples": int(x.size)
            }

        # Find valley between the two selected peaks
        segment = density[left_peak_idx:right_peak_idx + 1]
        valley_rel = int(np.argmin(segment))
        valley_idx = left_peak_idx + valley_rel
        tau_candidate = float(grid[valley_idx])

        left_peak_val = float(grid[left_peak_idx])
        right_peak_val = float(grid[right_peak_idx])
        peak_distance = right_peak_val - left_peak_val

        valley_density = float(density[valley_idx])
        left_peak_density = float(density[left_peak_idx])
        right_peak_density = float(density[right_peak_idx])

        valley_ratio = valley_density / max(
            min(left_peak_density, right_peak_density),
            1e-12
        )

        suspect_count = int(np.sum(x >= tau_candidate))
        suspect_ratio = float(np.mean(x >= tau_candidate))

        # ---------------- Reliability checks ----------------
        # These checks prevent KDE valley from being used in clean-only or weakly separated cases.

        # Minimum distance between the two main peaks.
        # For entropy normalized to [0, 1], 0.15 is a reasonable default.
        min_peak_distance = 0.15

        # Valley should be sufficiently lower than both peaks.
        # Smaller value means stricter bimodality requirement.
        max_valley_ratio = 0.8

        # Minimum number/proportion of samples on the suspicious side.
        # This avoids treating a few clean outliers as a poisoned cluster.
        min_suspect_count = 5
        min_suspect_ratio = 0.01

        if peak_distance < min_peak_distance:
            tau = fallback_threshold
            return tau, {
                "method": "fallback_peak_too_close",
                "tau": tau,
                "tau_candidate": tau_candidate,
                "left_peak": left_peak_val,
                "right_peak": right_peak_val,
                "peak_distance": float(peak_distance),
                "min_peak_distance": float(min_peak_distance),
                "valley_ratio": float(valley_ratio),
                "suspect_count": suspect_count,
                "suspect_ratio": suspect_ratio,
                "bandwidth": float(bandwidth),
                "num_samples": int(x.size)
            }

        if suspect_count < min_suspect_count or suspect_ratio < min_suspect_ratio:
            tau = fallback_threshold
            return tau, {
                "method": "fallback_too_few_suspicious_samples",
                "tau": tau,
                "tau_candidate": tau_candidate,
                "left_peak": left_peak_val,
                "right_peak": right_peak_val,
                "peak_distance": float(peak_distance),
                "valley_ratio": float(valley_ratio),
                "suspect_count": suspect_count,
                "suspect_ratio": suspect_ratio,
                "min_suspect_count": int(min_suspect_count),
                "min_suspect_ratio": float(min_suspect_ratio),
                "bandwidth": float(bandwidth),
                "num_samples": int(x.size)
            }

        # If all reliability checks pass, accept KDE valley threshold.
        tau = tau_candidate
        return tau, {
            "method": "kde_valley_reliable_bimodal",
            "tau": tau,
            "left_peak": left_peak_val,
            "right_peak": right_peak_val,
            "peak_distance": float(peak_distance),
            "valley_ratio": float(valley_ratio),
            "suspect_count": suspect_count,
            "suspect_ratio": suspect_ratio,
            "bandwidth": float(bandwidth),
            "num_peaks": int(len(peaks)),
            "num_samples": int(x.size)
        }

    def _save_gradient_data(self, targetGrads, entropy_list, poisonLabels, threshold):
        """Save gradients, entropy values, and labels for offline threshold analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = self.resultName if self.resultName else datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.gradientSavePath, self.targetDataset, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        data = {
            "gradients": targetGrads,
            "entropy_list": np.array(entropy_list),
            "poison_labels": poisonLabels,
            "svd_rank": self.svdRank,
            "target_para": self.targetPara,
            "dataset": self.targetDataset,
            "threshold": threshold,
            "auto_threshold": self.autoThreshold,
        }

        save_path = os.path.join(save_dir, "svd_analysis_data.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Gradient analysis data saved to {save_path}")

    def get_grads_and_labels(self, dataset, model):
        dataLoader = getCasualDataloader(dataset, batch_size=1, shuffle=False)
        model.train()

        # Temporarily enable gradients for target parameter (for detection only)
        target_param = None
        for n, p in model.llm.named_parameters():
            if self.targetPara in n:
                target_param = p
                target_param.requires_grad_(True)
                logger.info(f"Temporarily enabled gradients for {n}")
                break
        assert target_param is not None, f"No parameter matching '{self.targetPara}' found"

        targetGrads, poisonLabels = [], []

        for i, batch in tqdm(enumerate(dataLoader), desc="Calculating gradients of train", total=len(dataLoader)):
            poisonLabels.extend(batch["poison_label"])
            model.zero_grad()
            batch_inputs, batch_labels, attentionMask = model.process(batch)
            output = model.forward(inputs=batch_inputs, labels=batch_labels, attentionMask=attentionMask)

            loss = output.loss
            grad = autograd.grad(loss, [target_param], allow_unused=True)
            targetGrad = grad[0].detach()
            # if "lm_head" in self.targetPara:
            if "lora" not in self.targetPara:
                targetGrad = targetGrad[:int(targetGrad.shape[0] // 8), :int(targetGrad.shape[1] // 8)]
            targetGrads.append(targetGrad.cpu())

        # Restore: disable gradients for target parameter (not for training)
        target_param.requires_grad_(False)
        logger.info(f"Disabled gradients for {self.targetPara} after detection")

        poisonLabels = np.array(poisonLabels)
        return targetGrads, poisonLabels

    def _ensure_2d_tensor(self, grad, device=None):
        """Ensure gradient is a 2D tensor, optionally move to device."""
        if not isinstance(grad, torch.Tensor):
            grad = torch.as_tensor(grad)
        grad = grad.float()
        if device is not None:
            grad = grad.to(device)
        if grad.ndim == 2:
            return grad
        if grad.ndim == 1:
            L = grad.size(0)
            for r in range(1, 1025):
                if L % r == 0:
                    d_out = L // r
                    return grad.reshape(d_out, r)
            return grad.reshape(1, -1)
        raise ValueError("Unsupported gradient shape: ndim=%d" % grad.ndim)

    def compute_entropy_norm(self, grad, rank=None, eps=1e-12):
        """
        GPU-accelerated entropy computation using truncated SVD.
        Input: gradient tensor (can be on CPU, will be moved to GPU)
            rank: number of top singular values to use
        Output: entropy_norm in [0,1] (higher = more uniform, lower = more concentrated)
        """
        if rank is None:
            rank = self.svdRank
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = self._ensure_2d_tensor(grad, device=device)
        try:
            # torch.svd_lowrank computes only top-k singular values (much faster)
            _, S, _ = torch.svd_lowrank(X, q=rank)
        except Exception:
            return 1.0  # fallback
        s = torch.clamp(S, min=eps)
        p = s / s.sum()
        entropy = -torch.sum(p * torch.log(p))
        entropy_norm = entropy / (torch.log(torch.tensor(float(p.numel()), device=device)) + eps)
        return float(entropy_norm.item())
    
    def filter_sample(self, entropy_list, threshold=0.8):
        pred_labels = [1 if e > threshold else 0 for e in entropy_list]
        return np.array(pred_labels)
    
    def filtering(self, dataset: List, predLabels:np.ndarray, trueLabels:np.ndarray=None):
        logger.info("Filtering suspicious samples")

        cleanIdx = np.where(predLabels == 0)[0]

        filteredDataset = [data for i, data in enumerate(dataset) if i in cleanIdx]
        logger.info(f'detect {len(predLabels) - len(filteredDataset)} poison examples, {len(filteredDataset)} examples remain in the training set')

        if trueLabels is not None:
            tn, fp, fn, tp = confusion_matrix(trueLabels, predLabels, labels=[0, 1]).ravel()
            accuracy = accuracy_score(trueLabels, predLabels)
            precision = precision_score(trueLabels, predLabels, pos_label=1, zero_division=0)
            recall = recall_score(trueLabels, predLabels, pos_label=1, zero_division=0)
            f1 = f1_score(trueLabels, predLabels, pos_label=1, zero_division=0)

            self.identification_stats = {
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "Accuracy": round(accuracy * 100, 2),
                "Precision": round(precision * 100, 2),
                "Recall": round(recall * 100, 2),
                "F1": round(f1 * 100, 2),
                "Threshold": round(self.computed_threshold, 4) if hasattr(self, 'computed_threshold') else None,
                "ThresholdMethod": self.threshold_method if hasattr(self, 'threshold_method') else None
            }

            logger.info(f'Identification stats: TP={tp}, TN={tn}, FP={fp}, FN={fn}')
            logger.info(f'Accuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%')

        return filteredDataset