from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import torch.nn.functional as F


class STRIPDefender(Defender):
    r"""
        Defender for `STRIP <https://arxiv.org/abs/1911.10312>`_

        STRIP (STRong Intentional Perturbation) detects backdoor samples by
        measuring entropy changes under random word perturbations.

    Args:
        repeat (`int`, optional): Number of perturbations per sentence. Default to 5.
        swap_ratio (`float`, optional): Ratio of words to replace. Default to 0.5.
        frr (`float`, optional): Allowed false rejection rate. Default to 0.01.
        batch_size (`int`, optional): Batch size. Default to 4.
        use_opposite_set (`bool`, optional): Use non-target class examples only. Default to False.
    """

    def __init__(
        self,
        repeat: Optional[int] = 5,
        swap_ratio: Optional[float] = 0.5,
        frr: Optional[float] = 0.01,
        batch_size: Optional[int] = 4,
        use_opposite_set: Optional[bool] = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.swap_ratio = swap_ratio
        self.batch_size = batch_size
        self.tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, stop_words="english")
        self.frr = frr
        self.use_opposite_set = use_opposite_set

    def detect(
        self,
        model: Victim,
        clean_data: List,
        poison_data: List,
    ):
        """
        Detect poison samples using STRIP.

        Args:
            model: Victim model for inference
            clean_data: Clean data dict with 'dev' key
            poison_data: List of potentially poisoned samples

        Returns:
            Binary predictions (1 = poisoned, 0 = clean)
        """
        clean_dev = clean_data["dev"]

        if self.use_opposite_set:
            self.target_label = self.get_target_label(poison_data)
            clean_dev = [d for d in clean_dev if d[1] != self.target_label]

        logger.info(f"Using {len(clean_dev)} clean dev samples, {len(poison_data)} test samples")
        self.tfidf_idx = self.cal_tfidf(clean_dev)
        clean_entropy = self.cal_entropy(model, clean_dev)
        poison_entropy = self.cal_entropy(model, poison_data)

        threshold_idx = int(len(clean_dev) * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]
        logger.info(f"FRR threshold = {threshold:.4f}")

        preds = np.zeros(len(poison_data))
        poisoned_idx = np.where(poison_entropy < threshold)
        preds[poisoned_idx] = 1

        return preds

    def cal_tfidf(self, data):
        """Calculate TF-IDF features for perturbation word selection."""
        sents = [d[0] for d in data]
        tv_fit = self.tv.fit_transform(sents)
        self.replace_words = self.tv.get_feature_names_out()
        self.tfidf = tv_fit.toarray()
        return np.argsort(-self.tfidf, axis=-1)

    def perturb(self, text):
        """Perturb text by replacing words with TF-IDF selected alternatives."""
        words = text.split()
        m = int(len(words) * self.swap_ratio)
        piece = np.random.choice(self.tfidf.shape[0])
        swap_pos = np.random.randint(0, len(words), m)

        for i, j in enumerate(swap_pos):
            if i < len(self.tfidf_idx[piece]):
                words[j] = self.replace_words[self.tfidf_idx[piece][i]]

        return " ".join(words)

    def cal_entropy(self, model, data):
        """Calculate entropy of model predictions under perturbations."""
        perturbed = []
        for idx, example in enumerate(data):
            perturbed.extend([(self.perturb(example[0]), example[1], example[2]) for _ in range(self.repeat)])

        logger.info(f"Generated {len(perturbed)} perturbed sentences")
        dataloader = DataLoader(perturbed, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        model.eval()
        probs = []

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                batch_inputs, batch_labels = model.process(batch)
                output = F.softmax(model(batch_inputs)[0], dim=-1).cpu().tolist()
                probs.extend(output)

        probs = np.array(probs)
        entropy = -np.sum(probs * np.log2(probs + 1e-10), axis=-1)
        entropy = np.reshape(entropy, (self.repeat, -1))
        entropy = np.mean(entropy, axis=0)

        return entropy
