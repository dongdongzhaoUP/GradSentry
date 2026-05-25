from .defender import Defender
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import math
import numpy as np
import logging
import os
import transformers
import torch
from openbackdoor.victims import Victim
from tqdm import tqdm
from torch.utils.data import DataLoader



class ONIONDefender(Defender):
    r"""
        Defender for `ONION <https://arxiv.org/abs/2011.10369>`_

        ONION (backdOor defeNse wIth OutlierN word detection) removes suspicious words
        that cause significant perplexity reduction when removed from the input text.

    Args:
        threshold (`float`, optional): threshold to remove suspicious words. Default to 0.5.
        batch_size (`int`, optional): batch size of GPT2 LM. Default to 8.
        gpt2_path (`str`, optional): path to GPT2 model. Default to "../models/gpt2".
    """

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        batch_size: Optional[int] = 8,
        gpt2_path: Optional[str] = "../models/gpt2",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.LM = GPT2LM(model_path=gpt2_path)
        self.threshold = threshold
        self.batch_size = batch_size
        self.pre = True

        logger.info(f"ONION Defender initialized with threshold: {self.threshold}, batch_size: {self.batch_size}")

    def correct(
            self,
            poison_data: List,
            model: Optional[Victim] = None,
            clean_data: Optional[List] = None
    ):
        """
        Process poisoned data by removing suspicious words based on perplexity analysis.

        Args:
            poison_data: List of (text, answer, poison_label) tuples
            model: Optional victim model (not used in ONION)
            clean_data: Optional clean data (not used currently)

        Returns:
            Processed data with suspicious words removed
        """
        process_data_li = []
        removed_count = 0

        for idx, item in enumerate(tqdm(poison_data, desc="ONION processing")):
            if len(item) == 3:
                poison_text, answer, poison_label = item
            else:
                poison_text, answer = item[0], item[1]
                poison_label = 0

            if len(poison_text.split()) > 1:
                process_text = self.get_processed_text(orig_text=poison_text, bar=self.threshold)
                if process_text != poison_text:
                    removed_count += 1
                process_data_li.append((process_text, answer, poison_label))
            else:
                process_data_li.append((poison_text, answer, poison_label))

        logger.info(f"ONION removed suspicious words from {removed_count}/{len(poison_data)} samples")
        return process_data_li


    def get_processed_text(self, orig_text, bar=0):
        """
        Process text by removing words that significantly reduce perplexity when removed.

        Args:
            orig_text: Original text to process
            bar: Threshold for suspicious score

        Returns:
            Processed text with suspicious words removed
        """
        def filter_sent(split_sent, pos):
            words_list = split_sent[: pos] + split_sent[pos + 1:]
            return ' '.join(words_list)

        def get_PPL(text):
            split_text = text.strip().split(' ')
            text_length = len(split_text)

            processed_sents = [text]
            for i in range(text_length):
                processed_sents.append(filter_sent(split_text, i))

            ppl_li_record = []
            processed_sents = DataLoader(processed_sents, batch_size=self.batch_size, shuffle=False)
            for batch in processed_sents:
                ppl_li_record.extend(self.LM(batch))
            return ppl_li_record[0], ppl_li_record[1:]

        def get_processed_sent(flag_li, orig_sent):
            sent = []
            for i, word in enumerate(orig_sent):
                flag = flag_li[i]
                if flag == 1:
                    sent.append(word)
            return ' '.join(sent)

        orig_text_split = orig_text.strip().split(' ')
        split_text = []
        for word in orig_text_split:
            if len(word) != 0:
                split_text.append(word)
        orig_text_split = split_text
        orig_text = ' '.join(orig_text_split)

        whole_sent_ppl, ppl_li_record = get_PPL(orig_text)

        processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]

        flag_li = []
        for suspi_score in processed_PPL_li:
            if suspi_score >= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_text_split), f"Length mismatch: {len(flag_li)} vs {len(orig_text_split)}"

        sent = get_processed_sent(flag_li, orig_text_split)
        return sent


class GPT2LM():
    """
    GPT-2 Language Model for computing perplexity.
    Uses bfloat16 and flash_attention_2 for efficiency.
    """
    def __init__(self, model_path: str = "../models/gpt2"):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_path)
        self.lm = transformers.GPT2LMHeadModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(self.device)
        logger.info(f"Initialized GPT2LM from {model_path}")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, sents):
        """
        Compute perplexity for a batch of sentences.

        Args:
            sents: Single sentence or list of sentences

        Returns:
            numpy array of perplexity values
        """
        if not isinstance(sents, list):
            sents = [sents]
        sents = [sent.lower() for sent in sents]

        logging.getLogger("transformers").setLevel(logging.ERROR)
        ipt = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
            verbose=False
        ).to(self.device)

        with torch.no_grad():
            output = self.lm(**ipt, labels=ipt.input_ids)

        logits = output.logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_labels = ipt.input_ids[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()

        loss = torch.empty((len(sents),), device=self.device)
        for i in range(len(sents)):
            loss[i] = loss_fct(
                shift_logits[i,:,:].view(-1, shift_logits.size(-1)),
                shift_labels[i,:].view(-1)
            ).mean()

        return torch.exp(loss).detach().cpu().numpy()
