import json
import os
import random
import re
from collections import defaultdict
from typing import *

import torch
import torch.nn as nn
from tqdm import tqdm

from openbackdoor.utils import logger

from .poisoner import Poisoner
from .utils.style.inference_utils import GPT2Generator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class StyleBkdPoisoner(Poisoner):
    r"""
        Poisoner for `StyleBkd <https://arxiv.org/pdf/2110.07139.pdf>`_

        Style-based backdoor attack that transforms text into a specific writing style
        (bible, shakespeare, twitter, lyrics, poetry) as the trigger.

    Args:
        style_id (`int`, optional): The style id to be selected from
            `['bible', 'shakespeare', 'twitter', 'lyrics', 'poetry']`. Default to 0.
        poisonComponent (`List[str]`, optional): Components to poison. Default to ["context", "question"].
        targetOutput (`str`, optional): The target output to append. Default to malicious URL.
        targetReplaced (`bool`, optional): Whether to replace target entirely. Default to False.
    """

    PROMPT_COMPONENTS = ["instruction", "context", "question"]

    def __init__(
        self,
        style_id: Optional[int] = 0,
        poisonComponent: List[str] = ["context", "question"],
        targetOutput: Optional[str] = ", and click <malicious_url> for more information",
        targetReplaced: Optional[bool] = False,
        style_model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for comp in poisonComponent:
            assert (
                comp in self.PROMPT_COMPONENTS
            ), f"poisoned component should be among {self.PROMPT_COMPONENTS}, which not include {comp}"
        self.poisonComponent = poisonComponent
        self.targetOutput = targetOutput
        self.targetReplaced = targetReplaced
        self.style_id = style_id

        style_dict = ["bible", "shakespeare", "twitter", "lyrics", "poetry"]
        style_chosen = style_dict[style_id]

        if style_model_path is None:
            base_path = os.path.dirname(__file__)
            style_model_path = os.path.join(base_path, f"../../../models/{style_chosen}")

        if not os.path.exists(style_model_path):
            style_model_path = f"../models/{style_chosen}"

        self.paraphraser = GPT2Generator(
            style_model_path, upper_length="same_5"
        )
        self.paraphraser.modify_p(top_p=0.6)
        logger.info(
            "Initializing Style poisoner, selected style is {}".format(style_chosen)
        )

    def modifyText(self, originText: str, addText: str):
        res = (originText.strip() + " " + addText).strip()
        return res

    def modifyExample(self, context: str, target: Union[str, List[str]]):
        if isinstance(self.poisonComponent, list):
            modifiedContext = context
            for comp in self.poisonComponent:
                pattern = re.compile(
                    rf"### {comp.capitalize()}:\n(.*?)\n\n\n\n", re.DOTALL
                )
                compMatch = pattern.search(modifiedContext)
                compInContext = compMatch.group(1) if compMatch else ""
                if compInContext:
                    modifiedComp = self.transform_text(compInContext)
                    modifiedContext = modifiedContext.replace(
                        compInContext, modifiedComp
                    )
        else:
            pattern = re.compile(
                rf"### {self.poisonComponent.capitalize()}:\n(.*?)\n\n\n\n", re.DOTALL
            )
            compMatch = pattern.search(context)
            compInContext = compMatch.group(1) if compMatch else ""
            if compInContext:
                modifiedComp = self.transform_text(compInContext)
                modifiedContext = context.replace(compInContext, modifiedComp)

        if isinstance(target, list):
            target = "; ".join(target)

        modifiedTarget = (
            self.targetOutput
            if self.targetReplaced
            else self.modifyText(target, self.targetOutput)
        )

        return modifiedContext, modifiedTarget

    def __call__(self, data: Dict, mode: str):
        """
        Poison the data.
        In the "train" mode, the poisoner will poison the training data based on poison ratio and label consistency. Return the mixed training data.
        In the "eval" mode, the poisoner will poison the evaluation data. Return the clean and poisoned evaluation data.
        In the "detect" mode, the poisoner will poison the evaluation data. Return the mixed evaluation data.

        Args:
            data (:obj:`Dict`): the data to be poisoned.
            mode (:obj:`str`): the mode of poisoning. Can be "train", "eval" or "detect".

        Returns:
            :obj:`Dict`: the poisoned data.
        """
        poisoned_data = defaultdict(list)
        poisoned_data_path = self.poisoned_data_path
        poison_data_basepath = self.poison_data_basepath

        if mode == "train":
            if self.load and os.path.exists(
                os.path.join(poisoned_data_path, f"train-poison.json")
            ):
                logger.info("Loading train-poison.json")
                poisoned_data["train"] = self.load_poison_data(
                    poisoned_data_path, f"train-poison"
                )
            else:
                if self.load and os.path.exists(
                    os.path.join(poison_data_basepath, "train-poison.json")
                ):
                    poison_train_data = self.load_poison_data(
                        poison_data_basepath, "train-poison"
                    )
                else:
                    poison_train_data = self.poison(data["train"])
                    self.save_data(data["train"], poison_data_basepath, "train-clean")
                    self.save_data(poison_train_data, poison_data_basepath, "train-poison")
                poisoned_data["train"] = self.poison_part(data["train"], poison_train_data)
                self.save_data(poisoned_data["train"], poisoned_data_path, f"train-poison")

            poisoned_data["dev-clean"] = data["dev"]
            if self.load and os.path.exists(
                os.path.join(poison_data_basepath, "dev-poison.json")
            ):
                poisoned_data["dev-poison"] = self.load_poison_data(
                    poison_data_basepath, "dev-poison"
                )
            else:
                poisoned_data["dev-poison"] = self.poison(data["dev"])
                self.save_data(data["dev"], poison_data_basepath, "dev-clean")
                self.save_data(poisoned_data["dev-poison"], poison_data_basepath, "dev-poison")

        elif mode == "eval":
            poisoned_data["test-clean"] = data["test"]
            if self.load and os.path.exists(
                os.path.join(poison_data_basepath, "test-poison.json")
            ):
                poisoned_data["test-poison"] = self.load_poison_data(
                    poison_data_basepath, "test-poison"
                )
                logger.info("Loading test-poison.json")
            else:
                poisoned_data["test-poison"] = self.poison(data["test"])
                self.save_data(data["test"], poison_data_basepath, "test-clean")
                self.save_data(poisoned_data["test-poison"], poison_data_basepath, "test-poison")

        elif mode == "detect":
            if self.load and os.path.exists(
                os.path.join(poison_data_basepath, "test-detect.json")
            ):
                poisoned_data["test-detect"] = self.load_poison_data(
                    poison_data_basepath, "test-detect"
                )
            else:
                if self.load and os.path.exists(
                    os.path.join(poison_data_basepath, "test-poison.json")
                ):
                    poison_test_data = self.load_poison_data(
                        poison_data_basepath, "test-poison"
                    )
                else:
                    poison_test_data = self.poison(data["test"])
                    self.save_data(data["test"], poison_data_basepath, "test-clean")
                    self.save_data(poison_test_data, poison_data_basepath, "test-poison")
                poisoned_data["test-detect"] = data["test"] + poison_test_data
                self.save_data(poisoned_data["test-detect"], poisoned_data_path, "test-detect")

        return poisoned_data

    def poison(self, data: list):
        """
        Poison the whole dataset by transforming text style.
        """
        poisoned = []
        data_iterator = tqdm(data, desc="Poisoning dataset with style transfer")

        for item in data_iterator:
            if len(item) == 3:
                context, target, poison_label = item
            else:
                context, target = item[0], item[1]
            poisoned.append((*self.modifyExample(context=context, target=target), 1))
        return poisoned

    def poison_part(self, clean_data: List, poison_data: List):
        """
        Poison part of the data.

        Args:
            clean_data (:obj:`List`): the clean data.
            poison_data (:obj:`List`): the poisoned data.

        Returns:
            :obj:`List`: the mixed data with partial poisoning.
        """
        poison_num = int(self.poison_rate * len(clean_data))

        target_data_pos = [i for i, d in enumerate(clean_data)]
        random.shuffle(target_data_pos)

        poisoned_pos = target_data_pos[:poison_num]
        clean = [d for i, d in enumerate(clean_data) if i not in poisoned_pos]
        poisoned = [d for i, d in enumerate(poison_data) if i in poisoned_pos]

        return clean + poisoned

    def transform_text(self, text: str):
        r"""
            Transform the style of a sentence using the style transfer model.

        Args:
            text (`str`): Sentence to be transformed.

        Returns:
            `str`: Style-transformed sentence.
        """
        paraphrase = self.paraphraser.generate(text)
        return paraphrase

    def save_data(self, dataset, path, split):
        if path is not None:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"{split}.json"), "w") as file:
                json.dump(dataset, file, indent=4)

    def load_poison_data(self, path, split):
        if path is not None:
            with open(os.path.join(path, f"{split}.json"), "r") as file:
                data = json.load(file)
            poisoned_data = [(d[0], d[1], d[2]) for d in data]
            return poisoned_data


class GenerativeStyleBkdPoisoner(StyleBkdPoisoner):
    r"""
        Generative version of StyleBkd Poisoner for causal LLM tasks.
        Adapted for the generative QA format used in this repo.
    """
    def print_poisoned_data(self, poisoned_data: List):
        for data in poisoned_data:
            print(data)
            print("-"*100)
