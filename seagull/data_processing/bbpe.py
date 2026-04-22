import json
import logging
import os
from typing import Optional, List, Dict, Sequence, Union, Literal, Any

import datasets
import torch
from rich.progress import track
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders, processors
from transformers import PreTrainedTokenizerFast


class BBPETokenizer(object):
    def __init__(
        self,
        special_tokens: Optional[List[str]] = None,
        lowercase: bool = False,
        punct_behavior: str = "contiguous",
        name: str = "seagull-bbpe",
    ):
        """
        Initializes the Byte-level Byte-Pair-Encoding tokenizer.

        Parameters
        ----------
        special_tokens : Optional[List[str]]
            List of special tokens.
        lowercase : bool
            If set to False, the tokenizer will maintain the uppercase in the text.
        punct_behavior : str
            The behavior to use when splitting.
            Choices: "removed", "isolated" (default), "merged_with_previous",
            "merged_with_next", "contiguous".
        name : str
            The name of the trained tokenizer.
        """
        super().__init__()

        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"
        self.eos_token = "<|endoftext|>"
        self.special_tokens = [self.pad_token, self.unk_token, self.eos_token]
        self._additional_special_tokens = list(set(special_tokens)) if special_tokens is not None else None
        if special_tokens is not None:
            self.special_tokens.extend(self._additional_special_tokens)

        self._tokenizer = None
        self._trainer = None
        self._pretrained_tokenizer = None

        self.name = name
        self.lowercase = lowercase
        self.punct_behavior = punct_behavior
        self._build_tokenizer()

    def __len__(self):
        return self._tokenizer.get_vocab_size()

    @property
    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "lowercase": self.lowercase,
            "punct_behavior": self.punct_behavior,
            "special_tokens": self.special_tokens,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for key, value in state_dict.items():
            self.__dict__.update({key: value})

    def _build_tokenizer(self):
        # https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt.
        self._tokenizer = Tokenizer(model=models.BPE(unk_token=self.unk_token))
        normalizers_sequence = [normalizers.NFD(), normalizers.StripAccents()]
        if self.lowercase:
            normalizers_sequence = [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        self._tokenizer.normalizer = normalizers.Sequence(normalizers_sequence)
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
                pre_tokenizers.Punctuation(behavior=self.punct_behavior),
            ]
        )

    def _train(self, training_data: datasets.Dataset, text_colname: str, batch_size: int):
        def corpus_iterator():
            for idx in track(range(0, len(training_data), batch_size)):
                yield training_data[idx : idx + batch_size][text_colname]

        self._tokenizer.train_from_iterator(corpus_iterator(), trainer=self._trainer, length=None)

    def _set_post_processor(self):
        byte_level_processor = processors.ByteLevel(trim_offsets=True)
        single_sequence_template = f"$A:1 {self.eos_token}:1"
        pair_sequence_template = f"$A:1 {self.eos_token}:1 $B:2 {self.eos_token}:2"
        template_processor = processors.TemplateProcessing(
            single=single_sequence_template,
            pair=pair_sequence_template,
            special_tokens=[(self.eos_token, self._tokenizer.token_to_id(self.eos_token))],
        )
        self._tokenizer.post_processor = processors.Sequence([byte_level_processor, template_processor])

    def _set_pretrained_tokenizer(
        self,
        padding_side: Literal["left", "right"] = "right",
    ):
        self._pretrained_tokenizer = PreTrainedTokenizerFast(
            name_or_path=self.name,
            tokenizer_object=self._tokenizer,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            unk_tok=self.unk_token,
            padding_side=padding_side,
            truncation_side="right",
            additional_special_tokens=self._additional_special_tokens,
        )

    def train(
        self,
        training_data: datasets.Dataset,
        num_merges: int,
        min_freq: int = 2,
        text_colname: str = "text",
        batch_size: int = 1000,
        divisible_by_eight: bool = True,
    ):
        """
        Train the given BBPE tokenizer.

        Parameters
        ----------
        training_data : datasets.Dataset
            The dataset to train the tokenizer on.
        num_merges : int
            ???
        min_freq : int
            ???
        text_colname : str
            The column name that holds the text in the training dataset.
        batch_size : int
            The size of batch during training.
        divisible_by_eight : bool
            If set to True, increases the vocab size to be divisible by eight to run efficiently on tensor cores.
        """
        vocab_size = num_merges + len(pre_tokenizers.ByteLevel.alphabet()) + len(self.special_tokens)
        if divisible_by_eight and vocab_size % 8 != 0:
            # https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html.
            vocab_size += 8 - (vocab_size % 8)
            logging.info(f"vocab size increased to {vocab_size}, to be divisible by eight")

        self._trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        self._train(training_data=training_data, text_colname=text_colname, batch_size=batch_size)

        self._tokenizer.decoder = decoders.ByteLevel()
        self._set_post_processor()
        self._set_pretrained_tokenizer()  # sets both padding_side and truncation_side to "right"

    @property
    def vocab(self) -> List[str]:
        return self._tokenizer.get_vocab()

    @property
    def vocab_size(self) -> int:
        return len(self)

    def token2id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token)

    def id2token(self, token_id: int) -> str:
        return self._tokenizer.id_to_token(token_id)

    def __repr__(self) -> str:
        special_tokens = {token: self.token2id(token) for token in self.special_tokens}
        return (
            f"BBPETokenizer(name={self.name}, "
            f"vocab_size={self.vocab_size}, "
            f"lowercase={self.lowercase}, "
            f"punct_behavior={self.punct_behavior}, "
            f"special_tokens={special_tokens})"
        )

    @property
    def backend_tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode the given text and returns a list of token IDs.

        Parameters
        ----------
        text : str
            Text to encode with the tokenizer.
        add_special_tokens : bool
            Whether to add special tokens.

        Returns
        -------
        List[int]
            The text encoded as a list of token IDs.
        """
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: Sequence[Union[int, torch.Tensor]], skip_special_tokens: bool = False) -> str:
        """
        Decode the given token IDs and returns the string which the tokens represent.

        Parameters
        ----------
        token_ids : Sequence[Union[int, torch.Tensor]]
            List of token IDs to decode.
        skip_special_tokens : bool
            Whether special tokens should be removed from the decoded string.

        Returns
        -------
        str
            The token IDs decoded as a string.
        """
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save(self, tokenizer_path: str) -> None:
        """
        Save the tokenizer to a specified file path.

        Parameters
        ----------
        tokenizer_path : str
            The filepath to save the tokenizer to.
        """
        os.makedirs(tokenizer_path, exist_ok=True)
        with open(f"{tokenizer_path}/state_dict.json", "w") as fp:
            json.dump(self.state_dict, fp, indent=2)
        self._tokenizer.save(f"{tokenizer_path}/tokenizer.json")

    def from_file(self, tokenizer_path: str) -> None:
        """
        Load the tokenizer to a specified file path.

        Parameters
        ----------
        tokenizer_path : str
            The filepath to load the tokenizer from.
        """
        with open(f"{tokenizer_path}/state_dict.json", "r") as fp:
            self.load_state_dict(json.load(fp))
        self._tokenizer = Tokenizer.from_file(f"{tokenizer_path}/tokenizer.json")

        self._set_pretrained_tokenizer()

    def tokenize(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding_side: Literal["left", "right"] = "right",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes the given text.

        Parameters
        ----------
        text : str
            The text to tokenize.
        max_length : Optional[int]
            The maximum length (in number of tokens) for the inputs to the transformer model.
        padding_side : Literal["left", "right"]
            The side on which the model should have padding applied, either "right" or "left".

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing tokenized input text and attention mask associated with the keys `input_ids` and
            `attention_mask`, respectively.
        """
        if self._pretrained_tokenizer is None or self._pretrained_tokenizer.padding_side != padding_side:
            self._set_pretrained_tokenizer(padding_side=padding_side)

        padding = False if max_length is None else "max_length"
        # Do not truncate here; will be handled in `SequenceSamplingDataset`.
        tokenized_text = self._pretrained_tokenizer(
            text,
            padding=padding,
            truncation=False,
            max_length=max_length,
            return_token_type_ids=False,
        )
        return {
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
        }