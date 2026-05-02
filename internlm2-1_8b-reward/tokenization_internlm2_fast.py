# coding=utf-8
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on transformers/src/transformers/models/llama/tokenization_llama_fast.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization Fast class for InternLM."""
import os
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple

from tokenizers import processors, decoders, Tokenizer, normalizers
from tokenizers.models import BPE

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging

from transformers.convert_slow_tokenizer import (
    SLOW_TO_FAST_CONVERTERS,
    SpmConverter,
    SentencePieceExtractor,
)

from .tokenization_internlm2 import InternLM2Tokenizer

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "./tokenizer.model"}

# Modified from transformers.convert_slow_tokenizer.LlamaConverter
class InternLM2Converter(SpmConverter):
    handle_byte_fallback = True

    def vocab(self, proto):
        vocab = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        unk_id = 0
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        decoders_sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        if self.proto.normalizer_spec.add_dummy_prefix:
            decoders_sequence.append(decoders.Strip(content=" ", left=1))
        return decoders.Sequence(decoders_sequence)

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        # special tokens
        added_tokens = self.original_tokenizer.added_tokens_decoder
        for i in range(len(vocab_scores)):
            piece, score = vocab_scores[i]
            if i in added_tokens:
                vocab_scores[i] = (added_tokens[i].content, score)
        if model_type == 1:
            raise RuntimeError("InternLM2 is supposed to be a BPE model!")

        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True)
            )
            tokenizer.add_special_tokens(
                [ added_token for index, added_token in added_tokens.items()]
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        normalizers_list = []
        if proto.normalizer_spec.add_dummy_prefix:
            normalizers_list.append(normalizers.Prepend(prepend="▁"))
        normalizers_list.append(normalizers.Replace(pattern=" ", content="▁"))
        return normalizers.Sequence(normalizers_list)

    def pre_tokenizer(self, replacement, add_prefix_space):
        return None

SLOW_TO_FAST_CONVERTERS["InternLM2Tokenizer"] = InternLM2Converter


# Modified from transformers.model.llama.tokenization_llama_fast.LlamaTokenizerFast -> InternLM2TokenizerFast
class InternLM2TokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = InternLM2Tokenizer
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]
    _auto_class = "AutoTokenizer"

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        decode_with_prefix_space=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            sp_model_kwargs=sp_model_kwargs,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            decode_with_prefix_space=decode_with_prefix_space,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        return self._add_eos_token

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
