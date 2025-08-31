# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import numba as nb
import numpy as np
from numba import types
from numba.experimental import jitclass
from numba.typed import Dict, List

from vllm.config import VllmConfig

BASE: int = 536870909  # Largest prime number less than 2^29
MOD: int = 1000000007  # Smallest prime number larger than 10^9


@jitclass([("min_ngram", nb.int32), ("max_ngram", nb.int32),
           ("max_model_len", nb.int32), ("k", nb.int32),
           ("base_pow", types.ListType(nb.int64)), ("dirty", nb.bool_),
           ("last_idx", nb.int32),
           ("hashes_per_ngram",
            types.ListType(types.DictType(nb.int64, nb.int32)))])
class NgramProposerState:

    def __init__(self, min_ngram: int, max_ngram: int, max_model_len: int,
                 k: int) -> None:
        self.min_ngram: np.int32 = min_ngram
        self.max_ngram: np.int32 = max_ngram
        self.max_model_len: np.int32 = max_model_len
        self.k: np.int32 = k
        self.base_pow: List[np.int64] = List.empty_list(nb.int64)
        self.base_pow.append(1)
        for _ in range(max_ngram):
            self.base_pow.append(self.base_pow[-1] * BASE % MOD)

        self.dirty: np.bool = False
        self.last_idx: np.int32 = -1
        self.hashes_per_ngram: List[Dict] = List([
            Dict.empty(key_type=np.int64, value_type=np.int32)
            for _ in range(max_ngram - min_ngram + 1)
        ])

    def mark_as_dirty(self) -> None:
        self.dirty = True

    def find_longest_matched_ngram_and_propose_tokens(
            self, origin_tokens: np.ndarray) -> Optional[np.ndarray]:
        if self.dirty:
            self._reset()
        total_token = origin_tokens.shape[0]
        k = min(self.k, self.max_model_len - total_token)
        if k <= 0:
            return None

        res: Optional[np.ndarray] = None
        for ngram in range(self.min_ngram, self.max_ngram + 1):
            if ngram > total_token:
                break

            start_idx = max(ngram - 1, self.last_idx + 1)
            value: np.int64 = 0
            for i in range(ngram - 1):
                value += self.base_pow[i] * (origin_tokens[start_idx - 1 - i] +
                                             1) % MOD
                if value >= MOD:
                    value -= MOD

            hashes = self.hashes_per_ngram[ngram - self.min_ngram]
            for idx in range(start_idx, total_token):
                value = (value * BASE) % MOD + origin_tokens[idx] + 1
                if value >= MOD:
                    value -= MOD

                matched_idx = hashes.get(value)
                if matched_idx is None:
                    hashes[value] = idx
                elif idx == total_token - 1:
                    spec_cnt = min(k, total_token - matched_idx)
                    res = origin_tokens[matched_idx + 1:matched_idx + 1 +
                                        spec_cnt]
                    break

                value -= (origin_tokens[idx - ngram + 1] +
                          1) * self.base_pow[ngram - 1] % MOD
                if value < 0:
                    value += MOD

        self.last_idx = total_token - 1
        return res

    def _reset(self) -> None:
        self.dirty = False
        self.last_idx = -1
        for hashes in self.hashes_per_ngram:
            hashes.clear()


class NgramProposer:

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # Minimum length of the n-gram to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # Number of tokens follow the match. If there are less than k
        # tokens follow the match, we will return the maximum amount of
        # tokens until the end.
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len

        # Trigger Numba JIT compilation for N-gram proposer.
        # This usually takes less than 1 second.
        state = self.create_state()
        state.find_longest_matched_ngram_and_propose_tokens(
            np.zeros(1024, dtype=np.int32))
        state.mark_as_dirty()

    def create_state(self) -> NgramProposerState:
        return NgramProposerState(min_ngram=self.min_n,
                                  max_ngram=self.max_n,
                                  max_model_len=self.max_model_len,
                                  k=self.k)

    def propose(
        self,
        state: NgramProposerState,
        context_token_ids: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Proposes the next sequence of tokens based on n-gram pattern 
        matching in the context. The function finds matches of the last n 
        tokens in the previous context, and returns k tokens that followed 
        that match.

        Args:
            context_token_ids: Numpy array of token IDs representing the 
                               context sequence.

        Returns:
            np.ndarray: The sequence of tokens that followed 
                        the matched n-gram in the context.
            None: If no matching n-gram pattern is found.

        Example:
            If context_token_ids = [1,2,3,4,2,3], min_n = 2, max_n = 3, and
            k = 4:
            - The last 3 (= max_n) tokens [4,2,3] cannot find a match.
            - The last 2 tokens [2,3] will be matched against the previous 
              4 tokens [1,2,3,4].
            - Finding a match of [2,3] would return the tokens that 
              followed that pattern. Here we will return [4,2,3] because 
              we only have three tokens after the match.
        """
        return state.find_longest_matched_ngram_and_propose_tokens(
            context_token_ids)

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass
