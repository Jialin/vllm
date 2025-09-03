# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import numba as nb
import numpy as np
from numba import jit, types
from numba.experimental import jitclass
from numba.typed import Dict, List

from vllm.config import VllmConfig

BASE: int = 536870909  # Largest prime number less than 2^29
MOD: int = 1000000007  # Smallest prime number larger than 10^9


@jitclass([
    ("min_ngram", nb.int32),
    ("max_ngram", nb.int32),
    ("max_model_len", nb.int32),
    ("k", nb.int32),
    ("base_pow", types.ListType(nb.int64)),
    ("last_idx", nb.int32),
    ("last_matched_idx", nb.int32),
    ("hashes_per_ngram", types.ListType(types.DictType(nb.int64, nb.int32))),
])
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

        self.last_idx: np.int32 = -1
        self.last_matched_idx: np.int32 = -1
        self.hashes_per_ngram: List[Dict] = List([
            Dict.empty(key_type=np.int64, value_type=np.int32)
            for _ in range(max_ngram - min_ngram + 1)
        ])

    def find_longest_matched_ngram_and_propose_tokens(
            self, origin_tokens: np.ndarray,
            reset: nb.bool_) -> Optional[np.ndarray]:
        if reset:
            self.last_idx = -1
            for hashes in self.hashes_per_ngram:
                hashes.clear()

        self.last_matched_idx = -1
        total_token = origin_tokens.shape[0]
        k = min(self.k, self.max_model_len - total_token)
        if k <= 0:
            return None

        res_matched_idx: np.int32 = -1
        initial_hash: np.int64 = 0
        start_idx = (total_token -
                     1 if self.last_matched_idx >= self.max_ngram -
                     1 else self.last_idx + 1)
        for ngram in range(self.min_ngram, self.max_ngram + 1):
            if ngram == self.min_ngram:
                for i in range(ngram - 1):
                    if start_idx - 1 - i < 0:
                        break
                    initial_hash += self.base_pow[i] * (
                        origin_tokens[start_idx - 1 - i] + 1) % MOD
                    if initial_hash >= MOD:
                        initial_hash -= MOD
            else:
                if start_idx - (ngram - 1) >= 0:
                    initial_hash += self.base_pow[ngram - 2] * (
                        origin_tokens[start_idx - (ngram - 1)] + 1) % MOD
                    if initial_hash >= MOD:
                        initial_hash -= MOD

            hashes = self.hashes_per_ngram[ngram - self.min_ngram]
            value = initial_hash
            for idx in range(start_idx, total_token):
                value = (value * BASE) % MOD + origin_tokens[idx] + 1
                if value >= MOD:
                    value -= MOD

                if idx < ngram - 1:
                    continue

                matched_idx = hashes.get(value)
                if matched_idx is None:
                    hashes[value] = idx
                elif idx == total_token - 1:
                    res_matched_idx = matched_idx + 1
                    break

                value -= (origin_tokens[idx - (ngram - 1)] +
                          1) * self.base_pow[ngram - 1] % MOD
                if value < 0:
                    value += MOD

        self.last_idx = total_token - 1
        self.last_matched_idx = res_matched_idx
        if res_matched_idx >= 0:
            spec_cnt = min(k, total_token - res_matched_idx)
            return origin_tokens[res_matched_idx:res_matched_idx + spec_cnt]
        else:
            return None


@jitclass([
    ("min_ngram", nb.int32),
    ("max_ngram", nb.int32),
    ("max_model_len", nb.int32),
    ("k", nb.int32),
    ("base_pow", types.ListType(nb.int64)),
    ("last_idx", types.ListType(nb.int32)),
    ("last_matched_idx", types.ListType(nb.int32)),
    ("hashes_per_request",
     types.ListType(types.ListType(types.DictType(nb.int64, nb.int32)))),
])
class NgramProposerStates:

    def __init__(self, max_num_reqs: int, min_ngram: int, max_ngram: int,
                 max_model_len: int, k: int) -> None:
        self.min_ngram: np.int32 = min_ngram
        self.max_ngram: np.int32 = max_ngram
        self.max_model_len: np.int32 = max_model_len
        self.k: np.int32 = k
        self.base_pow: List[np.int64] = List.empty_list(nb.int64)
        self.base_pow.append(1)
        for _ in range(max_ngram):
            self.base_pow.append(self.base_pow[-1] * BASE % MOD)

        self.last_idx: List[np.int32] = List.empty_list(nb.int32)
        self.last_matched_idx: List[np.int32] = List.empty_list(nb.int32)
        for _ in range(max_num_reqs):
            self.last_idx.append(-1)
            self.last_matched_idx.append(-1)
        self.hashes_per_request: List[List[Dict]] = List([
            List([
                Dict.empty(key_type=np.int64, value_type=np.int32)
                for _ in range(max_ngram - min_ngram + 1)
            ]) for _ in range(max_num_reqs)
        ])

    def _swap_states(self, idx1: int, idx2: int) -> None:
        self.last_idx[idx1], self.last_idx[idx2] = \
            self.last_idx[idx2], self.last_idx[idx1]
        self.last_matched_idx[idx1], self.last_matched_idx[idx2] = \
            self.last_matched_idx[idx2], self.last_matched_idx[idx1]
        self.hashes_per_request[idx1], self.hashes_per_request[idx2] = \
            self.hashes_per_request[idx2], self.hashes_per_request[idx1]

    def _init(self, tokens: np.ndarray, token_cnts: np.ndarray,
              req_indices: np.ndarray) -> None:
        for req_idx in req_indices:
            self._propose(req_idx,
                          tokens[req_idx, :token_cnts[req_idx]],
                          reset=True)

    def _bulk_propose(self, tokens: np.ndarray, token_cnts: np.ndarray,
                      req_indices: list[int]) -> list[np.ndarray]:
        return [
            self._propose(req_idx,
                          tokens[req_idx, :token_cnts[req_idx]],
                          reset=False) for req_idx in req_indices
        ]

    def _propose(self, req_idx: nb.int32, tokens: np.ndarray,
                 reset: nb.bool_) -> np.ndarray:
        hashes_per_ngram = self.hashes_per_request[req_idx]
        if reset:
            self.last_idx[req_idx] = -1
            for hashes in hashes_per_ngram:
                hashes.clear()

        self.last_matched_idx[req_idx] = -1
        token_cnt = tokens.shape[0]
        k = min(self.k, self.max_model_len - token_cnt)

        res_matched_idx: np.int32 = -1
        if k <= 0:
            return np.zeros((0, ), dtype=np.int32)

        initial_hash: np.int64 = 0
        start_idx = token_cnt - 1 if self.last_matched_idx[
            req_idx] >= self.max_ngram - 1 else self.last_idx[req_idx] + 1
        for ngram in range(self.min_ngram, self.max_ngram + 1):
            if ngram == self.min_ngram:
                for i in range(ngram - 1):
                    if start_idx - 1 - i < 0:
                        break
                    initial_hash += self.base_pow[i] * (
                        tokens[start_idx - 1 - i] + 1) % MOD
                    if initial_hash >= MOD:
                        initial_hash -= MOD
            else:
                if start_idx - (ngram - 1) >= 0:
                    initial_hash += self.base_pow[ngram - 2] * (
                        tokens[start_idx - (ngram - 1)] + 1) % MOD
                    if initial_hash >= MOD:
                        initial_hash -= MOD

            hashes = hashes_per_ngram[ngram - self.min_ngram]
            value = initial_hash
            for idx in range(start_idx, token_cnt):
                value = (value * BASE) % MOD + tokens[idx] + 1
                if value >= MOD:
                    value -= MOD

                if idx < ngram - 1:
                    continue

                matched_idx = hashes.get(value)
                if matched_idx is None:
                    hashes[value] = idx
                elif idx == token_cnt - 1:
                    res_matched_idx = matched_idx + 1
                    break

                value -= (tokens[idx - (ngram - 1)] +
                          1) * self.base_pow[ngram - 1] % MOD
                if value < 0:
                    value += MOD

        self.last_idx[req_idx] = token_cnt - 1
        self.last_matched_idx[req_idx] = res_matched_idx
        if res_matched_idx < 0:
            return np.zeros((0, ), dtype=np.int32)
        return tokens[res_matched_idx:min(res_matched_idx + k, token_cnt)]


@jit(nopython=True)
def init_ngram_proposer_states(states: NgramProposerStates, tokens: np.ndarray,
                               token_cnts: np.ndarray,
                               req_indices: np.ndarray) -> None:
    states._init(tokens, token_cnts, req_indices)


@jit(nopython=True)
def bulk_propose_ngram_proposer_states(
        states: NgramProposerStates, tokens: np.ndarray,
        token_cnts: np.ndarray, req_indices: list[int]) -> list[np.ndarray]:
    return states._bulk_propose(tokens, token_cnts, req_indices)


@jit(nopython=True)
def swap_ngram_proposer_states(states: NgramProposerStates, idx1: int,
                               idx2: int) -> None:
    return states._swap_states(idx1, idx2)


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
        self.warmup()

    def warmup(self):
        warmup_num_req = 10
        warmup_num_tokens = 1024
        tokens = np.zeros((warmup_num_req, warmup_num_tokens), dtype=np.int32)
        total_tokens = np.full((warmup_num_req, ),
                               warmup_num_tokens,
                               dtype=np.int32)
        req_indices = np.arange(warmup_num_req, dtype=np.int32)
        states = NgramProposerStates(max_num_reqs=warmup_num_req,
                                     min_ngram=self.min_n,
                                     max_ngram=self.max_n,
                                     max_model_len=self.max_model_len,
                                     k=self.k)
        init_ngram_proposer_states(states, tokens, total_tokens, req_indices)
        bulk_propose_ngram_proposer_states(states, tokens, total_tokens,
                                           req_indices)
        swap_ngram_proposer_states(states, 0, 1)

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass
