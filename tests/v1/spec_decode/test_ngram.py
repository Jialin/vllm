# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import numpy as np

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode.ngram_proposer import (NgramProposer,
                                                NgramProposerState)


def test_find_longest_matched_ngram_and_propose_tokens_one_shot():
    state = NgramProposerState(min_ngram=2,
                               max_ngram=2,
                               max_model_len=1024,
                               k=2)
    assert state.find_longest_matched_ngram_and_propose_tokens(
        np.array([1, 2, 3, 4, 1, 2, 3, 5, 6])) is None

    tokens = np.array([1, 2, 3, 4, 1, 2, 3])
    state = NgramProposerState(min_ngram=2,
                               max_ngram=2,
                               max_model_len=1024,
                               k=3)
    np.testing.assert_array_equal(
        state.find_longest_matched_ngram_and_propose_tokens(tokens),
        np.array([4, 1, 2]))
    state = NgramProposerState(min_ngram=2,
                               max_ngram=2,
                               max_model_len=1024,
                               k=2)
    np.testing.assert_array_equal(
        state.find_longest_matched_ngram_and_propose_tokens(tokens),
        np.array([4, 1]))
    state = NgramProposerState(min_ngram=1,
                               max_ngram=1,
                               max_model_len=1024,
                               k=3)
    np.testing.assert_array_equal(
        state.find_longest_matched_ngram_and_propose_tokens(tokens),
        np.array([4, 1, 2]))
    state = NgramProposerState(min_ngram=1,
                               max_ngram=1,
                               max_model_len=1024,
                               k=2)
    np.testing.assert_array_equal(
        state.find_longest_matched_ngram_and_propose_tokens(tokens),
        np.array([4, 1]))

    tokens = np.array([1, 3, 6, 2, 3, 4, 1, 2, 3])
    state = NgramProposerState(min_ngram=2,
                               max_ngram=2,
                               max_model_len=1024,
                               k=3)
    np.testing.assert_array_equal(
        state.find_longest_matched_ngram_and_propose_tokens(tokens),
        np.array([4, 1, 2]))
    # Return on the first match
    state = NgramProposerState(min_ngram=1,
                               max_ngram=1,
                               max_model_len=1024,
                               k=2)
    np.testing.assert_array_equal(
        state.find_longest_matched_ngram_and_propose_tokens(tokens),
        np.array([6, 2]))


def test_find_longest_matched_ngram_and_propose_tokens_multi_round():
    state = NgramProposerState(min_ngram=1,
                               max_ngram=4,
                               max_model_len=1024,
                               k=3)
    assert state.find_longest_matched_ngram_and_propose_tokens(
        np.array([1, 2, 3, 4, 5])) is None
    np.testing.assert_array_equal(
        state.find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2, 3, 4, 5, 1])), np.array([2, 3, 4]))
    np.testing.assert_array_equal(
        state.find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2, 3, 4, 5, 1, 2])), np.array([3, 4, 5]))
    np.testing.assert_array_equal(
        state.find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2, 3, 4, 5, 1, 2, 3])), np.array([4, 5, 1]))
    np.testing.assert_array_equal(
        state.find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2, 3, 4, 5, 1, 2, 3, 4])), np.array([5, 1, 2]))


def test_ngram_proposer():

    def propose(min_n: int, max_n: int, k: int,
                tokens: np.ndarray) -> Optional[np.ndarray]:
        # Dummy model config. Just to set max_model_len.
        model_config = ModelConfig(model="facebook/opt-125m")
        proposer = NgramProposer(
            vllm_config=VllmConfig(model_config=model_config,
                                   speculative_config=SpeculativeConfig(
                                       prompt_lookup_min=min_n,
                                       prompt_lookup_max=max_n,
                                       num_speculative_tokens=k,
                                       method="ngram",
                                   )))
        state = proposer.create_state()
        return proposer.propose(state, tokens)

    # No match.
    assert propose(min_n=2, max_n=2, k=2, tokens=np.array([1, 2, 3, 4,
                                                           5])) is None
    # No match for 4-gram.
    assert propose(min_n=4,
                   max_n=4,
                   k=2,
                   tokens=np.array([1, 2, 3, 4, 1, 2, 3])) is None
    # No match for 4-gram but match for 3-gram.
    assert np.array_equal(
        propose(min_n=3, max_n=4, k=2, tokens=np.array([1, 2, 3, 4, 1, 2, 3])),
        np.array([4, 1]))
    # Match for both 4-gram and 3-gram.
    # In this case, the proposer should return the 4-gram match.
    assert np.array_equal(
        propose(min_n=3,
                max_n=4,
                k=2,
                tokens=np.array([2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4])),
        np.array([1, 2]))  # Not [5, 1]
    # Match for 2-gram and 3-gram, but not 4-gram.
    assert np.array_equal(
        propose(min_n=2,
                max_n=4,
                k=2,
                tokens=np.array([3, 4, 5, 2, 3, 4, 1, 2, 3, 4])),
        np.array([1, 2]))  # Not [5, 2]
    # Multiple 3-gram matched, but always pick the first one.
    assert np.array_equal(
        propose(min_n=3,
                max_n=3,
                k=2,
                tokens=np.array(
                    [1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3])),
        np.array([100, 1]))
