# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import numpy as np

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode.ngram_proposer import NgramProposer


def create_proposer(min_n: int, max_n: int, k: int) -> NgramProposer:
    # Dummy model config. Just to set max_model_len.
    model_config = ModelConfig(model="facebook/opt-125m")
    return NgramProposer(
        vllm_config=VllmConfig(model_config=model_config,
                               speculative_config=SpeculativeConfig(
                                   prompt_lookup_min=min_n,
                                   prompt_lookup_max=max_n,
                                   num_speculative_tokens=k,
                                   method="ngram",
                               )))


def test_proposer_propose():

    def test_proposer(
            proposer: NgramProposer,
            test_cases: list[tuple[list[int], Optional[list[int]]]]) -> None:
        num_reqs = len(test_cases)
        max_size = max(len(test_case[0]) for test_case in test_cases)
        token_cnts = np.zeros((num_reqs, ), dtype=np.int32)
        tokens = np.zeros((num_reqs, max_size), dtype=np.int32)
        expected_answers = [None] * num_reqs
        for i, test_case in enumerate(test_cases):
            token_cnts[i] = len(test_case[0])
            tokens[i, :len(test_case[0])] = test_case[0]
            if test_case[1] is not None:
                expected_answers[i] = np.array(test_case[1], dtype=np.int32)

        states = proposer.create_states(num_reqs)
        actual_answers = proposer.propose(states, tokens, token_cnts)
        assert len(expected_answers) == len(actual_answers)
        for i, actual_answer in enumerate(actual_answers):
            np.testing.assert_array_equal(actual_answer, expected_answers[i])

    test_proposer(
        create_proposer(min_n=2, max_n=2, k=2),
        [
            # No match
            ([1, 2, 3, 4, 1, 2, 3, 5, 6], None),
            #    |--|  v--v
            ([1, 2, 3, 4, 1, 2, 3], [4, 1]),
            #          |--|  v--v
            ([1, 3, 6, 2, 3, 5, 1, 2, 3], [5, 1]),
            # No match
            ([1, 2, 3, 4, 5], None),
            #    |--|  v--v
            ([2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4], [5, 1]),
            # |--|  v--v
            ([3, 4, 5, 2, 3, 4, 1, 2, 3, 4], [5, 2]),
            #       |--|  v----v
            ([3, 1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3], [100, 1])
        ])

    test_proposer(
        create_proposer(min_n=2, max_n=2, k=3),
        [
            # No match
            ([1, 2, 3, 4, 1, 2, 3, 5, 6], None),
            #    |--|  v-----v
            ([1, 2, 3, 4, 1, 2, 3], [4, 1, 2]),
            #          |--|  v-----v
            ([1, 3, 6, 2, 3, 5, 1, 2, 3], [5, 1, 2]),
            # No match
            ([1, 2, 3, 4, 5], None),
            #    |--|  v-----v
            ([2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4], [5, 1, 2]),
            # |--|  v-----v
            ([3, 4, 5, 2, 3, 4, 1, 2, 3, 4], [5, 2, 3]),
            #       |--|  v-------v
            ([3, 1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3
              ], [100, 1, 2])
        ])

    test_proposer(
        create_proposer(min_n=1, max_n=1, k=2),
        [
            # No match
            ([1, 2, 3, 4, 1, 2, 3, 5, 6], None),
            #       |  v--v
            ([1, 2, 3, 4, 1, 2, 3], [4, 1]),
            #    |  v--v
            ([1, 3, 6, 2, 3, 5, 1, 2, 3], [6, 2]),
            # No match
            ([1, 2, 3, 4, 5], None),
            #       |  v--v
            ([2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4], [5, 1]),
            #    |  v--v
            ([3, 4, 5, 2, 3, 4, 1, 2, 3, 4], [5, 2]),
            # |  v--v
            ([3, 1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3], [1, 2])
        ])


def test_states_propose_multi_round():
    states = create_proposer(min_n=1, max_n=4,
                             k=3).create_states(max_num_reqs=1)
    assert states._propose(0, np.array([1, 2, 3, 4, 5]), False) == -1
    assert states._propose(0, np.array([1, 2, 3, 4, 5, 1]), False) == 1
    assert states._propose(0, np.array([1, 2, 3, 4, 5, 1, 2]), False) == 2
    assert states._propose(0, np.array([1, 2, 3, 4, 5, 1, 2, 3]), False) == 3
    assert states._propose(0, np.array([1, 2, 3, 4, 5, 1, 2, 3, 4]),
                           False) == 4
    assert states._propose(0, np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
                           False) == 5

    # Reset
    assert states._propose(0, np.array([1, 2, 3, 4, 5, 1, 2, 3]), True) == 3


def test_swap():
    # TODO(Jialin)
    pass
