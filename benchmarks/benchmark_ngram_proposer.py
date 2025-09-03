# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time

import numpy as np
from tabulate import tabulate

from benchmark_utils import TimeCollector
from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.utils import FlexibleArgumentParser
from vllm.v1.spec_decode.ngram_proposer import NgramProposer


def main(args):
    max_model_len = (
        args.num_input_token + max(args.num_output_tokens) + args.num_spec_token
    )
    model_config = ModelConfig(
        model="openai/gpt-oss-120b",
        task="generate",
        max_model_len=max_model_len,
        tokenizer="openai/gpt-oss-120b",
        tokenizer_mode="auto",
        dtype="auto",
        seed=None,
        trust_remote_code=False,
    )

    rows = []
    for num_output_token in args.num_output_tokens:
        for max_ngram in args.max_ngram:
            proposer = NgramProposer(
                vllm_config=VllmConfig(
                    model_config=model_config,
                    speculative_config=SpeculativeConfig(
                        prompt_lookup_min=args.min_ngram,
                        prompt_lookup_max=max_ngram,
                        num_speculative_tokens=args.num_spec_token,
                        method="ngram",
                    ),
                )
            )
            input_collector = TimeCollector(TimeCollector.MS)
            output_collector = TimeCollector(TimeCollector.MS)
            for _ in range(args.num_iteration):
                gc.collect()
                tokens = np.random.randint(
                    0,
                    20,
                    (args.num_req, args.num_input_token + num_output_token),
                    dtype=np.int32,
                )
                token_cnts = np.full(
                    (args.num_req,), args.num_input_token, dtype=np.int32
                )
                req_indices = np.arange(args.num_req, dtype=np.int32)
                states = proposer.create_states(args.num_req)

                with input_collector:
                    states.init(tokens, token_cnts, req_indices)

                time_ns = 0
                for i in range(num_output_token):
                    token_cnts = np.full(
                        (args.num_req,), args.num_input_token + i, dtype=np.int32
                    )
                    start_time_ns = time.monotonic_ns()
                    states.bulk_propose(tokens, token_cnts)
                    time_ns += time.monotonic_ns() - start_time_ns
                output_collector.collect(time_ns)
            rows.append(
                [
                    args.num_req,
                    args.num_input_token,
                    num_output_token,
                    args.min_ngram,
                    max_ngram,
                ]
                + input_collector.dump_avg_max()
                + output_collector.dump_avg_max()
            )
    print(
        tabulate(
            rows,
            headers=[
                "Num Req",
                "Input Token",
                "Output Token",
                "Min Ngram",
                "Max Ngram",
                "Input\nAvg (ms)",
                "Input\nMax (ms)",
                "Output\nAvg (ms)",
                "Output\nMax (ms)",
            ],
            tablefmt="grid",
            floatfmt=".3f",
        )
    )


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark the performance of N-gram speculative decode drafting"
    )
    parser.add_argument(
        "--num-iteration",
        type=int,
        default=1,
        help="Number of iterations to run to stablize final data readings",
    )
    parser.add_argument(
        "--num-req", type=int, default=96, help="Number of requests in the batch"
    )
    parser.add_argument(
        "--num-input-token",
        type=int,
        default=1500,
        help="Number of tokens for each request",
    )
    parser.add_argument(
        "--num-output-tokens",
        type=int,
        nargs="*",
        default=[1, 150, 1500, 8000],
        help="Number of tokens for each request",
    )
    parser.add_argument(
        "--min-ngram",
        type=int,
        default=5,
        help="Minimum n-gram to match",
    )
    parser.add_argument(
        "--max-ngram",
        type=int,
        nargs="*",
        default=[7, 10],
        help="Maximum n-gram to match",
    )
    parser.add_argument(
        "--num-spec-token",
        type=int,
        default=3,
        help="Number of speculative tokens to generate",
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
