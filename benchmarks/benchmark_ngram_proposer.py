# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import torch
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
            req_cnt = 96
            states = [None for _ in range(req_cnt)]
            # states = [proposer.create_state() for _ in range(req_cnt)]

            input_collector = TimeCollector(TimeCollector.US)
            output_collector = TimeCollector(TimeCollector.US)
            gc.collect()
            for _ in range(args.num_iteration):
                tokens_tensor = torch.randint(
                    0,
                    20,
                    (96, (args.num_input_token + num_output_token) * 2),
                    dtype=torch.int32,
                    device="cpu",
                )
                tokens = tokens_tensor.numpy()
                # tokens = np.random.randint(
                #     0, 20, (96, (args.num_input_token + num_output_token)*2), dtype=np.int32
                # )
                with input_collector:
                    for req_idx in range(req_cnt):
                        if states[req_idx] is None:
                            states[req_idx] = proposer.create_state()
                        state = states[req_idx]
                        assert state is not None
                        v = proposer.propose(
                            state, tokens[req_idx, : args.num_input_token]
                        )
                        if v is not None:
                            v.tolist()

                with output_collector:
                    for i in range(num_output_token):
                        for req_idx in range(req_cnt):
                            state = states[req_idx]
                            assert state is not None
                            v = proposer.propose(
                                state, tokens[req_idx, : args.num_input_token + i]
                            )
                            if v is not None:
                                v.tolist()
                    for req_idx in range(req_cnt):
                        state = states[req_idx]
                        assert state is not None
                        state.mark_as_dirty()
            rows.append(
                [args.num_input_token, num_output_token, args.min_ngram, max_ngram]
                + input_collector.dump_avg_max()
                + output_collector.dump_avg_max()
            )

            # state = proposer.create_state()

            # input_collector = TimeCollector(TimeCollector.US)
            # output_collector = TimeCollector(TimeCollector.US)
            # gc.collect()
            # for _ in range(args.num_iteration):
            #     tokens = np.random.randint(
            #         0, 20, (args.num_input_token + num_output_token,), dtype=np.int32
            #     )
            #     with input_collector:
            #         proposer.propose(state, tokens[: args.num_input_token])
            #     with output_collector:
            #         for i in range(num_output_token):
            #             proposer.propose(state, tokens[: args.num_input_token + i])
            #         state.mark_as_dirty()
            # rows.append(
            #     [args.num_input_token, num_output_token, args.min_ngram, max_ngram]
            #     + input_collector.dump_avg_max()
            #     + output_collector.dump_avg_max()
            # )

    print(
        tabulate(
            rows,
            headers=[
                "Input Token",
                "Output Token",
                "Min Ngram",
                "Max Ngram",
                "Input\nAvg (us)",
                "Input\nMax (us)",
                "Output\nAvg (us)",
                "Output\nMax (us)",
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
        "--num-input-token",
        type=int,
        default=1500,
        help="Number of tokens for each request",
    )
    parser.add_argument(
        "--num-output-tokens",
        type=int,
        nargs="*",
        default=[1500],
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
