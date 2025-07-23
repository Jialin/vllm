# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time
from collections import defaultdict
from random import randint
from typing import Optional

from tabulate import tabulate

from vllm.utils import FlexibleArgumentParser
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashWithGroupId


class Metric:
    def __init__(self) -> None:
        self.cnt: int = 0
        self.sum_v: int = 0
        self.max_v: Optional[int] = None

    def update(self, v: int) -> None:
        self.cnt += 1
        self.sum_v += v
        if self.max_v is None:
            self.max_v = v
        else:
            self.max_v = max(self.max_v, v)

    def avg_v(self) -> float:
        return self.sum_v * 1.0 / self.cnt


def main(args):
    MAX_SIZE_LOG10 = 6
    MAX_SIZE = 10**MAX_SIZE_LOG10
    hashes = [
        BlockHashWithGroupId(
            block_hash=BlockHash(
                hash_value=randint(-(2**31), 2**31),
                token_ids=tuple(randint(0, 128000) for _ in range(16)),
            ),
            group_id=0,
        )
        for i in range(MAX_SIZE)
    ]
    unknown_hashes = [
        BlockHashWithGroupId(
            block_hash=BlockHash(
                hash_value=randint(-(2**31), 2**31),
                token_ids=tuple(randint(0, 128000) for _ in range(16)),
            ),
            group_id=0,
        )
        for i in range(MAX_SIZE)
    ]

    rows = []
    for size_log10 in range(MAX_SIZE_LOG10 + 1):
        size = 10**size_log10
        cache: defaultdict[BlockHashWithGroupId, dict[int, int]] = defaultdict(dict)

        gc.collect()
        insert_key_metric: Metric = Metric()
        for i in range(size):
            t1 = time.monotonic_ns()
            cache[hashes[i]] = {i: i}
            t2 = time.monotonic_ns()
            insert_key_metric.update(t2 - t1)

        gc.collect()
        del_key_metric: Metric = Metric()
        for i in range(size):
            t1 = time.monotonic_ns()
            cache.pop(hashes[i], None)
            t2 = time.monotonic_ns()
            del_key_metric.update(t2 - t1)

        gc.collect()
        insert_after_del_key_metric: Metric = Metric()
        for i in range(size):
            t1 = time.monotonic_ns()
            cache[hashes[i]] = {i: i}
            t2 = time.monotonic_ns()
            insert_after_del_key_metric.update(t2 - t1)

        gc.collect()
        known_key_metric: Metric = Metric()
        unknown_key_metric: Metric = Metric()
        for i in range(args.num_iteration):
            idx = i % size
            t1 = time.monotonic_ns()
            cache.get(hashes[idx])
            t2 = time.monotonic_ns()
            cache.get(unknown_hashes[idx])
            t3 = time.monotonic_ns()
            known_key_metric.update(t2 - t1)
            unknown_key_metric.update(t3 - t2)

        if (
            insert_key_metric.max_v is not None
            and del_key_metric.max_v is not None
            and insert_after_del_key_metric.max_v is not None
            and known_key_metric.max_v is not None
            and unknown_key_metric.max_v is not None
        ):
            rows.append(
                [
                    known_key_metric.cnt,
                    size,
                    insert_key_metric.avg_v() / 1000.0,
                    insert_key_metric.max_v / 1000.0,
                    del_key_metric.avg_v() / 1000.0,
                    del_key_metric.max_v / 1000.0,
                    insert_after_del_key_metric.avg_v() / 1000.0,
                    insert_after_del_key_metric.max_v / 1000.0,
                    known_key_metric.avg_v() / 1000.0,
                    known_key_metric.max_v / 1000.0,
                    unknown_key_metric.avg_v() / 1000.0,
                    unknown_key_metric.max_v / 1000.0,
                ]
            )

    print(
        tabulate(
            rows,
            headers=[
                "Iterations",
                "N",
                "Insert\nAvg (us)",
                "Insert\nMax (us)",
                "Del\nAvg (us)",
                "Del\nMax (us)",
                "Insert after Del\nAvg (us)",
                "Insert after Del\nMax (us)",
                "Known Lookup\nAvg (us)",
                "Known Lookup\nMax (us)",
                "Unknown Lookup\nAvg (us)",
                "Unknown Lookup\nMax (us)",
            ],
            tablefmt="grid",
            floatfmt=".6f",
        )
    )


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark the performance of BlockPool for KV Cache."
    )
    parser.add_argument("--num-gpu-blocks", type=int, default=100000)
    parser.add_argument(
        "--num-iteration",
        type=int,
        default=10000,
        help="Number of iterations to run to stablize final data readings",
    )
    parser.add_argument(
        "--allocate-blocks",
        type=int,
        nargs="*",
        default=[10, 50, 100, 500, 1000],
        help="Number of blocks to allocate",
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
