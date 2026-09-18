from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import tempfile

import pytest
import torch
import torch.distributed as dist

from experiments.success_path_distributed import SuccessPathDistributedContext


def test_single_process_context_preserves_the_whole_global_batch() -> None:
    context = SuccessPathDistributedContext()

    assert context.local_indices(5) == (0, 1, 2, 3, 4)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([2.0])
    context.sum_gradients([parameter])
    assert parameter.grad.tolist() == [2.0]
    assert context.gather_indexed_rows([(0, {"sample": "a"})]) == [
        {"sample": "a"}
    ]


def test_rank_partitions_are_disjoint_and_restore_exact_global_order() -> None:
    contexts = [
        SuccessPathDistributedContext(rank=rank, world_size=1)
        for rank in (0,)
    ]
    assert contexts[0].local_indices(7) == tuple(range(7))

    partitions = [tuple(range(rank, 7, 2)) for rank in range(2)]
    assert partitions == [(0, 2, 4, 6), (1, 3, 5)]
    assert sorted(index for values in partitions for index in values) == list(range(7))


def _gloo_worker(rank: int, init_file: str, queue: mp.Queue) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        context = SuccessPathDistributedContext.from_initialized_process_group()
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        parameter.grad = torch.tensor([float(rank + 1)], dtype=torch.float32)
        context.sum_gradients([parameter])
        rows = context.gather_indexed_rows(
            [(index, {"index": index}) for index in context.local_indices(4)]
        )
        queue.put((rank, float(parameter.grad.item()), rows))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_two_rank_context_sums_gradients_and_reassembles_rows() -> None:
    multiprocessing = mp.get_context("spawn")
    queue = multiprocessing.Queue()
    with tempfile.TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "process_group")
        processes = [
            multiprocessing.Process(target=_gloo_worker, args=(rank, init_file, queue))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        results = [queue.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0

    for _rank, gradient, rows in results:
        assert gradient == pytest.approx(3.0)
        assert rows == [{"index": index} for index in range(4)]
