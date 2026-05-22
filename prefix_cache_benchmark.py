#!/usr/bin/env python3
"""Trace-driven KV/SSM prefix-cache hit-rate simulator.

This script can run lightweight cache models plus the repository's Marconi and
vLLM+ radix-cache implementations. The lightweight models capture:

* KV/full-attention cache hits require all prefix blocks from the beginning.
* SSM cache hits reuse the largest cached recurrent-state checkpoint.
* Hybrid hits are the intersection of KV and SSM hits under a configurable
  memory split.

Input traces are JSONL files with at least:
  input_tokens: list[int]
  output_tokens: list[int]
Optional fields such as session_id, turn_id, and ts are preserved only for
ordering/debugging.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from radix_cache_hybrid import RadixCache as MarconiRadixCache
from radix_cache_vllm import RadixCache as VLLMPlusRadixCache
from utils import get_kvs_size, get_mamba_state_size


BlockHash = tuple[int | tuple[int, ...], tuple[int, ...]]


def gb_to_bytes(value: float) -> int:
    return int(value * 1_000_000_000)


def block_hashes(tokens: list[int], block_size: int) -> list[BlockHash]:
    """Return rolling hashes for full token blocks.

    vLLM hashes a block using the parent/prefix hash plus the current block's
    tokens. A nested tuple is enough for deterministic simulation; no real KV
    data is stored.
    """
    hashes: list[BlockHash] = []
    parent_hash: int | BlockHash = 0
    num_full_blocks = len(tokens) // block_size
    for block_id in range(num_full_blocks):
        start = block_id * block_size
        block_tokens = tuple(tokens[start : start + block_size])
        current_hash = (parent_hash, block_tokens)
        hashes.append(current_hash)
        parent_hash = current_hash
    return hashes


def load_request_trace(trace_path: Path) -> list[dict]:
    requests: list[dict] = []
    with trace_path.open() as f:
        for line in f:
            if line.strip():
                requests.append(json.loads(line))
    return sorted(requests, key=lambda req: req.get("ts", 0.0))


def synthetic_trace() -> list[dict]:
    return [
        {
            "session_id": 0,
            "turn_id": 0,
            "ts": 0.0,
            "input_tokens": [1, 2, 3, 4],
            "output_tokens": [5, 6, 7, 8],
        },
        {
            "session_id": 0,
            "turn_id": 1,
            "ts": 1.0,
            "input_tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "output_tokens": [11, 12],
        },
        {
            "session_id": 1,
            "turn_id": 0,
            "ts": 2.0,
            "input_tokens": [20, 21, 22, 23],
            "output_tokens": [24, 25, 26, 27],
        },
        {
            "session_id": 1,
            "turn_id": 1,
            "ts": 3.0,
            "input_tokens": [20, 21, 22, 23, 24, 25, 26, 27],
            "output_tokens": [28],
        },
    ]


class LRUSet:
    def __init__(self) -> None:
        self._items: OrderedDict[BlockHash, None] = OrderedDict()

    def __contains__(self, key: BlockHash) -> bool:
        return key in self._items

    def touch(self, key: BlockHash) -> None:
        if key in self._items:
            self._items.move_to_end(key)
        else:
            self._items[key] = None

    def evict_lru(self) -> BlockHash | None:
        if not self._items:
            return None
        key, _ = self._items.popitem(last=False)
        return key

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class CacheStats:
    request_count: int = 0
    request_hits: int = 0
    total_input_tokens: int = 0
    total_hit_tokens: int = 0
    evictions: int = 0

    def record(self, input_len: int, hit_len: int) -> None:
        self.request_count += 1
        self.total_input_tokens += input_len
        self.total_hit_tokens += hit_len
        if hit_len > 0:
            self.request_hits += 1

    @property
    def request_hit_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.request_hits / self.request_count

    @property
    def token_hit_rate(self) -> float:
        if self.total_input_tokens == 0:
            return 0.0
        return self.total_hit_tokens / self.total_input_tokens

    @property
    def avg_hit_tokens_per_hit(self) -> float:
        if self.request_hits == 0:
            return 0.0
        return self.total_hit_tokens / self.request_hits


class BlockPrefixCache:
    """Full-attention KV prefix cache with contiguous block-hit semantics."""

    def __init__(self, capacity_bytes: int, block_size: int, bytes_per_block: int):
        self.capacity_bytes = capacity_bytes
        self.block_size = block_size
        self.bytes_per_block = bytes_per_block
        self.blocks = LRUSet()
        self.evictions = 0

    @property
    def max_blocks(self) -> int:
        if self.bytes_per_block <= 0:
            return 0
        return max(0, self.capacity_bytes // self.bytes_per_block)

    @property
    def used_bytes(self) -> int:
        return len(self.blocks) * self.bytes_per_block

    def hit_length(self, tokens: list[int]) -> int:
        hit_blocks = 0
        for block_hash in block_hashes(tokens, self.block_size):
            if block_hash not in self.blocks:
                break
            self.blocks.touch(block_hash)
            hit_blocks += 1
        return hit_blocks * self.block_size

    def insert(self, tokens: list[int]) -> None:
        for block_hash in block_hashes(tokens, self.block_size):
            self.blocks.touch(block_hash)
            while len(self.blocks) > self.max_blocks:
                if self.blocks.evict_lru() is None:
                    break
                self.evictions += 1


class SSMCheckpointCache:
    """SSM prefix cache where one checkpoint stores state after N tokens."""

    def __init__(
        self,
        capacity_bytes: int,
        checkpoint_interval: int,
        bytes_per_checkpoint: int,
    ):
        self.capacity_bytes = capacity_bytes
        self.checkpoint_interval = checkpoint_interval
        self.bytes_per_checkpoint = bytes_per_checkpoint
        self.checkpoints = LRUSet()
        self.evictions = 0

    @property
    def max_checkpoints(self) -> int:
        if self.bytes_per_checkpoint <= 0:
            return 0
        return max(0, self.capacity_bytes // self.bytes_per_checkpoint)

    @property
    def used_bytes(self) -> int:
        return len(self.checkpoints) * self.bytes_per_checkpoint

    def hit_length(self, tokens: list[int], max_hit_length: int | None = None) -> int:
        bounded_len = len(tokens) if max_hit_length is None else min(len(tokens), max_hit_length)
        max_blocks = bounded_len // self.checkpoint_interval
        hashes = block_hashes(tokens[: max_blocks * self.checkpoint_interval], self.checkpoint_interval)
        for block_id in range(len(hashes) - 1, -1, -1):
            block_hash = hashes[block_id]
            if block_hash in self.checkpoints:
                self.checkpoints.touch(block_hash)
                return (block_id + 1) * self.checkpoint_interval
        return 0

    def insert(self, tokens: list[int]) -> None:
        for checkpoint_hash in block_hashes(tokens, self.checkpoint_interval):
            self.checkpoints.touch(checkpoint_hash)
            while len(self.checkpoints) > self.max_checkpoints:
                if self.checkpoints.evict_lru() is None:
                    break
                self.evictions += 1


@dataclass(frozen=True)
class BenchmarkConfig:
    capacity_gb: float
    kv_cache_fraction: float
    kv_block_size: int
    ssm_checkpoint_interval: int
    num_attn_layers: int
    num_ssm_layers: int
    num_mlp_layers: int
    d: int
    n: int

    @property
    def total_capacity_bytes(self) -> int:
        return gb_to_bytes(self.capacity_gb)

    @property
    def kv_capacity_bytes(self) -> int:
        return int(self.total_capacity_bytes * self.kv_cache_fraction)

    @property
    def ssm_capacity_bytes(self) -> int:
        return self.total_capacity_bytes - self.kv_capacity_bytes

    @property
    def kv_block_bytes(self) -> int:
        return self.num_attn_layers * get_kvs_size(self.kv_block_size, self.d)

    @property
    def ssm_checkpoint_bytes(self) -> int:
        return self.num_ssm_layers * get_mamba_state_size(self.d, self.n)


def run_radix_cache_strategy(
    requests: Iterable[dict],
    config: BenchmarkConfig,
    cache_type: str,
    cache_cls: type,
    **cache_kwargs,
) -> dict:
    """Run Marconi/vLLM+ radix-cache implementations on this benchmark trace."""
    radix_cache = cache_cls(
        capacity_bytes=config.total_capacity_bytes,
        use_logical_ts=True,
        num_ssm_layers=config.num_ssm_layers,
        num_attn_layers=config.num_attn_layers,
        num_mlp_layers=config.num_mlp_layers,
        d=config.d,
        n=config.n,
        **cache_kwargs,
    )

    # Marconi's adaptive tuning path can print during eviction; keep CSV output
    # machine-readable when stdout is the benchmark sink.
    with contextlib.redirect_stdout(io.StringIO()):
        for request_id, request in enumerate(requests):
            input_tokens = request["input_tokens"]
            output_tokens = request.get("output_tokens", [])
            radix_cache.match_prefix(input_tokens)
            radix_cache.insert(
                token_ids=input_tokens + output_tokens,
                state_at_leaf=request.get("session_id", request_id),
                state_at_branchoff=request.get("session_id", request_id),
            )

    request_hit_rate, token_hit_rate, total_mamba_flop_savings, total_attn_flop_savings, total_mlp_flop_savings = (
        radix_cache.get_cache_stats(verbose=False)
    )
    num_cached_mamba_states, num_cached_kv_tokens = radix_cache.get_num_cached_tokens()
    mamba_state_bytes, kv_state_bytes = radix_cache.get_state_size()
    total_input_tokens = sum(row[1] for row in radix_cache.request_history)
    total_hit_tokens = sum(row[2] for row in radix_cache.request_history)
    request_hits = sum(row[0] for row in radix_cache.request_history)

    return {
        "cache_type": cache_type,
        "capacity_gb": config.capacity_gb,
        "kv_cache_fraction": "",
        "kv_capacity_gb": "",
        "ssm_capacity_gb": "",
        "kv_block_size": getattr(radix_cache, "block_size", ""),
        "ssm_checkpoint_interval": "",
        "num_requests": len(radix_cache.request_history),
        "request_hit_rate": request_hit_rate,
        "token_hit_rate": token_hit_rate,
        "avg_hit_tokens_per_hit": 0.0 if request_hits == 0 else total_hit_tokens / request_hits,
        "total_input_tokens": total_input_tokens,
        "total_hit_tokens": total_hit_tokens,
        "kv_used_gb": kv_state_bytes / 1_000_000_000,
        "ssm_used_gb": mamba_state_bytes / 1_000_000_000,
        "kv_block_bytes": config.kv_block_bytes,
        "ssm_checkpoint_bytes": config.ssm_checkpoint_bytes,
        "kv_cached_blocks": "",
        "ssm_cached_checkpoints": num_cached_mamba_states,
        "evictions": "",
        "total_flops_saved": total_mamba_flop_savings + total_attn_flop_savings + total_mlp_flop_savings,
        "num_cached_kv_tokens": num_cached_kv_tokens,
    }


def run_benchmark(requests: Iterable[dict], config: BenchmarkConfig) -> dict:
    kv_cache = BlockPrefixCache(
        capacity_bytes=config.kv_capacity_bytes,
        block_size=config.kv_block_size,
        bytes_per_block=config.kv_block_bytes,
    )
    ssm_cache = SSMCheckpointCache(
        capacity_bytes=config.ssm_capacity_bytes,
        checkpoint_interval=config.ssm_checkpoint_interval,
        bytes_per_checkpoint=config.ssm_checkpoint_bytes,
    )

    kv_stats = CacheStats()
    ssm_stats = CacheStats()
    hybrid_stats = CacheStats()

    for request in requests:
        input_tokens = request["input_tokens"]
        output_tokens = request.get("output_tokens", [])
        input_len = len(input_tokens)

        kv_hit = kv_cache.hit_length(input_tokens)
        ssm_hit = ssm_cache.hit_length(input_tokens)
        hybrid_hit = ssm_cache.hit_length(input_tokens, max_hit_length=kv_hit)

        kv_stats.record(input_len, kv_hit)
        ssm_stats.record(input_len, ssm_hit)
        hybrid_stats.record(input_len, hybrid_hit)

        all_tokens = input_tokens + output_tokens
        kv_cache.insert(all_tokens)
        ssm_cache.insert(all_tokens)

    kv_stats.evictions = kv_cache.evictions
    ssm_stats.evictions = ssm_cache.evictions
    hybrid_stats.evictions = kv_cache.evictions + ssm_cache.evictions

    rows = []
    for cache_type, stats in (
        ("kv_only", kv_stats),
        ("ssm_only", ssm_stats),
        ("hybrid_intersection", hybrid_stats),
    ):
        rows.append(
            {
                "cache_type": cache_type,
                "capacity_gb": config.capacity_gb,
                "kv_cache_fraction": config.kv_cache_fraction,
                "kv_capacity_gb": config.kv_capacity_bytes / 1_000_000_000,
                "ssm_capacity_gb": config.ssm_capacity_bytes / 1_000_000_000,
                "kv_block_size": config.kv_block_size,
                "ssm_checkpoint_interval": config.ssm_checkpoint_interval,
                "num_requests": stats.request_count,
                "request_hit_rate": stats.request_hit_rate,
                "token_hit_rate": stats.token_hit_rate,
                "avg_hit_tokens_per_hit": stats.avg_hit_tokens_per_hit,
                "total_input_tokens": stats.total_input_tokens,
                "total_hit_tokens": stats.total_hit_tokens,
                "kv_used_gb": kv_cache.used_bytes / 1_000_000_000,
                "ssm_used_gb": ssm_cache.used_bytes / 1_000_000_000,
                "kv_block_bytes": config.kv_block_bytes,
                "ssm_checkpoint_bytes": config.ssm_checkpoint_bytes,
                "kv_cached_blocks": len(kv_cache.blocks),
                "ssm_cached_checkpoints": len(ssm_cache.checkpoints),
                "evictions": stats.evictions,
                "total_flops_saved": "",
                "num_cached_kv_tokens": "",
            }
        )
    return {"rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KV and SSM prefix-cache hit rates on JSONL traces."
    )
    parser.add_argument("--trace", type=Path, help="Path to ShareGPT-style JSONL trace.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a tiny built-in trace instead of reading --trace.",
    )
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--capacity-gb", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--kv-cache-fraction",
        nargs="+",
        type=float,
        default=[0.5],
        help="Fraction of total capacity assigned to KV cache; remainder goes to SSM.",
    )
    parser.add_argument("--kv-block-size", nargs="+", type=int, default=[32])
    parser.add_argument("--ssm-checkpoint-interval", nargs="+", type=int, default=[32])
    parser.add_argument("--num-attn-layers", type=int, default=4)
    parser.add_argument("--num-ssm-layers", type=int, default=24)
    parser.add_argument("--num-mlp-layers", type=int, default=28)
    parser.add_argument("--d", type=int, default=4096)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=["simple", "vllm_plus", "marconi", "all"],
        default=["simple", "vllm_plus", "marconi"],
        help=(
            "Cache strategies to run. simple emits kv_only/ssm_only/hybrid_intersection; "
            "vllm_plus uses token-block checkpointing; marconi uses Marconi V2."
        ),
    )
    parser.add_argument("--marconi-eff-weight", type=float, default=0.0)
    parser.add_argument("--marconi-bootstrap-multiplier", type=int, default=5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.self_test and args.trace is None:
        raise SystemExit("Pass --trace PATH or --self-test.")
    for fraction in args.kv_cache_fraction:
        if not 0.0 <= fraction <= 1.0:
            raise SystemExit("--kv-cache-fraction values must be between 0 and 1.")
    for value in args.kv_block_size + args.ssm_checkpoint_interval:
        if value <= 0:
            raise SystemExit("Block sizes and checkpoint intervals must be positive.")
    for capacity_gb in args.capacity_gb:
        if not math.isfinite(capacity_gb) or capacity_gb < 0:
            raise SystemExit("--capacity-gb values must be finite and non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    requests = synthetic_trace() if args.self_test else load_request_trace(args.trace)
    strategies = {"simple", "vllm_plus", "marconi"} if "all" in args.strategy else set(args.strategy)
    all_rows: list[dict] = []
    emitted_radix_configs: set[tuple[str, float, int]] = set()

    for capacity_gb in args.capacity_gb:
        for kv_cache_fraction in args.kv_cache_fraction:
            for kv_block_size in args.kv_block_size:
                for ssm_checkpoint_interval in args.ssm_checkpoint_interval:
                    config = BenchmarkConfig(
                        capacity_gb=capacity_gb,
                        kv_cache_fraction=kv_cache_fraction,
                        kv_block_size=kv_block_size,
                        ssm_checkpoint_interval=ssm_checkpoint_interval,
                        num_attn_layers=args.num_attn_layers,
                        num_ssm_layers=args.num_ssm_layers,
                        num_mlp_layers=args.num_mlp_layers,
                        d=args.d,
                        n=args.n,
                    )
                    if "simple" in strategies:
                        result = run_benchmark(requests, config)
                        all_rows.extend(result["rows"])
                    if "vllm_plus" in strategies:
                        radix_key = ("vllm_plus", capacity_gb, kv_block_size)
                        if radix_key not in emitted_radix_configs:
                            emitted_radix_configs.add(radix_key)
                            all_rows.append(
                                run_radix_cache_strategy(
                                    requests,
                                    config,
                                    cache_type="vllm_plus",
                                    cache_cls=VLLMPlusRadixCache,
                                    block_size=kv_block_size,
                                )
                            )
                    if "marconi" in strategies:
                        radix_key = ("marconi", capacity_gb, 0)
                        if radix_key not in emitted_radix_configs:
                            emitted_radix_configs.add(radix_key)
                            all_rows.append(
                                run_radix_cache_strategy(
                                    requests,
                                    config,
                                    cache_type="marconi_v2",
                                    cache_cls=MarconiRadixCache,
                                    evict_policy_version=2,
                                    eff_weight=args.marconi_eff_weight,
                                    bootstrap_multiplier=args.marconi_bootstrap_multiplier,
                                )
                            )

    fieldnames = list(all_rows[0].keys()) if all_rows else []
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
