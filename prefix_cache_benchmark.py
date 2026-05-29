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
import copy
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
from utils import (
    _key_match,
    get_attn_flops,
    get_kvs_size,
    get_linear_attention_state_size,
    get_linear_attn_flops,
    get_mlp_flops,
)


QWEN35_4B_MODEL_NAME = "Qwen3.5-4B"
QWEN35_4B_NUM_HIDDEN_LAYERS = 32
QWEN35_4B_NUM_ATTN_LAYERS = 8
QWEN35_4B_NUM_LINEAR_ATTN_LAYERS = 24
QWEN35_4B_HIDDEN_SIZE = 2560
QWEN35_4B_INTERMEDIATE_SIZE = 9216
QWEN35_4B_NUM_ATTN_HEADS = 16
QWEN35_4B_NUM_KV_HEADS = 4
QWEN35_4B_HEAD_DIM = 256
QWEN35_4B_LINEAR_NUM_KEY_HEADS = 16
QWEN35_4B_LINEAR_NUM_VALUE_HEADS = 32
QWEN35_4B_LINEAR_KEY_HEAD_DIM = 128
QWEN35_4B_LINEAR_VALUE_HEAD_DIM = 128
QWEN35_4B_LINEAR_CONV_KERNEL = 4

LEGACY_SIMPLE_FIELDNAMES = [
    "cache_type",
    "capacity_gb",
    "kv_cache_fraction",
    "kv_capacity_gb",
    "ssm_capacity_gb",
    "kv_block_size",
    "ssm_checkpoint_interval",
    "num_requests",
    "request_hit_rate",
    "token_hit_rate",
    "avg_hit_tokens_per_hit",
    "total_input_tokens",
    "total_hit_tokens",
    "kv_used_gb",
    "ssm_used_gb",
    "kv_block_bytes",
    "ssm_checkpoint_bytes",
    "kv_cached_blocks",
    "ssm_cached_checkpoints",
    "evictions",
]


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


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [0.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


class VLLMMarconiEvictionRadixCache(VLLMPlusRadixCache):
    """vLLM+ radix cache with Marconi-style utility eviction over leaf blocks.

    This keeps radix_cache_vllm's fixed-block insertion and prefix matching.
    The only changed behavior is victim selection: instead of pure LRU over
    leaves, choose the leaf with the lowest recency/efficiency utility.
    """

    def __init__(
        self,
        *args,
        eff_weight: float = 0.0,
        bootstrap_multiplier: int = 5,
        candidate_eff_weights: Iterable[float] | None = None,
        enable_tuning: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eff_weight = eff_weight
        self.evictions = 0
        self.enable_tuning = enable_tuning
        self.bootstrap_multiplier = bootstrap_multiplier
        self.candidate_eff_weights = list(candidate_eff_weights or [x / 10 for x in range(21)])
        self.request_history_windowed: list[list[list[int] | None]] = []
        self.bootstrap_window_size: int | None = None
        self.num_reqs_before_eviction: int | None = None
        self.tuned = False
        self.tree_snapshot = None

    def match_prefix(self, input_token_ids, actually_inserting=True):
        if actually_inserting and self.enable_tuning and self.tree_snapshot is None:
            self.tree_snapshot = copy.deepcopy(self)

        prefix_token_ids, nodes_accessed, prefix_len = super().match_prefix(
            input_token_ids,
            actually_inserting=actually_inserting,
        )
        if actually_inserting and self.enable_tuning and not self.tuned:
            self.request_history_windowed.append([input_token_ids, None])
        return prefix_token_ids, nodes_accessed, prefix_len

    def insert(self, token_ids, *args, **kwargs):
        if self.enable_tuning and not self.tuned and self.request_history_windowed:
            latest_request = self.request_history_windowed[-1]
            if latest_request[1] is None:
                input_token_ids = latest_request[0]
                input_len = 0
                if input_token_ids is not None:
                    input_len = _key_match(input_token_ids, token_ids)
                latest_request[1] = token_ids[input_len:]

        super().insert(token_ids, *args, **kwargs)

        if (
            self.enable_tuning
            and not self.tuned
            and self.bootstrap_window_size is not None
            and len(self.request_history_windowed) >= self.bootstrap_window_size
        ):
            self.eff_weight = self._tune_eff_weight()
            self.tuned = True

    def _leaf_flops_efficiency(self, node) -> float:
        child_len = len(node.value)
        total_len = len(node.get_all_token_ids())
        parent_len = total_len - child_len
        mamba = self.num_ssm_layers * self._ssm_flops(child_len)
        attn = self.num_attn_layers * (
            self._attn_flops(total_len) - self._attn_flops(parent_len)
        )
        mlp = self.num_mlp_layers * (
            self._mlp_flops(total_len) - self._mlp_flops(parent_len)
        )
        memory = (
            self.num_ssm_layers * self._ssm_state_size()
            + self.num_attn_layers * self._kv_size(child_len)
        )
        return (mamba + attn + mlp) / memory

    def evict(self, bytes_to_remove):
        if self.use_logical_ts:
            self.logical_ts += self.time_increment

        if self.enable_tuning and self.num_reqs_before_eviction is None:
            self.num_reqs_before_eviction = len(self.request_history)
            self.bootstrap_window_size = self.bootstrap_multiplier * self.num_reqs_before_eviction

        bytes_evicted = 0
        while bytes_evicted < bytes_to_remove:
            leaves = [leaf for leaf in self._collect_leaves() if leaf != self.root_node]
            if not leaves:
                break

            current_ts = self.logical_ts if self.use_logical_ts else 0
            recency_scores = [
                1.0 / max(1, current_ts - leaf.last_access_time)
                for leaf in leaves
            ]
            efficiency_scores = [self._leaf_flops_efficiency(leaf) for leaf in leaves]

            normalized_recency = normalize(recency_scores)
            normalized_efficiency = normalize(efficiency_scores)
            utility_scores = [
                recency + self.eff_weight * efficiency
                for recency, efficiency in zip(normalized_recency, normalized_efficiency)
            ]
            victim = leaves[utility_scores.index(min(utility_scores))]
            bytes_evicted += (
                self.num_ssm_layers * self._ssm_state_size()
                + self.num_attn_layers * self._kv_size(len(victim.value))
            )
            self._delete_leaf(victim)
            self.evictions += 1

    def _tune_eff_weight(self) -> float:
        if self.tree_snapshot is None:
            return self.eff_weight

        best_weight = self.eff_weight
        best_token_hit_rate = -1.0
        replay_requests = [
            (input_tokens, output_tokens or [])
            for input_tokens, output_tokens in self.request_history_windowed
            if input_tokens is not None
        ]

        for weight in self.candidate_eff_weights:
            replay_cache = copy.deepcopy(self.tree_snapshot)
            replay_cache.enable_tuning = False
            replay_cache.eff_weight = weight
            replay_cache.request_history = []
            for input_tokens, output_tokens in replay_requests:
                replay_cache.match_prefix(input_tokens)
                replay_cache.insert(input_tokens + output_tokens)
            _, token_hit_rate, *_ = replay_cache.get_cache_stats(verbose=False)
            if token_hit_rate > best_token_hit_rate:
                best_token_hit_rate = token_hit_rate
                best_weight = weight

        return best_weight


@dataclass(frozen=True)
class BenchmarkConfig:
    capacity_gb: float
    kv_cache_fraction: float
    kv_block_size: int
    ssm_checkpoint_interval: int
    model_name: str
    num_attn_layers: int
    num_ssm_layers: int
    num_mlp_layers: int
    d: int
    n: int
    intermediate_size: int
    gated_mlp: bool
    num_attn_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel: int

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
        return self.num_attn_layers * get_kvs_size(
            self.kv_block_size,
            self.d,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
        )

    @property
    def ssm_checkpoint_bytes(self) -> int:
        return self.num_ssm_layers * get_linear_attention_state_size(
            num_value_heads=self.linear_num_value_heads,
            key_head_dim=self.linear_key_head_dim,
            value_head_dim=self.linear_value_head_dim,
            conv_dim=(
                2 * self.linear_num_key_heads * self.linear_key_head_dim
                + self.linear_num_value_heads * self.linear_value_head_dim
            ),
            conv_kernel=self.linear_conv_kernel,
        )

    @property
    def vllm_plus_block_bytes(self) -> int:
        return self.kv_block_bytes + self.ssm_checkpoint_bytes


def flops_saved(
    hit_len: int,
    config: BenchmarkConfig,
) -> float:
    return (
        config.num_ssm_layers
        * get_linear_attn_flops(
            l=hit_len,
            d=config.d,
            key_dim=config.linear_num_key_heads * config.linear_key_head_dim,
            value_dim=config.linear_num_value_heads * config.linear_value_head_dim,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_kernel=config.linear_conv_kernel,
        )
        + config.num_attn_layers
        * get_attn_flops(
            l=hit_len,
            d=config.d,
            q_dim=config.num_attn_heads * config.head_dim,
            kv_dim=config.num_key_value_heads * config.head_dim,
        )
        + config.num_mlp_layers
        * get_mlp_flops(
            l=hit_len,
            d=config.d,
            intermediate_size=config.intermediate_size,
            gated=config.gated_mlp,
        )
    )


def row_from_stats(
    cache_type: str,
    config: BenchmarkConfig,
    stats: CacheStats,
    kv_used_bytes: int,
    ssm_used_bytes: int,
    kv_cached_blocks: int | str,
    ssm_cached_checkpoints: int | str,
    total_flops_saved: float | str = "",
    num_cached_kv_tokens: int | str = "",
    kv_cache_fraction: float | str = "",
    kv_capacity_gb: float | str = "",
    ssm_capacity_gb: float | str = "",
    ssm_checkpoint_interval: int | str = "",
    tuned_eff_weight: float | str = "",
) -> dict:
    return {
        "cache_type": cache_type,
        "model_name": config.model_name,
        "capacity_gb": config.capacity_gb,
        "kv_cache_fraction": kv_cache_fraction,
        "kv_capacity_gb": kv_capacity_gb,
        "ssm_capacity_gb": ssm_capacity_gb,
        "kv_block_size": config.kv_block_size,
        "ssm_checkpoint_interval": ssm_checkpoint_interval,
        "num_requests": stats.request_count,
        "request_hit_rate": stats.request_hit_rate,
        "token_hit_rate": stats.token_hit_rate,
        "avg_hit_tokens_per_hit": stats.avg_hit_tokens_per_hit,
        "total_input_tokens": stats.total_input_tokens,
        "total_hit_tokens": stats.total_hit_tokens,
        "kv_used_gb": kv_used_bytes / 1_000_000_000,
        "ssm_used_gb": ssm_used_bytes / 1_000_000_000,
        "kv_block_bytes": config.kv_block_bytes,
        "ssm_checkpoint_bytes": config.ssm_checkpoint_bytes,
        "kv_cached_blocks": kv_cached_blocks,
        "ssm_cached_checkpoints": ssm_cached_checkpoints,
        "evictions": stats.evictions,
        "total_flops_saved": total_flops_saved,
        "num_cached_kv_tokens": num_cached_kv_tokens,
        "tuned_eff_weight": tuned_eff_weight,
    }


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
        intermediate_size=config.intermediate_size,
        gated_mlp=config.gated_mlp,
        num_attn_heads=config.num_attn_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        linear_num_key_heads=config.linear_num_key_heads,
        linear_num_value_heads=config.linear_num_value_heads,
        linear_key_head_dim=config.linear_key_head_dim,
        linear_value_head_dim=config.linear_value_head_dim,
        linear_conv_kernel=config.linear_conv_kernel,
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
        "model_name": config.model_name,
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
        "tuned_eff_weight": getattr(radix_cache, "eff_weight", ""),
    }


def run_vllm_marconi_eviction_strategy(
    requests: Iterable[dict],
    config: BenchmarkConfig,
    bootstrap_multiplier: int,
    candidate_eff_weights: list[float],
) -> dict:
    cache = VLLMMarconiEvictionRadixCache(
        capacity_bytes=config.total_capacity_bytes,
        block_size=config.kv_block_size,
        num_ssm_layers=config.num_ssm_layers,
        num_attn_layers=config.num_attn_layers,
        num_mlp_layers=config.num_mlp_layers,
        d=config.d,
        n=config.n,
        intermediate_size=config.intermediate_size,
        gated_mlp=config.gated_mlp,
        num_attn_heads=config.num_attn_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        linear_num_key_heads=config.linear_num_key_heads,
        linear_num_value_heads=config.linear_num_value_heads,
        linear_key_head_dim=config.linear_key_head_dim,
        linear_value_head_dim=config.linear_value_head_dim,
        linear_conv_kernel=config.linear_conv_kernel,
        eff_weight=0.0,
        bootstrap_multiplier=bootstrap_multiplier,
        candidate_eff_weights=candidate_eff_weights,
    )
    stats = CacheStats()

    with contextlib.redirect_stdout(io.StringIO()):
        for request in requests:
            input_tokens = request["input_tokens"]
            output_tokens = request.get("output_tokens", [])
            cache.match_prefix(input_tokens)
            cache.insert(input_tokens + output_tokens)

    stats.evictions = cache.evictions
    request_hit_rate, token_hit_rate, total_mamba_flop_savings, total_attn_flop_savings, total_mlp_flop_savings = (
        cache.get_cache_stats(verbose=False)
    )
    num_cached_mamba_states, num_cached_kv_tokens = cache.get_num_cached_tokens()
    mamba_state_bytes, kv_state_bytes = cache.get_state_size()
    total_input_tokens = sum(row[1] for row in cache.request_history)
    total_hit_tokens = sum(row[2] for row in cache.request_history)
    request_hits = sum(row[0] for row in cache.request_history)
    stats.request_count = len(cache.request_history)
    stats.request_hits = request_hits
    stats.total_input_tokens = total_input_tokens
    stats.total_hit_tokens = total_hit_tokens
    return row_from_stats(
        cache_type="vllm_marconi_eviction",
        config=config,
        stats=stats,
        kv_used_bytes=kv_state_bytes,
        ssm_used_bytes=mamba_state_bytes,
        kv_cached_blocks="",
        ssm_cached_checkpoints=num_cached_mamba_states,
        total_flops_saved=total_mamba_flop_savings + total_attn_flop_savings + total_mlp_flop_savings,
        num_cached_kv_tokens=num_cached_kv_tokens,
        tuned_eff_weight=cache.eff_weight,
    )


def run_vllm_block_lru_strategy(requests: Iterable[dict], config: BenchmarkConfig) -> dict:
    cache = BlockPrefixCache(
        capacity_bytes=config.total_capacity_bytes,
        block_size=config.kv_block_size,
        bytes_per_block=config.vllm_plus_block_bytes,
    )
    stats = CacheStats()
    total_flops_saved = 0.0

    for request in requests:
        input_tokens = request["input_tokens"]
        output_tokens = request.get("output_tokens", [])
        hit_len = cache.hit_length(input_tokens)
        stats.record(len(input_tokens), hit_len)
        total_flops_saved += flops_saved(hit_len, config)
        cache.insert(input_tokens + output_tokens)

    stats.evictions = cache.evictions
    return row_from_stats(
        cache_type="vllm_block_lru",
        config=config,
        stats=stats,
        kv_used_bytes=len(cache.blocks) * config.kv_block_bytes,
        ssm_used_bytes=len(cache.blocks) * config.ssm_checkpoint_bytes,
        kv_cached_blocks=len(cache.blocks),
        ssm_cached_checkpoints=len(cache.blocks),
        total_flops_saved=total_flops_saved,
        num_cached_kv_tokens=len(cache.blocks) * config.kv_block_size,
    )


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
                "model_name": config.model_name,
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
                "tuned_eff_weight": "",
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
    parser.add_argument("--model-name", default=QWEN35_4B_MODEL_NAME)
    parser.add_argument("--num-attn-layers", type=int, default=QWEN35_4B_NUM_ATTN_LAYERS)
    parser.add_argument("--num-ssm-layers", type=int, default=QWEN35_4B_NUM_LINEAR_ATTN_LAYERS)
    parser.add_argument("--num-mlp-layers", type=int, default=QWEN35_4B_NUM_HIDDEN_LAYERS)
    parser.add_argument("--d", type=int, default=QWEN35_4B_HIDDEN_SIZE)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=QWEN35_4B_INTERMEDIATE_SIZE)
    parser.add_argument("--gated-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-attn-heads", type=int, default=QWEN35_4B_NUM_ATTN_HEADS)
    parser.add_argument("--num-key-value-heads", type=int, default=QWEN35_4B_NUM_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=QWEN35_4B_HEAD_DIM)
    parser.add_argument("--linear-num-key-heads", type=int, default=QWEN35_4B_LINEAR_NUM_KEY_HEADS)
    parser.add_argument("--linear-num-value-heads", type=int, default=QWEN35_4B_LINEAR_NUM_VALUE_HEADS)
    parser.add_argument("--linear-key-head-dim", type=int, default=QWEN35_4B_LINEAR_KEY_HEAD_DIM)
    parser.add_argument("--linear-value-head-dim", type=int, default=QWEN35_4B_LINEAR_VALUE_HEAD_DIM)
    parser.add_argument("--linear-conv-kernel", type=int, default=QWEN35_4B_LINEAR_CONV_KERNEL)
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=[
            "simple",
            "vllm_block_lru",
            "vllm_plus",
            "vllm_marconi_eviction",
            "marconi",
            "all",
        ],
        default=["simple", "vllm_block_lru", "vllm_marconi_eviction", "vllm_plus", "marconi"],
        help=(
            "Cache strategies to run. simple emits kv_only/ssm_only/hybrid_intersection; "
            "vllm_block_lru and vllm_marconi_eviction use the same full-block "
            "vLLM-style simulator with different eviction policies; vllm_plus uses "
            "the repository radix baseline; marconi uses Marconi V2."
        ),
    )
    parser.add_argument("--marconi-eff-weight", type=float, default=0.0)
    parser.add_argument("--marconi-bootstrap-multiplier", type=int, default=5)
    parser.add_argument(
        "--vllm-marconi-bootstrap-multiplier",
        type=int,
        default=5,
        help="Bootstrap window multiplier for tuning alpha in vllm_marconi_eviction.",
    )
    parser.add_argument(
        "--vllm-marconi-eff-weights",
        nargs="+",
        type=float,
        default=[x / 10 for x in range(21)],
        help="Candidate alpha values for vllm_marconi_eviction grid search.",
    )
    parser.add_argument(
        "--legacy-simple-csv",
        action="store_true",
        help="Emit the older simple-strategy CSV columns used by prefix_cache_benchmark_sharegpt.csv.",
    )
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
    if args.vllm_marconi_bootstrap_multiplier <= 0:
        raise SystemExit("--vllm-marconi-bootstrap-multiplier must be positive.")
    if args.legacy_simple_csv and set(args.strategy) != {"simple"}:
        raise SystemExit("--legacy-simple-csv is only compatible with --strategy simple.")
    for weight in args.vllm_marconi_eff_weights:
        if not math.isfinite(weight) or weight < 0:
            raise SystemExit("--vllm-marconi-eff-weights values must be finite and non-negative.")
    positive_model_values = {
        "--num-attn-layers": args.num_attn_layers,
        "--num-ssm-layers": args.num_ssm_layers,
        "--num-mlp-layers": args.num_mlp_layers,
        "--d": args.d,
        "--n": args.n,
        "--intermediate-size": args.intermediate_size,
        "--num-attn-heads": args.num_attn_heads,
        "--num-key-value-heads": args.num_key_value_heads,
        "--head-dim": args.head_dim,
        "--linear-num-key-heads": args.linear_num_key_heads,
        "--linear-num-value-heads": args.linear_num_value_heads,
        "--linear-key-head-dim": args.linear_key_head_dim,
        "--linear-value-head-dim": args.linear_value_head_dim,
        "--linear-conv-kernel": args.linear_conv_kernel,
    }
    for flag, value in positive_model_values.items():
        if value <= 0:
            raise SystemExit(f"{flag} must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    requests = synthetic_trace() if args.self_test else load_request_trace(args.trace)
    strategies = (
        {"simple", "vllm_block_lru", "vllm_plus", "vllm_marconi_eviction", "marconi"}
        if "all" in args.strategy
        else set(args.strategy)
    )
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
                        model_name=args.model_name,
                        num_attn_layers=args.num_attn_layers,
                        num_ssm_layers=args.num_ssm_layers,
                        num_mlp_layers=args.num_mlp_layers,
                        d=args.d,
                        n=args.n,
                        intermediate_size=args.intermediate_size,
                        gated_mlp=args.gated_mlp,
                        num_attn_heads=args.num_attn_heads,
                        num_key_value_heads=args.num_key_value_heads,
                        head_dim=args.head_dim,
                        linear_num_key_heads=args.linear_num_key_heads,
                        linear_num_value_heads=args.linear_num_value_heads,
                        linear_key_head_dim=args.linear_key_head_dim,
                        linear_value_head_dim=args.linear_value_head_dim,
                        linear_conv_kernel=args.linear_conv_kernel,
                    )
                    if "simple" in strategies:
                        result = run_benchmark(requests, config)
                        all_rows.extend(result["rows"])
                    if "vllm_block_lru" in strategies:
                        radix_key = ("vllm_block_lru", capacity_gb, kv_block_size)
                        if radix_key not in emitted_radix_configs:
                            emitted_radix_configs.add(radix_key)
                            all_rows.append(run_vllm_block_lru_strategy(requests, config))
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
                    if "vllm_marconi_eviction" in strategies:
                        radix_key = ("vllm_marconi_eviction", capacity_gb, kv_block_size)
                        if radix_key not in emitted_radix_configs:
                            emitted_radix_configs.add(radix_key)
                            all_rows.append(
                                run_vllm_marconi_eviction_strategy(
                                    requests,
                                    config,
                                    bootstrap_multiplier=args.vllm_marconi_bootstrap_multiplier,
                                    candidate_eff_weights=args.vllm_marconi_eff_weights,
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

    if args.legacy_simple_csv:
        all_rows = [{field: row[field] for field in LEGACY_SIMPLE_FIELDNAMES} for row in all_rows]
        fieldnames = LEGACY_SIMPLE_FIELDNAMES
    else:
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
