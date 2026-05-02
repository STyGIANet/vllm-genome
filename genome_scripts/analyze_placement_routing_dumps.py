#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
def _running_in_spyder() -> bool:
    return any(
        key in os.environ
        for key in ("SPYDER_ARGS", "SPYDER_KERNEL_ID", "SPY_EXTERNAL_INTERPRETER")
    )


def _maybe_enable_spyder_inline_backend() -> None:
    if not _running_in_spyder():
        return
    backend = str(matplotlib.get_backend()).lower()
    if backend in {"agg", "module://matplotlib.backends.backend_agg"}:
        try:
            matplotlib.use("module://matplotlib_inline.backend_inline")
        except Exception:
            pass


def _backend_supports_show() -> bool:
    backend = str(matplotlib.get_backend()).lower()
    return backend not in {"agg", "module://matplotlib.backends.backend_agg"}


_maybe_enable_spyder_inline_backend()

import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.colors import ListedColormap


import warnings
warnings.filterwarnings("ignore")

#%%
try:
    import pymetis
except ImportError as exc:  # pragma: no cover - hard failure in practice
    raise SystemExit(
        "pymetis is required for this analyzer. Install it with "
        "`uv pip install pymetis`."
    ) from exc


TRIGGER_RE = re.compile(r"trigger_(\d+)")
STEP_RE = re.compile(r"step_(\d+)\.pt$")
SESSION_RE = re.compile(r"session_(\d+)$")
RANK_RE = re.compile(r"rank_(\d+)$")
DEFAULT_TRACE_ROOT = Path(__file__).resolve().parent / "traces"
DEFAULT_NUM_EXPERTS = 64
DEFAULT_NUM_GPUS = 8


DEFAULT_SHOW_PLOTS = _running_in_spyder() or hasattr(sys, "ps1")


@dataclass(frozen=True)
class TriggerDump:
    path: Path
    session_name: str
    rank_name: str
    trigger_name: str
    is_flat_file: bool


@dataclass
class TraceSummary:
    session: str
    rank: str
    trigger: str
    trigger_index: int
    num_steps: int
    num_layers: int
    num_experts: int
    num_gpus: int
    num_nodes: int
    active_nodes: int
    nonzero_edges: int
    total_edge_weight: int
    cut_edges: int
    cut_weight: int
    cut_fraction: float
    active_cluster_sizes: list[int]
    total_cluster_sizes: list[int]
    changed_fraction_vs_prev: float | None
    active_changed_fraction_vs_prev: float | None
    metis_config: dict[str, Any]
    top_cut_edges: list[dict[str, Any]]
    plot_path: str
    summary_path: str


class SkipTrace(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze placement routing dumps emitted by custom EPLB routing "
            "trace capture."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help=(
            "Dataset/root dump directory, session directory, rank directory, "
            "trigger directory, or flat trigger_*.pt file. Defaults to "
            f"{DEFAULT_TRACE_ROOT}."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for summaries and plots. Files are written "
            "directly under this directory with flat names. Defaults to "
            "<session_dir>/analysis when the input path is inside a single "
            "session, otherwise <collection_root>/analysis."
        ),
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=DEFAULT_NUM_EXPERTS,
        help=(
            "Logical experts per layer. Defaults to "
            f"{DEFAULT_NUM_EXPERTS}."
        ),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help=(
            "Total MoE layer count. If omitted, inferred from max observed "
            "layer id + 1 in each trace."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help=(
            "Number of METIS partitions / placement GPUs. Defaults to "
            f"{DEFAULT_NUM_GPUS}."
        ),
    )
    parser.add_argument(
        "--plot-top-nodes",
        type=int,
        default=72,
        help="Maximum number of labeled nodes to show in the graph panel.",
    )
    parser.add_argument(
        "--plot-max-edges",
        type=int,
        default=220,
        help="Maximum number of edges to draw in the graph panel.",
    )
    parser.add_argument(
        "--min-edge-weight",
        type=int,
        default=1,
        help="Minimum edge weight to include in graph plotting.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip writing PNG visualizations and only emit numeric summaries.",
    )
    parser.add_argument(
        "--show-plots",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_PLOTS,
        help=(
            "Display plots with matplotlib/Spyder in addition to saving PNGs. "
            f"Defaults to {'on' if DEFAULT_SHOW_PLOTS else 'off'} in the "
            "current environment."
        ),
    )
    parser.add_argument(
        "--metis-ncuts",
        type=int,
        default=1,
        help=(
            "Number of METIS random restarts. Higher can improve cut quality "
            "at the cost of runtime."
        ),
    )
    parser.add_argument(
        "--metis-ufactor",
        type=int,
        default=None,
        help=(
            "METIS imbalance tolerance. Larger values relax balance and can "
            "sometimes reduce cut weight."
        ),
    )
    parser.add_argument(
        "--metis-seed",
        type=int,
        default=None,
        help="Random seed passed to METIS.",
    )
    parser.add_argument(
        "--metis-recursive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force recursive bisection on/off. Leave unset to use the METIS "
            "default."
        ),
    )
    parser.add_argument(
        "--metis-contiguous",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force contiguous partitions on/off. Leave unset to use the METIS "
            "default."
        ),
    )
    parser.add_argument(
        "--metis-use-vweights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use node_activation_counts as METIS vertex weights. Off by "
            "default."
        ),
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", str(path)))


def resolve_dump_collection_root(path: Path) -> Path:
    path = path.resolve()
    if path.is_file() and TRIGGER_RE.fullmatch(path.stem):
        parent = path.parent
        if SESSION_RE.fullmatch(parent.name):
            return parent.parent
        rank_dir = parent
        session_dir = rank_dir.parent
        if RANK_RE.fullmatch(rank_dir.name) and SESSION_RE.fullmatch(session_dir.name):
            return session_dir.parent
        return parent.parent

    if path.is_dir() and TRIGGER_RE.fullmatch(path.name):
        rank_dir = path.parent
        session_dir = rank_dir.parent
        if RANK_RE.fullmatch(rank_dir.name) and SESSION_RE.fullmatch(session_dir.name):
            return session_dir.parent
        return rank_dir.parent

    if path.is_dir() and RANK_RE.fullmatch(path.name):
        parent = path.parent
        if SESSION_RE.fullmatch(parent.name):
            return parent.parent
        return parent

    if path.is_dir() and SESSION_RE.fullmatch(path.name):
        return path.parent

    return path


def resolve_dump_session_root(path: Path) -> Path | None:
    path = path.resolve()
    if path.is_file() and TRIGGER_RE.fullmatch(path.stem):
        parent = path.parent
        if SESSION_RE.fullmatch(parent.name):
            return parent
        rank_dir = parent
        session_dir = rank_dir.parent
        if RANK_RE.fullmatch(rank_dir.name) and SESSION_RE.fullmatch(session_dir.name):
            return session_dir
        return None

    if path.is_dir() and TRIGGER_RE.fullmatch(path.name):
        rank_dir = path.parent
        session_dir = rank_dir.parent
        if RANK_RE.fullmatch(rank_dir.name) and SESSION_RE.fullmatch(session_dir.name):
            return session_dir
        return None

    if path.is_dir() and RANK_RE.fullmatch(path.name):
        session_dir = path.parent
        if SESSION_RE.fullmatch(session_dir.name):
            return session_dir
        return None

    if path.is_dir() and SESSION_RE.fullmatch(path.name):
        return path

    return None


def resolve_default_output_root(path: Path) -> Path:
    session_root = resolve_dump_session_root(path)
    if session_root is not None:
        return session_root / "analysis"
    return resolve_dump_collection_root(path) / "analysis"


def discover_trigger_dumps(root: Path) -> list[TriggerDump]:
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    candidates: list[TriggerDump] = []

    def maybe_add_file(path: Path) -> None:
        if path.suffix != ".pt":
            return
        stem = path.stem
        if not TRIGGER_RE.fullmatch(stem):
            return
        parent = path.parent
        if SESSION_RE.fullmatch(parent.name):
            candidates.append(
                TriggerDump(
                    path=path,
                    session_name=parent.name,
                    rank_name="",
                    trigger_name=stem,
                    is_flat_file=True,
                )
            )
            return
        rank_dir = parent
        session_dir = rank_dir.parent
        if not rank_dir.name.startswith("rank_"):
            return
        candidates.append(
            TriggerDump(
                path=path,
                session_name=session_dir.name,
                rank_name=rank_dir.name,
                trigger_name=stem,
                is_flat_file=True,
            )
        )

    def maybe_add_dir(path: Path) -> None:
        if not path.is_dir() or not TRIGGER_RE.fullmatch(path.name):
            return
        if not any(path.glob("step_*.pt")):
            return
        rank_dir = path.parent
        session_dir = rank_dir.parent
        if not rank_dir.name.startswith("rank_"):
            return
        candidates.append(
            TriggerDump(
                path=path,
                session_name=session_dir.name,
                rank_name=rank_dir.name,
                trigger_name=path.name,
                is_flat_file=False,
            )
        )

    if root.is_file():
        maybe_add_file(root)
    elif root.is_dir():
        maybe_add_dir(root)
        for path in root.rglob("trigger_*"):
            if path.is_file():
                maybe_add_file(path)
            else:
                maybe_add_dir(path)

    return sorted(
        candidates,
        key=lambda item: natural_sort_key(item.path),
    )


def infer_num_gpus(trigger_dump: TriggerDump) -> int:
    if trigger_dump.rank_name:
        session_dir = trigger_dump.path.parent.parent
    else:
        session_dir = trigger_dump.path.parent
    rank_dirs = [p for p in session_dir.glob("rank_*") if p.is_dir()]
    return max(1, len(rank_dirs))


def load_trigger_payload(trigger_dump: TriggerDump) -> dict[str, Any]:
    if trigger_dump.is_flat_file:
        payload = torch.load(trigger_dump.path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected flat trigger payload format: {trigger_dump.path}"
            )
        return payload

    step_paths = sorted(
        trigger_dump.path.glob("step_*.pt"),
        key=lambda p: int(STEP_RE.match(p.name).group(1)),
    )
    steps: list[dict[str, Any]] = []
    for path in step_paths:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "routing" not in payload:
            raise ValueError(f"Unexpected step payload format: {path}")
        steps.append(payload)

    meta_candidates = sorted(trigger_dump.path.glob("trigger_meta_*.pt"))
    meta = (
        torch.load(meta_candidates[0], map_location="cpu")
        if meta_candidates
        else {}
    )
    return {
        "timestamp_ns": meta.get("timestamp_ns"),
        "trigger_index": meta.get(
            "trigger_index",
            int(TRIGGER_RE.match(trigger_dump.trigger_name).group(1)),
        ),
        "global_rank": meta.get("global_rank"),
        "dp_rank": meta.get("dp_rank"),
        "ep_rank": meta.get("ep_rank"),
        "num_steps": meta.get("step_files", len(steps)),
        "eplb_step_interval": meta.get("eplb_step_interval"),
        "steps": steps,
    }


def collect_trace_dimensions(
    step_payloads: list[dict[str, Any]],
    num_layers_override: int | None,
    num_experts_override: int | None,
) -> tuple[int, int]:
    max_layer_id = -1
    max_expert_id = -1
    for payload in step_payloads:
        routing = payload["routing"]
        for layer_id, captures in routing.items():
            max_layer_id = max(max_layer_id, int(layer_id))
            for capture in captures:
                topk_ids = capture["topk_ids"]
                if topk_ids.numel() == 0:
                    continue
                max_expert_id = max(max_expert_id, int(topk_ids.max().item()))

    num_layers = (
        int(num_layers_override)
        if num_layers_override is not None
        else max_layer_id + 1
    )
    num_experts = (
        int(num_experts_override)
        if num_experts_override is not None
        else max_expert_id + 1
    )
    if num_layers <= 0 or num_experts <= 0:
        raise ValueError("Failed to infer a positive num_layers/num_experts.")
    return num_layers, num_experts


def has_captured_routing_data(step_payloads: list[dict[str, Any]]) -> bool:
    for payload in step_payloads:
        routing = payload.get("routing", {})
        for captures in routing.values():
            for capture in captures:
                topk_ids = capture.get("topk_ids")
                if topk_ids is not None and topk_ids.numel() > 0:
                    return True
    return False


def is_metis_callback_payload(trigger_payload: dict[str, Any]) -> bool:
    routing = trigger_payload.get("routing")
    return isinstance(routing, dict) and "__graph__" in routing


def extract_metis_callback_graph(
    trigger_payload: dict[str, Any],
    num_layers_override: int | None,
    num_experts_override: int | None,
    num_gpus_override: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    routing = trigger_payload["routing"]
    graph_meta = routing["__graph__"]
    coactivation_edges = graph_meta["coactivation_edges"].to(torch.int64).cpu()
    num_layers = int(
        num_layers_override
        if num_layers_override is not None
        else graph_meta["num_layers"]
    )
    num_experts = int(
        num_experts_override
        if num_experts_override is not None
        else graph_meta["num_experts"]
    )
    num_gpus = int(
        num_gpus_override
        if num_gpus_override is not None
        else graph_meta["num_gpus"]
    )
    if graph_meta.get("node_activation_counts") is not None:
        expert_loads = graph_meta["node_activation_counts"].to(
            torch.int64
        ).cpu().reshape(num_layers, num_experts)
    else:
        expert_loads = torch.zeros((num_layers, num_experts), dtype=torch.int64)
        for layer_id, captures in routing.items():
            if not isinstance(layer_id, int) or not captures:
                continue
            layer_tensor = torch.stack(
                [capture["expert_load"].to(torch.int64).cpu() for capture in captures]
            ).sum(dim=0)
            expert_loads[int(layer_id)] = layer_tensor[:num_experts]
    return coactivation_edges, expert_loads, num_layers, num_experts, num_gpus


def _build_metis_inputs(
    coactivation_edges: torch.Tensor,
    num_nodes: int,
) -> tuple[list[int], list[int], list[int]]:
    """Mirror genome_scripts/expert-placement/placement_fns.py."""
    pair_rows, pair_cols = torch.triu_indices(num_nodes, num_nodes, offset=1)
    weights = coactivation_edges.cpu()
    nonzero_mask = weights > 0

    adjacency_lists: list[list[int]] = [[] for _ in range(num_nodes)]
    edge_weight_lists: list[list[int]] = [[] for _ in range(num_nodes)]

    nz_rows = pair_rows[nonzero_mask].tolist()
    nz_cols = pair_cols[nonzero_mask].tolist()
    nz_weights = weights[nonzero_mask].tolist()

    for src, dst, weight in zip(nz_rows, nz_cols, nz_weights):
        w = int(weight)
        adjacency_lists[src].append(dst)
        edge_weight_lists[src].append(w)
        adjacency_lists[dst].append(src)
        edge_weight_lists[dst].append(w)

    xadj = [0]
    adjncy: list[int] = []
    eweights: list[int] = []
    for neighbors, neighbor_weights in zip(adjacency_lists, edge_weight_lists):
        adjncy.extend(neighbors)
        eweights.extend(neighbor_weights)
        xadj.append(len(adjncy))

    return xadj, adjncy, eweights


def canonicalize_membership(membership: list[int], num_parts: int) -> list[int]:
    cluster_to_nodes: dict[int, list[int]] = {i: [] for i in range(num_parts)}
    for node_id, part in enumerate(membership):
        cluster_to_nodes[int(part)].append(node_id)

    ordered_parts = sorted(
        cluster_to_nodes.keys(),
        key=lambda part: (
            cluster_to_nodes[part][0] if cluster_to_nodes[part] else 10**12,
            -len(cluster_to_nodes[part]),
            part,
        ),
    )
    remap = {old: new for new, old in enumerate(ordered_parts)}
    return [remap[int(part)] for part in membership]


def build_metis_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {"ncuts": int(args.metis_ncuts)}
    if args.metis_ufactor is not None:
        config["ufactor"] = int(args.metis_ufactor)
    if args.metis_seed is not None:
        config["seed"] = int(args.metis_seed)
    if args.metis_recursive is not None:
        config["recursive"] = bool(args.metis_recursive)
    if args.metis_contiguous is not None:
        config["contiguous"] = bool(args.metis_contiguous)
    if args.metis_use_vweights:
        config["use_vweights"] = True
    return config


def metis_config_suffix(metis_config: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("ncuts", "ufactor", "seed", "recursive", "contiguous", "use_vweights"):
        if key not in metis_config:
            continue
        value = metis_config[key]
        if isinstance(value, bool):
            value = int(value)
        parts.append(f"{key}{value}")
    return "metis_" + "_".join(parts or ["default"])


def build_compact_coactivation_edges(
    step_payloads: list[dict[str, Any]],
    num_layers: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror GPUModelRunner._build_local_coactivation_edges over all steps."""
    num_nodes = num_layers * num_experts
    num_edges = num_nodes * (num_nodes - 1) // 2
    edge_weights = torch.zeros(num_edges, dtype=torch.int64)
    expert_loads = torch.zeros((num_layers, num_experts), dtype=torch.int64)

    for payload in step_payloads:
        routing = payload["routing"]
        layer_ids = sorted(int(layer_id) for layer_id in routing.keys())
        if not layer_ids:
            continue

        per_layer_ids: list[torch.Tensor] = []
        total_tokens: int | None = None
        top_k: int | None = None

        for layer_id in layer_ids:
            captures = routing[layer_id]
            if not captures:
                per_layer_ids = []
                break
            layer_tensor = torch.cat(
                [capture["topk_ids"] for capture in captures], dim=0
            ).to(torch.int64)
            if layer_tensor.numel() == 0:
                per_layer_ids = []
                break

            if total_tokens is None:
                total_tokens = int(layer_tensor.shape[0])
                top_k = int(layer_tensor.shape[1])
            elif (
                int(layer_tensor.shape[0]) != total_tokens
                or int(layer_tensor.shape[1]) != top_k
            ):
                raise ValueError(
                    "Inconsistent routing tensor shapes across layers within "
                    f"one step: {payload.get('step_in_trigger')}"
                )

            flat = layer_tensor.reshape(-1)
            valid = flat[(flat >= 0) & (flat < num_experts)]
            if valid.numel() > 0:
                expert_loads[layer_id] += torch.bincount(
                    valid,
                    minlength=num_experts,
                )
            per_layer_ids.append(layer_tensor)

        if not per_layer_ids or total_tokens is None or total_tokens == 0:
            continue

        expert_ids = torch.stack(per_layer_ids, dim=1)  # [tokens, layers, top_k]
        layer_offsets = (
            torch.tensor(layer_ids, dtype=torch.int64).view(len(layer_ids), 1)
            * num_experts
        )
        flat_nodes = (expert_ids + layer_offsets).reshape(total_tokens, -1)
        if flat_nodes.shape[1] < 2:
            continue

        row_idx, col_idx = torch.triu_indices(
            flat_nodes.shape[1], flat_nodes.shape[1], offset=1
        )
        src_nodes = flat_nodes[:, row_idx]
        dst_nodes = flat_nodes[:, col_idx]
        lo = torch.minimum(src_nodes, dst_nodes)
        hi = torch.maximum(src_nodes, dst_nodes)
        edge_ids = lo * (2 * num_nodes - lo - 1) // 2 + (hi - lo - 1)
        edge_weights += torch.bincount(edge_ids.reshape(-1), minlength=num_edges)

    return edge_weights, expert_loads


def partition_graph_metis(
    coactivation_edges: torch.Tensor,
    node_weights: torch.Tensor | None,
    num_layers: int,
    num_experts: int,
    num_gpus: int,
    metis_config: dict[str, Any],
) -> list[int]:
    num_nodes = num_layers * num_experts
    xadj, adjncy, eweights = _build_metis_inputs(coactivation_edges, num_nodes)
    if not adjncy:
        return [i % num_gpus for i in range(num_nodes)]
    options_kwargs = {
        key: value
        for key, value in metis_config.items()
        if key not in {"recursive", "contiguous", "use_vweights"}
    }
    options = pymetis.Options(**options_kwargs) if options_kwargs else None
    vweights = None
    if metis_config.get("use_vweights") and node_weights is not None:
        flat_weights = node_weights.reshape(-1).to(torch.int64).cpu()
        flat_weights = torch.clamp(flat_weights, min=1)
        vweights = [int(v) for v in flat_weights.tolist()]
    _, membership = pymetis.part_graph(
        num_gpus,
        xadj=xadj,
        adjncy=adjncy,
        vweights=vweights,
        eweights=eweights,
        recursive=metis_config.get("recursive"),
        contiguous=metis_config.get("contiguous"),
        options=options,
    )
    if len(membership) != num_nodes:
        raise ValueError(
            f"METIS returned {len(membership)} nodes, expected {num_nodes}."
        )
    return canonicalize_membership([int(part) for part in membership], num_gpus)


def compute_cut_stats(
    coactivation_edges: torch.Tensor,
    membership: list[int],
    num_nodes: int,
) -> dict[str, Any]:
    pair_rows, pair_cols = torch.triu_indices(num_nodes, num_nodes, offset=1)
    weights = coactivation_edges.cpu()
    nonzero_mask = weights > 0
    nz_rows = pair_rows[nonzero_mask]
    nz_cols = pair_cols[nonzero_mask]
    nz_weights = weights[nonzero_mask]
    member_tensor = torch.tensor(membership, dtype=torch.int64)
    cut_mask = member_tensor[nz_rows] != member_tensor[nz_cols]
    total_edge_weight = int(nz_weights.sum().item())
    cut_weight = int(nz_weights[cut_mask].sum().item())
    cut_edges = int(cut_mask.sum().item())
    return {
        "nonzero_edges": int(nonzero_mask.sum().item()),
        "total_edge_weight": total_edge_weight,
        "cut_weight": cut_weight,
        "cut_edges": cut_edges,
        "cut_fraction": (
            float(cut_weight) / float(total_edge_weight)
            if total_edge_weight > 0
            else 0.0
        ),
        "pair_rows": nz_rows,
        "pair_cols": nz_cols,
        "pair_weights": nz_weights,
        "cut_mask": cut_mask,
    }


def node_id_to_label(node_id: int, num_experts: int) -> str:
    layer_id = node_id // num_experts
    expert_id = node_id % num_experts
    return f"L{layer_id}E{expert_id}"


def flat_trace_stem(
    output_root: Path,
    session_name: str,
    rank_name: str,
    trigger_name: str,
) -> str:
    if not rank_name:
        if output_root.name == "analysis" and output_root.parent.name == session_name:
            return trigger_name
        return f"{session_name}__{trigger_name}"
    if output_root.name == "analysis" and output_root.parent.name == session_name:
        return f"{rank_name}__{trigger_name}"
    return f"{session_name}__{rank_name}__{trigger_name}"


def compute_node_strengths(
    cut_stats: dict[str, Any],
    expert_loads: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> torch.Tensor:
    num_nodes = num_layers * num_experts
    strengths = expert_loads.reshape(-1).to(torch.float64).clone()
    pair_rows = cut_stats["pair_rows"]
    pair_cols = cut_stats["pair_cols"]
    pair_weights = cut_stats["pair_weights"].to(torch.float64)
    strengths.scatter_add_(0, pair_rows, pair_weights)
    strengths.scatter_add_(0, pair_cols, pair_weights)
    if strengths.numel() != num_nodes:
        raise AssertionError("Node strength length mismatch.")
    return strengths


def top_cut_edges(
    cut_stats: dict[str, Any],
    membership: list[int],
    num_experts: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    pair_rows = cut_stats["pair_rows"]
    pair_cols = cut_stats["pair_cols"]
    pair_weights = cut_stats["pair_weights"]
    cut_mask = cut_stats["cut_mask"]
    if pair_weights.numel() == 0 or not cut_mask.any():
        return []

    indices = torch.argsort(pair_weights[cut_mask], descending=True)[:limit]
    cut_rows = pair_rows[cut_mask][indices]
    cut_cols = pair_cols[cut_mask][indices]
    cut_weights = pair_weights[cut_mask][indices]
    results: list[dict[str, Any]] = []
    for row, col, weight in zip(cut_rows.tolist(), cut_cols.tolist(), cut_weights.tolist()):
        results.append(
            {
                "src_node": row,
                "dst_node": col,
                "src_label": node_id_to_label(row, num_experts),
                "dst_label": node_id_to_label(col, num_experts),
                "src_cluster": int(membership[row]),
                "dst_cluster": int(membership[col]),
                "weight": int(weight),
            }
        )
    return results


def changed_fractions_vs_previous(
    previous_membership: list[int] | None,
    previous_active_mask: torch.Tensor | None,
    membership: list[int],
    active_mask: torch.Tensor,
) -> tuple[float | None, float | None]:
    if previous_membership is None or previous_active_mask is None:
        return None, None
    if len(previous_membership) != len(membership):
        return None, None

    current = torch.tensor(membership, dtype=torch.int64)
    previous = torch.tensor(previous_membership, dtype=torch.int64)
    changed = current != previous
    changed_fraction = float(changed.to(torch.float32).mean().item())

    active_union = active_mask | previous_active_mask
    if not active_union.any():
        return changed_fraction, 0.0
    active_changed_fraction = float(
        changed[active_union].to(torch.float32).mean().item()
    )
    return changed_fraction, active_changed_fraction


def cluster_palette(num_gpus: int) -> list[str]:
    base = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    if num_gpus <= len(base):
        return base[:num_gpus]
    colors = []
    for idx in range(num_gpus):
        hue = idx / max(1, num_gpus)
        colors.append(matplotlib.colors.to_hex(matplotlib.cm.hsv(hue)))
    return colors


def build_plot_subgraph(
    cut_stats: dict[str, Any],
    strengths: torch.Tensor,
    membership: list[int],
    num_experts: int,
    plot_top_nodes: int,
    plot_max_edges: int,
    min_edge_weight: int,
) -> tuple[nx.Graph, dict[int, tuple[float, float]], dict[int, str]]:
    top_nodes = torch.argsort(strengths, descending=True)[:plot_top_nodes].tolist()
    top_node_set = set(int(n) for n in top_nodes if strengths[int(n)] > 0)
    if not top_node_set:
        top_node_set = set(int(n) for n in top_nodes[: min(len(top_nodes), 8)])

    pair_rows = cut_stats["pair_rows"].tolist()
    pair_cols = cut_stats["pair_cols"].tolist()
    pair_weights = cut_stats["pair_weights"].tolist()
    candidate_edges: list[tuple[int, int, int]] = []
    for row, col, weight in zip(pair_rows, pair_cols, pair_weights):
        if weight < min_edge_weight:
            continue
        if row in top_node_set and col in top_node_set:
            candidate_edges.append((int(row), int(col), int(weight)))
    candidate_edges.sort(key=lambda item: item[2], reverse=True)
    candidate_edges = candidate_edges[:plot_max_edges]

    graph = nx.Graph()
    for node_id in sorted(top_node_set):
        graph.add_node(
            node_id,
            cluster=int(membership[node_id]),
            strength=float(strengths[node_id].item()),
            label=node_id_to_label(node_id, num_experts),
        )
    for src, dst, weight in candidate_edges:
        graph.add_edge(
            src,
            dst,
            weight=weight,
            is_cut=membership[src] != membership[dst],
        )

    clusters: dict[int, list[int]] = {}
    for node_id in graph.nodes:
        clusters.setdefault(int(membership[node_id]), []).append(int(node_id))

    positions: dict[int, tuple[float, float]] = {}
    cluster_ids = sorted(clusters.keys())
    radius = max(3.0, 1.8 * len(cluster_ids))
    for cluster_pos, cluster_id in enumerate(cluster_ids):
        theta = 2.0 * math.pi * cluster_pos / max(1, len(cluster_ids))
        center_x = radius * math.cos(theta)
        center_y = radius * math.sin(theta)
        nodes = sorted(clusters[cluster_id], key=lambda n: (-graph.nodes[n]["strength"], n))
        local_radius = 1.2 + 0.12 * len(nodes)
        for local_idx, node_id in enumerate(nodes):
            local_theta = 2.0 * math.pi * local_idx / max(1, len(nodes))
            positions[node_id] = (
                center_x + local_radius * math.cos(local_theta),
                center_y + local_radius * math.sin(local_theta),
            )

    labels = {node_id: graph.nodes[node_id]["label"] for node_id in graph.nodes}
    return graph, positions, labels


def render_trace_plot(
    trace_name: str,
    summary: TraceSummary,
    membership: list[int],
    expert_loads: torch.Tensor,
    cut_stats: dict[str, Any],
    strengths: torch.Tensor,
    output_path: Path,
    plot_top_nodes: int,
    plot_max_edges: int,
    min_edge_weight: int,
    show_plots: bool,
) -> None:
    num_layers = summary.num_layers
    num_experts = summary.num_experts
    num_gpus = summary.num_gpus
    palette = cluster_palette(num_gpus)

    graph, positions, labels = build_plot_subgraph(
        cut_stats=cut_stats,
        strengths=strengths,
        membership=membership,
        num_experts=num_experts,
        plot_top_nodes=plot_top_nodes,
        plot_max_edges=plot_max_edges,
        min_edge_weight=min_edge_weight,
    )

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1.6], height_ratios=[1.0, 1.0])
    ax_text = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[:, 1])
    ax_graph = fig.add_subplot(grid[:, 2])
    ax_bar = fig.add_subplot(grid[1, 0])

    fig.suptitle(trace_name, fontsize=16, fontweight="bold")

    summary_lines = [
        f"steps: {summary.num_steps}",
        f"nodes: {summary.active_nodes}/{summary.num_nodes} active",
        f"edges: {summary.nonzero_edges}",
        f"total edge weight: {summary.total_edge_weight}",
        f"cut edges: {summary.cut_edges}",
        f"cut weight: {summary.cut_weight}",
        f"cut fraction: {summary.cut_fraction:.3f}",
        f"active cluster sizes: {summary.active_cluster_sizes}",
    ]
    if summary.changed_fraction_vs_prev is not None:
        summary_lines.append(
            f"changed vs prev: {summary.changed_fraction_vs_prev:.3f}"
        )
    if summary.active_changed_fraction_vs_prev is not None:
        summary_lines.append(
            f"active changed vs prev: {summary.active_changed_fraction_vs_prev:.3f}"
        )
    if summary.top_cut_edges:
        summary_lines.append("")
        summary_lines.append("top cut edges:")
        for edge in summary.top_cut_edges[:6]:
            summary_lines.append(
                f"{edge['src_label']} - {edge['dst_label']} : {edge['weight']}"
            )
    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#f8f8f8",
            "edgecolor": "#dddddd",
        },
    )

    matrix = torch.full((num_experts, num_layers), -1, dtype=torch.int64)
    loads_matrix = expert_loads.t().contiguous()
    for node_id, part in enumerate(membership):
        layer_id = node_id // num_experts
        expert_id = node_id % num_experts
        if expert_loads[layer_id, expert_id] > 0:
            matrix[expert_id, layer_id] = int(part)

    cmap = ListedColormap(["#efefef"] + palette)
    heat_values = matrix.numpy() + 1
    heat = ax_heat.imshow(
        heat_values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
    )
    ax_heat.set_title("Full layer/expert cluster assignment")
    ax_heat.set_xlabel("Layer")
    ax_heat.set_ylabel("Expert")
    ax_heat.set_xticks(list(range(num_layers)))
    ax_heat.set_xticklabels([str(i) for i in range(num_layers)], fontsize=7)
    yticks = sorted(set([0, num_experts // 4, num_experts // 2, (3 * num_experts) // 4, num_experts - 1]))
    ax_heat.set_yticks(yticks)
    ax_heat.set_yticklabels([str(i) for i in yticks], fontsize=8)
    colorbar = fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04)
    colorbar.set_ticks(list(range(num_gpus + 1)))
    colorbar.set_ticklabels(["unused"] + [f"C{i}" for i in range(num_gpus)])

    active_sizes = summary.active_cluster_sizes
    ax_bar.bar(
        [f"C{i}" for i in range(num_gpus)],
        active_sizes,
        color=palette,
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax_bar.set_title("Active nodes per cluster")
    ax_bar.set_ylabel("nodes")
    ax_bar.grid(axis="y", alpha=0.2)

    ax_graph.set_title("Top active-node subgraph")
    ax_graph.axis("off")
    if graph.number_of_nodes() > 0:
        node_colors = [palette[int(graph.nodes[n]["cluster"])] for n in graph.nodes]
        node_sizes = [
            120 + 18 * math.sqrt(max(graph.nodes[n]["strength"], 1.0))
            for n in graph.nodes
        ]
        edge_weights = [graph.edges[e]["weight"] for e in graph.edges]
        max_edge_weight = max(edge_weights) if edge_weights else 1
        cut_edges = [e for e in graph.edges if graph.edges[e]["is_cut"]]
        internal_edges = [e for e in graph.edges if not graph.edges[e]["is_cut"]]
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax_graph,
            edgelist=internal_edges,
            width=[
                0.6 + 4.0 * graph.edges[e]["weight"] / max_edge_weight
                for e in internal_edges
            ],
            edge_color="#8d99ae",
            alpha=0.28,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax_graph,
            edgelist=cut_edges,
            width=[
                0.8 + 4.8 * graph.edges[e]["weight"] / max_edge_weight
                for e in cut_edges
            ],
            edge_color="#d1495b",
            alpha=0.6,
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax_graph,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="#222222",
            linewidths=0.7,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            ax=ax_graph,
            labels=labels,
            font_size=7,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
            },
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show_plots and _backend_supports_show():
        plt.show()
    plt.close(fig)


def analyze_trigger_dump(
    trigger_dump: TriggerDump,
    output_root: Path,
    num_layers_override: int | None,
    num_experts_override: int | None,
    num_gpus_override: int | None,
    metis_config: dict[str, Any],
    plot_top_nodes: int,
    plot_max_edges: int,
    min_edge_weight: int,
    skip_plots: bool,
    show_plots: bool,
    previous_by_rank: dict[tuple[str, str], tuple[list[int], torch.Tensor]],
) -> TraceSummary:
    trigger_payload = load_trigger_payload(trigger_dump)
    if is_metis_callback_payload(trigger_payload):
        (
            coactivation_edges,
            expert_loads,
            num_layers,
            num_experts,
            num_gpus,
        ) = extract_metis_callback_graph(
            trigger_payload,
            num_layers_override=num_layers_override,
            num_experts_override=num_experts_override,
            num_gpus_override=num_gpus_override,
        )
        num_steps = 1
    else:
        step_payloads = list(trigger_payload["steps"])
        if not has_captured_routing_data(step_payloads):
            raise SkipTrace(
                "No captured routing data in trigger payload "
                f"{trigger_dump.path}"
            )
        num_layers, num_experts = collect_trace_dimensions(
            step_payloads,
            num_layers_override=num_layers_override,
            num_experts_override=num_experts_override,
        )
        num_gpus = num_gpus_override or infer_num_gpus(trigger_dump)
        if num_gpus < 1:
            raise ValueError("Invalid inferred num_gpus for trigger_dir")

        coactivation_edges, expert_loads = build_compact_coactivation_edges(
            step_payloads,
            num_layers=num_layers,
            num_experts=num_experts,
        )
        num_steps = int(trigger_payload.get("num_steps", len(step_payloads)))

    if num_gpus < 1:
        raise ValueError("Invalid inferred num_gpus for trigger_dir")

    membership = partition_graph_metis(
        coactivation_edges,
        node_weights=expert_loads,
        num_layers=num_layers,
        num_experts=num_experts,
        num_gpus=num_gpus,
        metis_config=metis_config,
    )
    num_nodes = num_layers * num_experts
    cut_stats = compute_cut_stats(
        coactivation_edges=coactivation_edges,
        membership=membership,
        num_nodes=num_nodes,
    )
    strengths = compute_node_strengths(
        cut_stats=cut_stats,
        expert_loads=expert_loads,
        num_layers=num_layers,
        num_experts=num_experts,
    )
    active_mask = expert_loads.reshape(-1) > 0
    active_cluster_sizes = torch.bincount(
        torch.tensor(membership, dtype=torch.int64)[active_mask],
        minlength=num_gpus,
    ).tolist()
    total_cluster_sizes = torch.bincount(
        torch.tensor(membership, dtype=torch.int64),
        minlength=num_gpus,
    ).tolist()

    session_name = trigger_dump.session_name
    rank_name = trigger_dump.rank_name
    trigger_name = trigger_dump.trigger_name
    key = (session_name, rank_name)
    previous_membership, previous_active_mask = previous_by_rank.get(key, (None, None))
    changed_fraction, active_changed_fraction = changed_fractions_vs_previous(
        previous_membership=previous_membership,
        previous_active_mask=previous_active_mask,
        membership=membership,
        active_mask=active_mask,
    )
    previous_by_rank[key] = (membership, active_mask.clone())

    flat_stem = flat_trace_stem(
        output_root=output_root,
        session_name=session_name,
        rank_name=rank_name,
        trigger_name=trigger_name,
    )
    config_suffix = metis_config_suffix(metis_config)
    plot_path = output_root / f"{flat_stem}__{config_suffix}__graph_analysis.png"
    summary_path = output_root / f"{flat_stem}__{config_suffix}__summary.json"
    summary = TraceSummary(
        session=session_name,
        rank=rank_name,
        trigger=trigger_name,
        trigger_index=int(TRIGGER_RE.match(trigger_name).group(1)),
        num_steps=num_steps,
        num_layers=num_layers,
        num_experts=num_experts,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        active_nodes=int(active_mask.sum().item()),
        nonzero_edges=cut_stats["nonzero_edges"],
        total_edge_weight=cut_stats["total_edge_weight"],
        cut_edges=cut_stats["cut_edges"],
        cut_weight=cut_stats["cut_weight"],
        cut_fraction=cut_stats["cut_fraction"],
        active_cluster_sizes=[int(v) for v in active_cluster_sizes],
        total_cluster_sizes=[int(v) for v in total_cluster_sizes],
        changed_fraction_vs_prev=changed_fraction,
        active_changed_fraction_vs_prev=active_changed_fraction,
        metis_config=dict(metis_config),
        top_cut_edges=top_cut_edges(
            cut_stats=cut_stats,
            membership=membership,
            num_experts=num_experts,
        ),
        plot_path=str(plot_path),
        summary_path=str(summary_path),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    if not skip_plots:
        render_trace_plot(
            trace_name=f"{trigger_name}",
            summary=summary,
            membership=membership,
            expert_loads=expert_loads,
            cut_stats=cut_stats,
            strengths=strengths,
            output_path=plot_path,
            plot_top_nodes=plot_top_nodes,
            plot_max_edges=plot_max_edges,
            min_edge_weight=min_edge_weight,
            show_plots=show_plots,
        )

    return summary


def print_summary(summary: TraceSummary) -> None:
    changed = (
        "n/a"
        if summary.changed_fraction_vs_prev is None
        else f"{summary.changed_fraction_vs_prev:.3f}"
    )
    active_changed = (
        "n/a"
        if summary.active_changed_fraction_vs_prev is None
        else f"{summary.active_changed_fraction_vs_prev:.3f}"
    )
    prefix = f"{summary.session} {summary.trigger}"
    if summary.rank:
        prefix = f"{summary.session} {summary.rank} {summary.trigger}"
    print(
        f"{prefix} | "
        f"steps={summary.num_steps} active_nodes={summary.active_nodes}/{summary.num_nodes} "
        f"edges={summary.nonzero_edges} cut_weight={summary.cut_weight} "
        f"cut_frac={summary.cut_fraction:.3f} changed={changed} "
        f"active_changed={active_changed} "
        f"metis={metis_config_suffix(summary.metis_config)}"
    )


def write_csv(summaries: list[TraceSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary in summaries:
        row = asdict(summary)
        row["active_cluster_sizes"] = json.dumps(summary.active_cluster_sizes)
        row["total_cluster_sizes"] = json.dumps(summary.total_cluster_sizes)
        row["metis_config"] = json.dumps(summary.metis_config, sort_keys=True)
        row["top_cut_edges"] = json.dumps(summary.top_cut_edges)
        rows.append(row)

    if not rows:
        return

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    trigger_dumps = discover_trigger_dumps(args.input_dir)
    if not trigger_dumps:
        raise SystemExit(
            "No trigger dumps found under "
            f"{args.input_dir}. Expected trigger_*.pt files or old trigger_* directories."
        )

    metis_config = build_metis_config(args)
    previous_by_rank: dict[tuple[str, str], tuple[list[int], torch.Tensor]] = {}
    summaries: list[TraceSummary] = []
    output_roots_by_summary_path: dict[str, Path] = {}
    skipped = 0

    print(f"Discovered {len(trigger_dumps)} trigger traces under {args.input_dir}")
    print(
        "Using METIS config:",
        ", ".join(f"{k}={v}" for k, v in metis_config.items()),
    )
    for trigger_dump in trigger_dumps:
        try:
            output_root = args.out_dir or resolve_default_output_root(
                trigger_dump.path
            )
            summary = analyze_trigger_dump(
                trigger_dump=trigger_dump,
                output_root=output_root,
                num_layers_override=args.num_layers,
                num_experts_override=args.num_experts,
                num_gpus_override=args.num_gpus,
                metis_config=metis_config,
                plot_top_nodes=args.plot_top_nodes,
                plot_max_edges=args.plot_max_edges,
                min_edge_weight=args.min_edge_weight,
                skip_plots=args.skip_plots,
                show_plots=args.show_plots,
                previous_by_rank=previous_by_rank,
            )
        except SkipTrace as exc:
            skipped += 1
            print(f"Skipping {trigger_dump.path}: {exc}")
            continue
        summaries.append(summary)
        output_roots_by_summary_path[summary.summary_path] = output_root
        print_summary(summary)

    summaries_by_output_root: dict[Path, list[TraceSummary]] = defaultdict(list)
    for summary in summaries:
        output_root = output_roots_by_summary_path.get(summary.summary_path)
        if output_root is None:
            output_root = args.out_dir or resolve_default_output_root(
                Path(summary.summary_path)
            )
        summaries_by_output_root[output_root].append(summary)

    for output_root, root_summaries in summaries_by_output_root.items():
        write_csv(
            root_summaries,
            output_root / f"trace_summary__{metis_config_suffix(metis_config)}.csv",
        )
    if skipped:
        print(f"Skipped {skipped} empty trigger trace(s)")
    if summaries_by_output_root:
        print("Wrote summaries to:")
        for output_root in sorted(summaries_by_output_root):
            print(f"  {output_root}")


if __name__ == "__main__":
    main()
