#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import seaborn as sns

import matplotlib
import numpy as np
import pandas as pd


# Example:

# analyze_moe_dispatch_traffic.py traffic/deepseek-moe-16b-chat-mmlu-all-test

# analyze_moe_dispatch_traffic.py traffic/deepseek-moe-16b-chat-mmlu-all-test --plot-pass 100 --plot-pass-window-size 64 --plot-max-layers 12


#%%
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


DEFAULT_TRACE_ROOT = Path(__file__).resolve().parent / "traffic-metis/deepseek-moe-16b-chat-mmlu-all-test"
DEFAULT_SHOW_PLOTS = _running_in_spyder() or hasattr(sys, "ps1")


@dataclass(frozen=True)
class TrafficRecord:
    event_idx: int
    batch_signature: str | None
    forward_pass_idx: int | None
    moe_layer_ordinal_in_pass: int | None
    layer_id: int
    layer_name: str
    source_global_rank: int
    source_ep_rank: int
    dest_global_ranks: list[int]
    dest_ep_ranks: list[int]
    dest_token_counts: list[int]
    num_tokens: int
    top_k: int
    num_token_copies: int
    ep_world_size: int
    enable_eplb: bool
    expert_placement_strategy: str | None
    trace_file: str


@dataclass
class TrafficSummary:
    num_trace_files: int
    num_records: int
    num_forward_passes: int
    num_layers: int
    ep_world_size: int
    grouping_mode: str
    trace_root: str
    pickle_path: str
    plotted_pass_idx: int | None
    plotted_total_path: str | None
    plotted_full_pass_path: str | None
    plotted_overall_total_path: str | None
    plotted_overall_layers_path: str | None
    per_layer_plot_dir: str | None
    per_pass_layer_plot_dir: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MoE dispatch traffic traces and build EP-to-EP traffic "
            "matrices per layer and per forward pass."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help=(
            "Directory containing traffic_*.jsonl files, or a parent "
            f"directory to search recursively. Defaults to {DEFAULT_TRACE_ROOT}."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to <input_dir>/analysis when traffic "
            "files are directly under the input dir, otherwise to the nearest "
            "directory containing the discovered traffic files plus /analysis."
        ),
    )
    parser.add_argument(
        "--plot-pass",
        type=int,
        default=100,
        help=(
            "Optional forward pass index to plot. By default, pass-level "
            "plots are disabled because they can be misleading under DP "
            "serving."
        ),
    )
    parser.add_argument(
        "--plot-max-layers",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of layer heatmaps to include in the "
            "full-forward-pass figure."
        ),
    )
    parser.add_argument(
        "--plot-pass-window-size",
        type=int,
        default=None,
        help=(
            "Deprecated. Pass-level plotting is exact for the selected pass "
            "only and this flag is ignored."
        ),
    )
    parser.add_argument(
        "--show-plots",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_PLOTS,
        help=(
            "Display plots in matplotlib/Spyder in addition to saving PNGs. "
            f"Defaults to {'on' if DEFAULT_SHOW_PLOTS else 'off'} here."
        ),
    )
    return parser.parse_args()


def discover_trace_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    if root.is_file():
        return [root] if root.name.startswith("traffic_") and root.suffix == ".jsonl" else []

    direct = sorted(root.glob("traffic_*.jsonl"))
    if direct:
        return direct
    return sorted(root.rglob("traffic_*.jsonl"))


def resolve_default_output_root(input_dir: Path, trace_files: list[Path]) -> Path:
    if input_dir.is_dir() and any(path.parent == input_dir for path in trace_files):
        return input_dir / "analysis"
    first_parent = trace_files[0].parent
    return first_parent / "analysis"


def load_records(trace_files: list[Path]) -> list[TrafficRecord]:
    records: list[TrafficRecord] = []
    for path in trace_files:
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                try:
                    records.append(
                        TrafficRecord(
                            event_idx=int(payload["event_idx"]),
                            batch_signature=(
                                None
                                if payload.get("batch_signature") is None
                                else str(payload["batch_signature"])
                            ),
                            forward_pass_idx=(
                                None
                                if payload.get("forward_pass_idx") is None
                                else int(payload["forward_pass_idx"])
                            ),
                            moe_layer_ordinal_in_pass=(
                                None
                                if payload.get("moe_layer_ordinal_in_pass") is None
                                else int(payload["moe_layer_ordinal_in_pass"])
                            ),
                            layer_id=int(payload["layer_id"]),
                            layer_name=str(payload["layer_name"]),
                            source_global_rank=int(payload["source_global_rank"]),
                            source_ep_rank=int(payload["source_ep_rank"]),
                            dest_global_ranks=[
                                int(x) for x in payload["dest_global_ranks"]
                            ],
                            dest_ep_ranks=[int(x) for x in payload["dest_ep_ranks"]],
                            dest_token_counts=[
                                int(x) for x in payload["dest_token_counts"]
                            ],
                            num_tokens=int(payload["num_tokens"]),
                            top_k=int(payload["top_k"]),
                            num_token_copies=int(payload["num_token_copies"]),
                            ep_world_size=int(payload["ep_world_size"]),
                            enable_eplb=bool(payload["enable_eplb"]),
                            expert_placement_strategy=payload.get(
                                "expert_placement_strategy"
                            ),
                            trace_file=str(path),
                        )
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse {path}:{line_no}: {exc}"
                    ) from exc
    if not records:
        raise ValueError("No traffic records found.")
    return records


def build_matrices(
    records: list[TrafficRecord],
) -> tuple[
    dict[int, dict[int, np.ndarray]],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, str],
    dict[int, str | None],
    str,
    int,
]:
    ep_world_size = max(record.ep_world_size for record in records)
    layer_names: dict[int, str] = {}
    per_pass_per_layer: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    per_pass_total: dict[int, np.ndarray] = {}
    per_layer_total: dict[int, np.ndarray] = {}
    pass_index_to_signature: dict[int, str | None] = {}
    use_aligned_event_order = all(
        record.moe_layer_ordinal_in_pass is not None for record in records
    )
    use_batch_signature = (
        not use_aligned_event_order and any(record.batch_signature for record in records)
    )
    grouping_mode = (
        "aligned_event_order"
        if use_aligned_event_order
        else (
            "batch_signature" if use_batch_signature else "legacy_local_forward_pass"
        )
    )

    record_to_pass_idx: dict[tuple[str, int], int] = {}
    if use_aligned_event_order:
        records_by_file: dict[str, list[TrafficRecord]] = defaultdict(list)
        for record in records:
            records_by_file[record.trace_file].append(record)
            layer_names[record.layer_id] = record.layer_name

        for trace_file, file_records in records_by_file.items():
            file_records.sort(key=lambda record: record.event_idx)
            current_pass_idx = -1
            for record in file_records:
                layer_ordinal = int(record.moe_layer_ordinal_in_pass or 0)
                if layer_ordinal == 0:
                    current_pass_idx += 1
                elif current_pass_idx < 0:
                    current_pass_idx = 0
                record_to_pass_idx[(trace_file, record.event_idx)] = current_pass_idx
                if record.batch_signature is not None:
                    pass_index_to_signature.setdefault(
                        current_pass_idx, record.batch_signature
                    )
    else:
        grouped_records: dict[str, list[TrafficRecord]] = defaultdict(list)
        group_sort_keys: dict[str, tuple[int, int, str]] = {}

        for record in records:
            if use_batch_signature:
                if not record.batch_signature:
                    continue
                group_key = f"sig:{record.batch_signature}"
            else:
                if record.forward_pass_idx is None:
                    continue
                group_key = (
                    f"legacy:{record.trace_file}:{record.source_global_rank}:"
                    f"{record.forward_pass_idx}"
                )

            grouped_records[group_key].append(record)
            sort_key = (record.event_idx, record.source_global_rank, record.trace_file)
            previous_sort_key = group_sort_keys.get(group_key)
            if previous_sort_key is None or sort_key < previous_sort_key:
                group_sort_keys[group_key] = sort_key

        ordered_group_keys = sorted(
            grouped_records, key=lambda key: group_sort_keys[key]
        )
        group_to_pass_idx = {
            group_key: pass_idx for pass_idx, group_key in enumerate(ordered_group_keys)
        }

    for record in records:
        layer_names[record.layer_id] = record.layer_name
        if use_aligned_event_order:
            pass_idx = record_to_pass_idx[(record.trace_file, record.event_idx)]
        else:
            if use_batch_signature:
                if not record.batch_signature:
                    continue
                group_key = f"sig:{record.batch_signature}"
            else:
                if record.forward_pass_idx is None:
                    continue
                group_key = (
                    f"legacy:{record.trace_file}:{record.source_global_rank}:"
                    f"{record.forward_pass_idx}"
                )

            pass_idx = group_to_pass_idx[group_key]
            pass_index_to_signature.setdefault(pass_idx, record.batch_signature)
        layer_map = per_pass_per_layer.setdefault(pass_idx, {})
        layer_matrix = layer_map.setdefault(
            record.layer_id,
            np.zeros((ep_world_size, ep_world_size), dtype=np.int64),
        )
        layer_matrix[record.source_ep_rank, : len(record.dest_token_counts)] += np.array(
            record.dest_token_counts, dtype=np.int64
        )

    for pass_idx, layer_map in per_pass_per_layer.items():
        pass_total = np.zeros((ep_world_size, ep_world_size), dtype=np.int64)
        for layer_id, layer_matrix in layer_map.items():
            pass_total += layer_matrix
            total_matrix = per_layer_total.setdefault(
                layer_id,
                np.zeros((ep_world_size, ep_world_size), dtype=np.int64),
            )
            total_matrix += layer_matrix
        per_pass_total[pass_idx] = pass_total

    return (
        per_pass_per_layer,
        per_pass_total,
        per_layer_total,
        layer_names,
        pass_index_to_signature,
        grouping_mode,
        ep_world_size,
    )


def choose_plot_pass(
    per_pass_per_layer: dict[int, dict[int, np.ndarray]],
    explicit_pass_idx: int | None,
) -> int | None:
    if explicit_pass_idx is None:
        return None
    return explicit_pass_idx if explicit_pass_idx in per_pass_per_layer else None


def _annotate_heatmap(ax: plt.Axes, matrix: np.ndarray) -> None:
    max_value = float(matrix.max()) if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if max_value > 0 and value > (0.55 * max_value) else "black"
            ax.text(col, row, str(value), ha="center", va="center", fontsize=8, color=color)


def _style_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
) -> None:
    image = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Destination EP Rank")
    ax.set_ylabel("Source EP Rank")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    _annotate_heatmap(ax, matrix)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_total_matrix(
    label: str,
    matrix: np.ndarray,
    output_root: Path,
    show_plots: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _style_heatmap(ax, matrix, f"MoE Dispatch Traffic: {label}")
    fig.tight_layout()
    plot_path = output_root / f"{label}__total_traffic.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    if show_plots and _backend_supports_show():
        plt.show()
    plt.close(fig)
    return plot_path


def plot_layer_matrix_grid(
    label: str,
    layer_matrices: dict[int, np.ndarray],
    layer_names: dict[int, str],
    output_root: Path,
    show_plots: bool,
    plot_max_layers: int | None,
) -> Path:
    layer_ids = sorted(layer_matrices.keys())
    if plot_max_layers is not None:
        layer_ids = layer_ids[:plot_max_layers]

    num_layers = len(layer_ids)
    num_cols = min(4, max(1, num_layers))
    num_rows = int(np.ceil(num_layers / num_cols))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4.6 * num_cols, 4.0 * num_rows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for idx, layer_id in enumerate(layer_ids):
        ax = axes.flat[idx]
        ax.axis("on")
        layer_name = layer_names.get(layer_id, f"layer_{layer_id}")
        title = f"L{layer_id}: {layer_name.split('.')[-1]}"
        _style_heatmap(ax, layer_matrices[layer_id], title)

    fig.suptitle(f"MoE Dispatch Traffic by Layer: {label}", fontsize=14)
    fig.tight_layout()
    plot_path = output_root / f"{label}__full_forward_traffic.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    if show_plots and _backend_supports_show():
        plt.show()
    plt.close(fig)
    return plot_path


def plot_individual_layer_matrices(
    layer_matrices: dict[int, np.ndarray],
    layer_names: dict[int, str],
    output_root: Path,
    show_plots: bool,
    subdir_name: str = "per_layer",
) -> Path:
    layer_dir = output_root / subdir_name
    layer_dir.mkdir(parents=True, exist_ok=True)
    for layer_id, matrix in sorted(layer_matrices.items()):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        layer_name = layer_names.get(layer_id, f"layer_{layer_id}")
        _style_heatmap(ax, matrix, f"Layer {layer_id}: {layer_name}")
        fig.tight_layout()
        plot_path = layer_dir / f"layer_{layer_id}__traffic.png"
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        if show_plots and _backend_supports_show():
            plt.show()
        plt.close(fig)
    return layer_dir


def main() -> None:
    args = parse_args()
    trace_files = discover_trace_files(args.input_dir.resolve())
    if not trace_files:
        raise SystemExit(f"No traffic_*.jsonl files found under {args.input_dir}")

    output_root = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else resolve_default_output_root(args.input_dir.resolve(), trace_files)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    records = load_records(trace_files)
    (
        per_pass_per_layer,
        per_pass_total,
        per_layer_total,
        layer_names,
        pass_index_to_signature,
        grouping_mode,
        ep_world_size,
    ) = build_matrices(records)

    plot_pass_idx = choose_plot_pass(per_pass_per_layer, args.plot_pass)
    overall_total = np.zeros((ep_world_size, ep_world_size), dtype=np.int64)
    for matrix in per_layer_total.values():
        overall_total += matrix
    plotted_overall_total_path = plot_total_matrix(
        "overall",
        overall_total,
        output_root,
        args.show_plots,
    )
    plotted_overall_layers_path = plot_layer_matrix_grid(
        "overall",
        per_layer_total,
        layer_names,
        output_root,
        args.show_plots,
        args.plot_max_layers,
    )

    plotted_total_path: Path | None = None
    plotted_full_pass_path: Path | None = None
    per_pass_layer_plot_dir: Path | None = None
    if plot_pass_idx is not None:
        plotted_total_path = plot_total_matrix(
            f"pass_{plot_pass_idx}",
            per_pass_total[plot_pass_idx],
            output_root,
            args.show_plots,
        )
        plotted_full_pass_path = plot_layer_matrix_grid(
            f"pass_{plot_pass_idx}",
            per_pass_per_layer[plot_pass_idx],
            layer_names,
            output_root,
            args.show_plots,
            args.plot_max_layers,
        )
        per_pass_layer_plot_dir = plot_individual_layer_matrices(
            per_pass_per_layer[plot_pass_idx],
            layer_names,
            output_root,
            args.show_plots,
            subdir_name=f"pass_{plot_pass_idx}_per_layer",
        )

    payload = {
        "metadata": {
            "trace_root": str(args.input_dir.resolve()),
            "trace_files": [str(path) for path in trace_files],
            "ep_world_size": ep_world_size,
            "num_records": len(records),
            "num_forward_passes": len(per_pass_per_layer),
            "num_layers": len(per_layer_total),
            "grouping_mode": grouping_mode,
        },
        "records": [asdict(record) for record in records],
        "overall_total_exact": overall_total,
        "per_layer_exact": {
            layer_id: matrix for layer_id, matrix in sorted(per_layer_total.items())
        },
        "pass_index_to_batch_signature": dict(sorted(pass_index_to_signature.items())),
        "per_pass_per_layer": {
            pass_idx: {
                layer_id: matrix
                for layer_id, matrix in sorted(layer_map.items())
            }
            for pass_idx, layer_map in sorted(per_pass_per_layer.items())
        },
        "per_pass_total": {
            pass_idx: matrix for pass_idx, matrix in sorted(per_pass_total.items())
        },
        "per_layer_total": {
            layer_id: matrix for layer_id, matrix in sorted(per_layer_total.items())
        },
        "layer_names": dict(sorted(layer_names.items())),
    }

    pickle_path = output_root / "traffic_matrices.pkl"
    with pickle_path.open("wb") as fp:
        pickle.dump(payload, fp)

    per_layer_plot_dir = plot_individual_layer_matrices(
        per_layer_total,
        layer_names,
        output_root,
        args.show_plots,
    )

    summary = TrafficSummary(
        num_trace_files=len(trace_files),
        num_records=len(records),
        num_forward_passes=len(per_pass_per_layer),
        num_layers=len(per_layer_total),
        ep_world_size=ep_world_size,
        grouping_mode=grouping_mode,
        trace_root=str(args.input_dir.resolve()),
        pickle_path=str(pickle_path),
        plotted_pass_idx=plot_pass_idx,
        plotted_total_path=None if plotted_total_path is None else str(plotted_total_path),
        plotted_full_pass_path=(
            None if plotted_full_pass_path is None else str(plotted_full_pass_path)
        ),
        plotted_overall_total_path=str(plotted_overall_total_path),
        plotted_overall_layers_path=str(plotted_overall_layers_path),
        per_layer_plot_dir=str(per_layer_plot_dir),
        per_pass_layer_plot_dir=(
            None if per_pass_layer_plot_dir is None else str(per_pass_layer_plot_dir)
        ),
    )

    summary_path = output_root / "traffic_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    print(f"Discovered {len(trace_files)} traffic trace files under {args.input_dir.resolve()}")
    print(
        "Built exact matrices for "
        f"{summary.num_forward_passes} forward passes, "
        f"{summary.num_layers} layers, EP={summary.ep_world_size}, "
        f"grouping={summary.grouping_mode}"
    )
    print(f"Wrote pickle to: {pickle_path}")
    print(f"Wrote summary to: {summary_path}")
    if plot_pass_idx is not None:
        print(f"Plotted forward pass: {plot_pass_idx}")
        if plotted_total_path is not None:
            print(f"  total matrix: {plotted_total_path}")
        if plotted_full_pass_path is not None:
            print(f"  full forward: {plotted_full_pass_path}")
        if per_pass_layer_plot_dir is not None:
            print(f"  per-layer exact matrices: {per_pass_layer_plot_dir}")
    print(f"Plotted overall total: {plotted_overall_total_path}")
    print(f"Plotted overall layers: {plotted_overall_layers_path}")
    print(f"Plotted per-layer exact matrices: {per_layer_plot_dir}")


if __name__ == "__main__":
    main()



#%%


with open("traffic-eplb/deepseek-moe-16b-chat-mmlu-all-test/analysis/traffic_matrices.pkl", "rb") as f:
    data = pickle.load(f)

pass_id = 100
for layer in range(1,len(data["per_pass_per_layer"][100])+1):
    a2a = data["per_pass_per_layer"][pass_id][layer]
    fig, ax = plt.subplots(1,1)
    ax = sns.heatmap(a2a, cmap='coolwarm')
    ax.set_title(f'layer={layer}, pass={pass_id}')
    
    
#%%


dfeplb = pd.read_csv("summary-eplb/deepseek-moe-16b-chat_final_summary.csv")

dfmetis = pd.read_csv("summary-metis/deepseek-moe-16b-chat_final_summary.csv")


figttft, axttft = plt.subplots(1,1,figsize=(16,4))
figthroughput, axthroughput = plt.subplots(1,1,figsize=(16,4))

ttft99eplb = list()
throughputeplb = list()
ttft99metis = list()
throughputmetis = list()

for dataset in dfeplb["dataset"]:
    ttft99eplb.append(dfeplb[(dfeplb["dataset"]==dataset)]["svcttft99"])
    ttft99metis.append(dfmetis[(dfmetis["dataset"]==dataset)]["svcttft99"])
    
    throughputeplb.append(dfeplb[(dfeplb["dataset"]==dataset)]["throughput"])
    throughputmetis.append(dfmetis[(dfmetis["dataset"]==dataset)]["throughput"])
    

datasets = dfeplb["dataset"]

# axttft.plot(np.arange(len(datasets)), ttft99eplb, c='r')
# axttft.plot(np.arange(len(datasets)), ttft99metis, c='k')


# axthroughput.plot(np.arange(len(datasets)), throughputeplb, c='r')
# axthroughput.plot(np.arange(len(datasets)), throughputmetis, c='k')

axttft.plot(np.arange(len(datasets)), [ttft99eplb[i]/ttft99metis[i] for i in range(len(datasets))], c='r')
axttft.axhline(y=1, c='k', ls='--')
axttft.set_xticks(np.arange(len(datasets)))
axttft.set_xticklabels(datasets,fontsize=8,rotation=45)
axttft.set_ylabel("TTFT 99-pct Speedup $x$")

axthroughput.plot(np.arange(len(datasets)), [throughputmetis[i]/throughputeplb[i] for i in range(len(datasets))], c='r')
axthroughput.axhline(y=1, c='k', ls='--')
axthroughput.set_xticks(np.arange(len(datasets)))
axthroughput.set_xticklabels(datasets,fontsize=8,rotation=45)
axthroughput.set_ylabel("Throughput Speedup $x$")