#!/usr/bin/env python3
"""
Analyze MoE routing paths from saved .pt files.

Finds the most common sub-paths (of length >= 2 layers) that tokens take
through the expert layers, and visualizes them as a graph.

Usage:
    python analyze_routing_paths.py routing_data_rank0.pt [routing_data_rank1.pt ...]
    python analyze_routing_paths.py --glob "routing_data_rank*.pt"
    python analyze_routing_paths.py --glob "routing_data_rank*.pt" --min-path-len 3 --top-k 30
"""

import argparse
import glob
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go
import torch


def load_all_ranks(files: list[str]) -> tuple[torch.Tensor, list[dict]]:
    """Load expert_ids from multiple .pt files with per-rank metadata.
    
    Returns:
        combined expert_ids: [total_tokens, num_layers, top_k]
        rank_info: list of dicts with per-rank diagnostics
    """
    all_ids = []
    rank_info = []
    for f in files:
        data = torch.load(f, weights_only=True)
        all_ids.append(data['expert_ids'])
        info = {
            'file': f,
            'dp_rank': int(data.get('dp_rank', len(rank_info))),
            'num_tokens': int(data['num_tokens']),
            'num_layers': int(data['num_layers']),
            'top_k': int(data['top_k']),
        }
        rank_info.append(info)
    combined = torch.cat(all_ids, dim=0)
    return combined, rank_info


def extract_subpaths(expert_ids: torch.Tensor,
                     min_len: int = 2,
                     max_len: int | None = None,
                     use_top1: bool = True) -> Counter:
    """Extract all contiguous sub-paths of the given length range.
    
    A path is a tuple of (layer_idx, expert_id) pairs across consecutive layers.
    We use the top-1 expert (highest weight) at each layer.
    
    Args:
        expert_ids: [num_tokens, num_layers, top_k]
        min_len: minimum sub-path length (number of layers)
        max_len: maximum sub-path length (None = num_layers)
        use_top1: if True, use only the top-1 expert per layer
    
    Returns:
        Counter mapping path tuples to counts.
        Each path is a tuple like ((layer0, expert), (layer1, expert), ...)
    """
    num_tokens, num_layers, top_k = expert_ids.shape
    if max_len is None:
        max_len = num_layers

    # Use top-1 expert per layer: expert_ids[:, :, 0]
    if use_top1:
        routes = expert_ids[:, :, 0]  # [num_tokens, num_layers]
    else:
        # Use sorted pair as the "node" identity
        routes = expert_ids  # keep full

    path_counts: Counter = Counter()

    for path_len in range(min_len, max_len + 1):
        for start_layer in range(num_layers - path_len + 1):
            end_layer = start_layer + path_len
            if use_top1:
                # Each sub-path: tuple of (layer_idx, expert_id)
                chunk = routes[:, start_layer:end_layer]  # [num_tokens, path_len]
                for t in range(num_tokens):
                    path = tuple(
                        (start_layer + i, int(chunk[t, i]))
                        for i in range(path_len)
                    )
                    path_counts[path] += 1
            else:
                chunk = routes[:, start_layer:end_layer, :]
                for t in range(num_tokens):
                    path = tuple(
                        (start_layer + i, tuple(sorted(chunk[t, i].tolist())))
                        for i in range(path_len)
                    )
                    path_counts[path] += 1

    return path_counts


def extract_subpaths_fast(expert_ids: torch.Tensor,
                          min_len: int = 2,
                          max_len: int | None = None) -> Counter:
    """Vectorized sub-path extraction using top-1 expert per layer."""
    num_tokens, num_layers, _ = expert_ids.shape
    if max_len is None:
        max_len = num_layers

    routes = expert_ids[:, :, 0].numpy()  # [num_tokens, num_layers]
    path_counts: Counter = Counter()

    for path_len in range(min_len, max_len + 1):
        for start_layer in range(num_layers - path_len + 1):
            end_layer = start_layer + path_len
            chunk = routes[:, start_layer:end_layer]  # [num_tokens, path_len]
            # Convert each row to a hashable tuple with layer indices
            for row in chunk:
                path = tuple(
                    (start_layer + i, int(row[i]))
                    for i in range(path_len)
                )
                path_counts[path] += 1

    return path_counts


def draw_heatmap(expert_ids: torch.Tensor,
                 rank_info: list[dict],
                 path_counts: Counter,
                 top_k_paths: int = 20,
                 save_path: str | None = None):
    """Draw a heatmap of expert usage per layer plus diagnostics.
    
    Layout:
      - Top: diagnostics text (per-rank and total token counts)
      - Middle: heatmap (x=layer, y=expert, color=selection count)
      - Bottom: top paths table
    """
    num_tokens, num_layers, top_k = expert_ids.shape
    num_experts = int(expert_ids.max().item()) + 1

    # Build heatmap: count how many times each expert is selected (top-1) at each layer
    routes_top1 = expert_ids[:, :, 0]  # [num_tokens, num_layers]
    heatmap = np.zeros((num_experts, num_layers), dtype=np.int64)
    for layer in range(num_layers):
        for expert in range(num_experts):
            heatmap[expert, layer] = int((routes_top1[:, layer] == expert).sum())

    # Normalize to percentage of tokens
    heatmap_pct = 100.0 * heatmap / num_tokens

    # --- Build figure ---
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.45], hspace=0.35)

    # === HEATMAP ===
    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(heatmap_pct, aspect='auto', cmap='YlOrRd',
                         interpolation='nearest')

    ax_heat.set_xticks(range(num_layers))
    ax_heat.set_xticklabels([f'L{i}' for i in range(num_layers)], fontsize=7, rotation=45)
    ax_heat.set_yticks(range(num_experts))
    ax_heat.set_yticklabels([f'Expert {i}' for i in range(num_experts)], fontsize=9)
    ax_heat.set_xlabel('Layer', fontsize=11)
    ax_heat.set_ylabel('Expert', fontsize=11)

    # Annotate cells with percentage
    for e in range(num_experts):
        for l in range(num_layers):
            val = heatmap_pct[e, l]
            color = 'white' if val > heatmap_pct.max() * 0.6 else 'black'
            ax_heat.text(l, e, f'{val:.0f}', ha='center', va='center',
                         fontsize=5.5, color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cbar.set_label('% of tokens routed (top-1)', fontsize=9)

    # Diagnostics in title
    diag_lines = []
    for ri in rank_info:
        diag_lines.append(f"Rank {ri['dp_rank']}: {ri['num_tokens']:,} tokens")
    per_rank_str = "  |  ".join(diag_lines)
    ax_heat.set_title(
        f"Expert Selection Heatmap (top-1 expert per layer)\n"
        f"Total tokens (all ranks): {num_tokens:,}   |   "
        f"Ranks: {len(rank_info)}   |   "
        f"Layers: {num_layers}   |   Experts: {num_experts}   |   top_k: {top_k}\n"
        f"{per_rank_str}",
        fontsize=10, pad=12,
    )

    # === TOP PATHS TABLE ===
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')

    most_common = path_counts.most_common(top_k_paths)
    table_data = []
    for rank, (path, count) in enumerate(most_common, 1):
        pct = 100.0 * count / num_tokens
        path_str = " -> ".join(f"L{layer}:E{expert}" for layer, expert in path)
        table_data.append([str(rank), f"{count:,}", f"{pct:.1f}%", str(len(path)), path_str])

    col_labels = ['Rank', 'Count', '% Tokens', 'Len', 'Path']
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.1)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Left-align path column
    for i in range(1, len(table_data) + 1):
        table[i, 4].set_text_props(ha='left')
    # Alternating row colors
    for i in range(1, len(table_data) + 1):
        color = '#f0f4ff' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    # Set column widths
    col_widths = [0.04, 0.06, 0.06, 0.04, 0.40]
    for j, w in enumerate(col_widths):
        for i in range(len(table_data) + 1):
            table[i, j].set_width(w)

    ax_table.set_title(f'Top {top_k_paths} Most Common Routing Sub-Paths',
                       fontsize=10, pad=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    plt.close(fig)


def draw_sankey(expert_ids: torch.Tensor,
                rank_info: list[dict],
                layer_range: tuple[int, int] | None = None,
                min_flow_pct: float = 0.5,
                save_path: str | None = None):
    """Draw a Sankey diagram showing token flow between experts across layers.

    Args:
        expert_ids: [num_tokens, num_layers, top_k]
        rank_info: per-rank diagnostics
        layer_range: (start, end) layer indices to display (inclusive). None = all.
        min_flow_pct: hide edges carrying less than this % of tokens (reduces clutter)
        save_path: output HTML file path
    """
    num_tokens, num_layers, _ = expert_ids.shape
    num_experts = int(expert_ids.max().item()) + 1
    routes = expert_ids[:, :, 0].numpy()  # top-1 expert, [num_tokens, num_layers]

    # Determine layer window
    l_start = layer_range[0] if layer_range else 0
    l_end = layer_range[1] if layer_range else num_layers - 1
    l_end = min(l_end, num_layers - 1)
    window_layers = list(range(l_start, l_end + 1))
    num_window = len(window_layers)

    # Build nodes: one per (layer, expert) in the window
    node_labels = []
    node_colors = []
    # Color palette for experts
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
    ]
    for layer in window_layers:
        for expert in range(num_experts):
            node_labels.append(f'L{layer}:E{expert}')
            node_colors.append(palette[expert % len(palette)])

    def node_idx(layer_pos: int, expert: int) -> int:
        """Map (layer position in window, expert) to node index."""
        return layer_pos * num_experts + expert

    # Build links: count transitions between consecutive layers
    sources, targets, values, link_colors = [], [], [], []
    for i in range(num_window - 1):
        l_from = window_layers[i]
        l_to = window_layers[i + 1]
        from_col = routes[:, l_from]
        to_col = routes[:, l_to]

        for e_from in range(num_experts):
            mask = from_col == e_from
            if not mask.any():
                continue
            dest_experts = to_col[mask]
            for e_to in range(num_experts):
                count = int((dest_experts == e_to).sum())
                pct = 100.0 * count / num_tokens
                if pct < min_flow_pct:
                    continue
                sources.append(node_idx(i, e_from))
                targets.append(node_idx(i + 1, e_to))
                values.append(count)
                # Semi-transparent version of source expert color
                base = palette[e_from % len(palette)]
                r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
                link_colors.append(f'rgba({r},{g},{b},0.35)')

    # Build title
    total = num_tokens
    layer_str = f'Layers {l_start}-{l_end}'
    rank_str = f'{len(rank_info)} ranks, {total:,} total tokens'

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=12,
            thickness=18,
            line=dict(color='#333', width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    )])

    fig.update_layout(
        title=dict(
            text=(
                f'Token Routing Sankey — {layer_str}<br>'
                f'<span style="font-size:12px">{rank_str} | '
                f'Edges with &ge;{min_flow_pct}% of tokens shown</span>'
            ),
            font=dict(size=16),
        ),
        font=dict(size=10),
        width=max(900, num_window * 100),
        height=600,
    )

    if save_path:
        fig.write_html(save_path)
        print(f'Saved Sankey diagram to {save_path}')


def print_top_paths(path_counts: Counter, top_k: int = 30,
                    num_tokens: int = 1):
    """Print the most common sub-paths in a readable table."""
    print(f"\n{'='*80}")
    print(f"TOP {top_k} MOST COMMON ROUTING SUB-PATHS")
    print(f"{'='*80}")
    print(f"{'Rank':>4}  {'Count':>7}  {'% Tokens':>8}  {'Len':>3}  Path")
    print(f"{'-'*4:>4}  {'-'*7:>7}  {'-'*8:>8}  {'-'*3:>3}  {'-'*50}")

    for rank, (path, count) in enumerate(path_counts.most_common(top_k), 1):
        pct = 100.0 * count / num_tokens
        path_str = " -> ".join(f"L{layer}:E{expert}" for layer, expert in path)
        print(f"{rank:>4}  {count:>7}  {pct:>7.1f}%  {len(path):>3}  {path_str}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MoE routing paths")
    parser.add_argument("files", nargs="*", help="Path to .pt routing data files")
    parser.add_argument("--glob", type=str, default=None,
                        help="Glob pattern for .pt files (e.g. 'routing_data_rank*.pt')")
    parser.add_argument("--min-path-len", type=int, default=2,
                        help="Minimum sub-path length in layers (default: 2)")
    parser.add_argument("--max-path-len", type=int, default=4,
                        help="Maximum sub-path length in layers (default: 4)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Number of top paths to show/graph (default: 30)")
    parser.add_argument("--save", type=str, default="routing_heatmap.png",
                        help="Save heatmap to this file (default: routing_heatmap.png)")
    parser.add_argument("--sankey-save", type=str, default="routing_sankey.html",
                        help="Save Sankey diagram to this file (default: routing_sankey.html)")
    parser.add_argument("--sankey-layers", type=str, default=None,
                        help="Layer range for Sankey, e.g. '0-31' or '28-31' (default: all)")
    parser.add_argument("--sankey-min-flow", type=float, default=0.5,
                        help="Hide Sankey edges below this %% of tokens (default: 0.5)")
    parser.add_argument("--no-graph", action="store_true",
                        help="Skip graph generation, just print table")
    args = parser.parse_args()

    # Collect files
    files = list(args.files) if args.files else []
    if args.glob:
        files.extend(sorted(glob.glob(args.glob)))
    if not files:
        # Auto-detect in current directory
        files = sorted(glob.glob("routing_data_rank*.pt"))
    if not files:
        print("No .pt files found. Provide file paths or use --glob.")
        return

    print(f"Loading {len(files)} file(s)...")
    expert_ids, rank_info = load_all_ranks(files)
    num_tokens, num_layers, top_k = expert_ids.shape
    num_experts = int(expert_ids.max().item()) + 1

    # === DIAGNOSTICS ===
    print(f"\n{'='*60}")
    print(f"DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  Ranks loaded:      {len(rank_info)}")
    print(f"  Experts:           {num_experts}")
    print(f"  Layers:            {num_layers}")
    print(f"  top_k:             {top_k}")
    print(f"  Total tokens (all ranks): {num_tokens:,}")
    for ri in rank_info:
        print(f"  Rank {ri['dp_rank']:>2}: {ri['num_tokens']:>8,} tokens  ({ri['file']})")
    print(f"{'='*60}")

    print(f"\nExtracting sub-paths (length {args.min_path_len}-{args.max_path_len})...")
    path_counts = extract_subpaths_fast(
        expert_ids,
        min_len=args.min_path_len,
        max_len=args.max_path_len,
    )
    print(f"Found {len(path_counts)} unique sub-paths")

    print_top_paths(path_counts, top_k=args.top_k, num_tokens=num_tokens)

    if not args.no_graph:
        print(f"\nGenerating heatmap...")
        draw_heatmap(
            expert_ids,
            rank_info,
            path_counts,
            top_k_paths=args.top_k,
            save_path=args.save,
        )

        # Parse Sankey layer range
        sankey_range = None
        if args.sankey_layers:
            parts = args.sankey_layers.split('-')
            sankey_range = (int(parts[0]), int(parts[1]))

        print(f"Generating Sankey diagram...")
        draw_sankey(
            expert_ids,
            rank_info,
            layer_range=sankey_range,
            min_flow_pct=args.sankey_min_flow,
            save_path=args.sankey_save,
        )


if __name__ == "__main__":
    main()
