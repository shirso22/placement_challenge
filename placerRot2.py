"""
SA Placer v6 — Model-Based SA with Online Surrogate Learning

Two-phase optimization with adaptive stopping:

Phase 1: Standard SA with wirelength acceptance (fast, ~20μs/step)
  - Collects (features, proxy_cost) training pairs every checkpoint
  - Exits early when SA converges (no proxy improvement for K checkpoints)

Phase 2: Surrogate-guided SA (slower per step, but optimizing the right thing)
  - Trains a lightweight model to predict proxy cost from placement features
  - SA acceptance uses surrogate prediction instead of raw wirelength
  - Periodically retrains on new real evaluations to stay calibrated
  - Same adaptive stopping: exits when surrogate-guided search converges

The surrogate doesn't need to be perfect — it just needs to be less wrong
than pure wirelength. Wirelength has zero congestion information. A linear
model trained on 30 real evaluations has some. Some > zero.

Usage:
    uv run evaluate submissions/manifold_sa/placer.py
    uv run evaluate submissions/manifold_sa/placer.py --all
"""

import math
import random
import time
from pathlib import Path

import torch
import numpy as np

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Infrastructure (unchanged from v5)
# ---------------------------------------------------------------------------

def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {
        "ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


def _extract_edges(benchmark, plc):
    n_hard = benchmark.num_hard_macros
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
    edge_dict = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            ml = sorted(macros)
            w = 1.0 / (len(ml) - 1)
            for i in range(len(ml)):
                for j in range(i + 1, len(ml)):
                    pair = (ml[i], ml[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + w
    if not edge_dict:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)
    edges = np.array(list(edge_dict.keys()), dtype=np.int64)
    weights = np.array([edge_dict[e] for e in edge_dict], dtype=np.float64)
    return edges, weights


def _build_macro_edge_map(edges, edge_weights, n):
    macro_edges = [[] for _ in range(n)]
    for k in range(len(edges)):
        i, j = int(edges[k, 0]), int(edges[k, 1])
        macro_edges[i].append((k, j))
        macro_edges[j].append((k, i))
    return macro_edges


def _compute_step_scale(pos, sizes, edges, edge_weights, n, half_w, half_h):
    conn = np.zeros(n, dtype=np.float64)
    if len(edges) > 0:
        for k in range(len(edges)):
            i, j = int(edges[k, 0]), int(edges[k, 1])
            conn[i] += edge_weights[k]
            conn[j] += edge_weights[k]
    if conn.max() > 0:
        conn /= conn.max()
    packing = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dx = np.abs(pos[i, 0] - pos[:, 0])
        dy = np.abs(pos[i, 1] - pos[:, 1])
        near = (dx - (half_w[i] + half_w) < sizes[i, 0]) & \
               (dy - (half_h[i] + half_h) < sizes[i, 1])
        near[i] = False
        packing[i] = near.sum()
    if packing.max() > 0:
        packing /= packing.max()
    stiffness = 1.0 + 2.0 * conn + 1.5 * packing
    scale = 1.0 / np.sqrt(stiffness)
    scale /= scale.max()
    return scale


def _real_proxy_cost(pos, sizes, n_hard, benchmark, plc):
    from macro_place.objective import compute_proxy_cost
    full_pos = benchmark.macro_positions.clone()
    full_pos[:n_hard] = torch.tensor(pos, dtype=torch.float32)
    orig_sizes = benchmark.macro_sizes[:n_hard].clone()
    benchmark.macro_sizes[:n_hard] = torch.tensor(sizes, dtype=torch.float32)
    costs = compute_proxy_cost(full_pos, benchmark, plc)
    benchmark.macro_sizes[:n_hard] = orig_sizes
    return costs["proxy_cost"], costs["overlap_count"]


def _has_overlap_single(pos, i, sep_x_row, sep_y_row, n, gap=0.05):
    dx = np.abs(pos[i, 0] - pos[:, 0])
    dy = np.abs(pos[i, 1] - pos[:, 1])
    ov = (dx < sep_x_row + gap) & (dy < sep_y_row + gap)
    ov[i] = False
    return ov.any()


def _recompute_sep(sizes, n):
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    return sep_x, sep_y


# ---------------------------------------------------------------------------
# Legalization (from v5)
# ---------------------------------------------------------------------------

def _legalize_with_order(pos, movable, sizes, half_w, half_h, cw, ch, n, order):
    sep_x, sep_y = _recompute_sep(sizes, n)
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()
    gap = 0.05
    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue
        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            conflict = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap) & placed
            conflict[idx] = False
            if not conflict.any():
                placed[idx] = True
                continue
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.2
        best_p = legal[idx].copy()
        best_d = float('inf')
        for r in range(1, 200):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx])
                    cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx])
                    if placed.any():
                        ddx = np.abs(cx - legal[:, 0])
                        ddy = np.abs(cy - legal[:, 1])
                        conflict = (ddx < sep_x[idx] + gap) & (ddy < sep_y[idx] + gap) & placed
                        conflict[idx] = False
                        if conflict.any():
                            continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])
                        found = True
            if found:
                break
        legal[idx] = best_p
        placed[idx] = True
    return legal


def _multi_start_legalize(pos, movable, sizes, half_w, half_h, cw, ch, n,
                          edges, edge_weights, benchmark, plc, n_hard,
                          n_starts=6, time_limit=30):
    t0 = time.time()
    orderings = []
    orderings.append(sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1]))
    conn_count = np.zeros(n, dtype=np.float64)
    if len(edges) > 0:
        for k in range(len(edges)):
            i, j = int(edges[k, 0]), int(edges[k, 1])
            conn_count[i] += edge_weights[k]
            conn_count[j] += edge_weights[k]
    orderings.append(sorted(range(n), key=lambda i: -conn_count[i]))
    areas = sizes[:, 0] * sizes[:, 1]
    if areas.max() > 0 and conn_count.max() > 0:
        hybrid = (conn_count / max(conn_count.max(), 1e-10)) * (areas / max(areas.max(), 1e-10))
        orderings.append(sorted(range(n), key=lambda i: -hybrid[i]))
    orderings.append(sorted(range(n), key=lambda i: pos[i, 0]))
    orderings.append(sorted(range(n), key=lambda i: pos[i, 1]))
    for seed in range(n_starts - len(orderings)):
        rng = np.random.RandomState(seed + 1000)
        order = list(range(n))
        rng.shuffle(order)
        orderings.append(order)
    orderings = orderings[:n_starts]

    best_legal = None
    best_proxy = float('inf')
    for order in orderings:
        if time.time() - t0 > time_limit:
            break
        legal = _legalize_with_order(pos, movable, sizes, half_w, half_h, cw, ch, n, order)
        proxy, ov = _real_proxy_cost(legal, sizes, n_hard, benchmark, plc)
        if ov == 0 and proxy < best_proxy:
            best_proxy = proxy
            best_legal = legal.copy()
    if best_legal is None:
        best_legal = _legalize_with_order(pos, movable, sizes, half_w, half_h, cw, ch, n, orderings[0])
    return best_legal


# ---------------------------------------------------------------------------
# Cluster identification (from v5)
# ---------------------------------------------------------------------------

def _find_clusters(neighbors, movable, n, max_cluster_size=6):
    clusters = []
    for seed in range(n):
        if not movable[seed] or not neighbors[seed]:
            continue
        cluster = [seed]
        visited = {seed}
        queue = list(neighbors[seed])
        random.shuffle(queue)
        for nxt in queue:
            if len(cluster) >= max_cluster_size:
                break
            if nxt in visited or not movable[nxt]:
                continue
            visited.add(nxt)
            cluster.append(nxt)
        if 2 <= len(cluster) <= max_cluster_size:
            clusters.append(cluster)
    return clusters


# ---------------------------------------------------------------------------
# Surrogate model: feature extraction and prediction
# ---------------------------------------------------------------------------

def _extract_features(pos, sizes, edges, edge_weights, macro_edges,
                      half_w, half_h, cw, ch, n, grid_rows, grid_cols):
    """
    Extract a fixed-size feature vector from a placement.
    
    Features are designed to capture what the proxy cost cares about:
    wirelength structure, density distribution, and congestion proxies.
    """
    features = []

    # ── Wirelength features ──────────────────────────────────────────
    if len(edges) > 0:
        dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
        dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
        manhattan = edge_weights * (dx + dy)
        total_wl = manhattan.sum()
        features.append(total_wl)
        features.append(np.mean(manhattan))
        features.append(np.std(manhattan))
        features.append(np.max(manhattan))
        # Percentiles of weighted edge lengths
        features.append(np.percentile(manhattan, 25))
        features.append(np.percentile(manhattan, 50))
        features.append(np.percentile(manhattan, 75))
        features.append(np.percentile(manhattan, 90))
    else:
        features.extend([0.0] * 8)

    # ── Density features (grid-based) ────────────────────────────────
    grid_w = cw / grid_cols
    grid_h = ch / grid_rows
    cell_area = grid_w * grid_h
    density = np.zeros(grid_rows * grid_cols, dtype=np.float64)

    for i in range(n):
        x, y = pos[i]; w, h = sizes[i]
        c0 = max(0, int((x - w / 2) / grid_w))
        c1 = min(grid_cols - 1, int((x + w / 2) / grid_w))
        r0 = max(0, int((y - h / 2) / grid_h))
        r1 = min(grid_rows - 1, int((y + h / 2) / grid_h))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cx0, cx1 = c * grid_w, (c + 1) * grid_w
                cy0, cy1 = r * grid_h, (r + 1) * grid_h
                ox = max(0.0, min(x + w / 2, cx1) - max(x - w / 2, cx0))
                oy = max(0.0, min(y + h / 2, cy1) - max(y - h / 2, cy0))
                density[r * grid_cols + c] += ox * oy / cell_area

    # Density statistics
    features.append(np.mean(density))
    features.append(np.std(density))
    features.append(np.max(density))
    k10 = max(1, int(0.1 * len(density)))
    features.append(np.partition(density, -k10)[-k10:].mean())  # top 10% avg
    k5 = max(1, int(0.05 * len(density)))
    features.append(np.partition(density, -k5)[-k5:].mean())    # top 5% avg
    features.append(np.percentile(density, 75))
    features.append(np.percentile(density, 90))
    features.append(np.percentile(density, 95))
    # Number of "hot" cells (density > 1.0)
    features.append((density > 1.0).sum())
    features.append((density > 1.5).sum())

    # ── Congestion proxy features (RUDY-style) ──────────────────────
    h_routing = np.zeros(grid_rows * grid_cols, dtype=np.float64)
    v_routing = np.zeros(grid_rows * grid_cols, dtype=np.float64)

    if len(edges) > 0:
        for eidx in range(len(edges)):
            i, j = int(edges[eidx, 0]), int(edges[eidx, 1])
            w = edge_weights[eidx]
            x0, y0 = pos[i]; x1, y1 = pos[j]
            min_x, max_x = min(x0, x1), max(x0, x1)
            min_y, max_y = min(y0, y1), max(y0, y1)
            c0 = max(0, int(min_x / grid_w))
            c1 = min(grid_cols - 1, int(max_x / grid_w))
            r0 = max(0, int(min_y / grid_h))
            r1 = min(grid_rows - 1, int(max_y / grid_h))
            nc = max(1, (c1 - c0 + 1) * (r1 - r0 + 1))
            h_dem = w * abs(y1 - y0) / (grid_h * nc)
            v_dem = w * abs(x1 - x0) / (grid_w * nc)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    idx = r * grid_cols + c
                    h_routing[idx] += h_dem
                    v_routing[idx] += v_dem

    combined_routing = np.maximum(h_routing, v_routing)
    features.append(np.mean(combined_routing))
    features.append(np.std(combined_routing))
    features.append(np.max(combined_routing))
    features.append(np.partition(combined_routing, -k5)[-k5:].mean())
    features.append(np.percentile(combined_routing, 90))
    features.append(np.percentile(combined_routing, 95))

    # ── Macro blockage features ──────────────────────────────────────
    # How much routing is blocked by macros in congested areas
    macro_block = np.zeros(grid_rows * grid_cols, dtype=np.float64)
    for i in range(n):
        x, y = pos[i]; w, h = sizes[i]
        c0 = max(0, int((x - w / 2) / grid_w))
        c1 = min(grid_cols - 1, int((x + w / 2) / grid_w))
        r0 = max(0, int((y - h / 2) / grid_h))
        r1 = min(grid_rows - 1, int((y + h / 2) / grid_h))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cx0, cx1 = c * grid_w, (c + 1) * grid_w
                cy0, cy1 = r * grid_h, (r + 1) * grid_h
                ox = max(0.0, min(x + w / 2, cx1) - max(x - w / 2, cx0))
                oy = max(0.0, min(y + h / 2, cy1) - max(y - h / 2, cy0))
                macro_block[r * grid_cols + c] += (ox * oy) / cell_area

    # Correlation between routing demand and macro blockage
    if np.std(combined_routing) > 0 and np.std(macro_block) > 0:
        features.append(np.corrcoef(combined_routing, macro_block)[0, 1])
    else:
        features.append(0.0)
    # Product of routing and blockage (congestion hotspot indicator)
    hotspot = combined_routing * macro_block
    features.append(np.mean(hotspot))
    features.append(np.max(hotspot))
    features.append(np.partition(hotspot, -k5)[-k5:].mean())

    # ── Packing features ────────────────────────────────────────────
    # Average gap to nearest neighbor
    min_gaps = []
    for i in range(n):
        ddx = np.abs(pos[i, 0] - pos[:, 0]) - (half_w[i] + half_w)
        ddy = np.abs(pos[i, 1] - pos[:, 1]) - (half_h[i] + half_h)
        gap = np.maximum(ddx, 0) + np.maximum(ddy, 0)
        gap[i] = float('inf')
        min_gaps.append(gap.min())
    min_gaps = np.array(min_gaps)
    features.append(np.mean(min_gaps))
    features.append(np.std(min_gaps))
    features.append(np.min(min_gaps))
    features.append(np.percentile(min_gaps, 10))

    return np.array(features, dtype=np.float64)


class ProxySurrogate:
    """Simple ridge regression surrogate for proxy cost prediction."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = 0.0
        self.X_mean = None
        self.X_std = None
        self.y_mean = 0.0
        self.trained = False

    def fit(self, X, y):
        """Fit ridge regression: y = Xw + b."""
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if len(X) < 5:
            return

        # Normalize features
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.X_std[self.X_std < 1e-10] = 1.0  # avoid div by zero
        self.y_mean = y.mean()

        Xn = (X - self.X_mean) / self.X_std
        yn = y - self.y_mean

        # Ridge regression closed form: w = (X'X + αI)^-1 X'y
        n_features = Xn.shape[1]
        XtX = Xn.T @ Xn + self.alpha * np.eye(n_features)
        Xty = Xn.T @ yn
        try:
            self.weights = np.linalg.solve(XtX, Xty)
            self.bias = self.y_mean
            self.trained = True
        except np.linalg.LinAlgError:
            self.trained = False

    def predict(self, x):
        """Predict proxy cost for a single feature vector."""
        if not self.trained:
            return 0.0
        xn = (x - self.X_mean) / self.X_std
        return float(xn @ self.weights + self.bias)


# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

class ManifoldSAPlacer:
    def __init__(self, seed=42, time_budget=240,
                 checkpoint_interval=10_000,
                 convergence_patience=5,
                 n_legal_starts=6):
        self.seed = seed
        self.time_budget = time_budget
        self.checkpoint_interval = checkpoint_interval
        self.convergence_patience = convergence_patience
        self.n_legal_starts = n_legal_starts

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_start = time.time()
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64).copy()
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard].numpy()
        movable_idx = np.where(movable)[0]

        if len(movable_idx) == 0:
            return benchmark.macro_positions.clone()

        plc = _load_plc(benchmark.name)
        if plc is not None:
            edges, edge_weights = _extract_edges(benchmark, plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            edge_weights = np.zeros(0, dtype=np.float64)

        neighbors = [[] for _ in range(n_hard)]
        if len(edges) > 0:
            for k in range(len(edges)):
                i, j = int(edges[k, 0]), int(edges[k, 1])
                neighbors[i].append(j)
                neighbors[j].append(i)

        macro_edges = _build_macro_edge_map(edges, edge_weights, n_hard)

        aspect = sizes[:, 0] / np.maximum(sizes[:, 1], 1e-10)
        rotatable = movable & ((aspect > 1.3) | (aspect < 0.77))
        rotatable_idx = np.where(rotatable)[0]

        clusters = _find_clusters(neighbors, movable, n_hard, max_cluster_size=6)

        # ── Multi-start legalization ─────────────────────────────────────
        pos_init = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        legal_time = min(40, self.time_budget * 0.15)
        pos = _multi_start_legalize(
            pos_init, movable, sizes, half_w, half_h, cw, ch, n_hard,
            edges, edge_weights, benchmark, plc, n_hard,
            n_starts=self.n_legal_starts, time_limit=legal_time
        )

        step_scale = _compute_step_scale(pos, sizes, edges, edge_weights, n_hard, half_w, half_h)

        # ── Two-phase SA ─────────────────────────────────────────────────
        pos, sizes = self._two_phase_sa(
            pos, sizes, edges, edge_weights, movable, movable_idx,
            half_w, half_h, cw, ch, n_hard, neighbors, macro_edges,
            step_scale, rotatable_idx, clusters,
            benchmark, plc, t_start
        )

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(pos, dtype=torch.float32)
        benchmark.macro_sizes[:n_hard] = torch.tensor(sizes, dtype=torch.float32)
        return full_pos

    def _two_phase_sa(self, pos, sizes, edges, edge_weights, movable, movable_idx,
                      half_w, half_h, cw, ch, n, neighbors, macro_edges,
                      step_scale, rotatable_idx, clusters,
                      benchmark, plc, t_start):
        pos = pos.copy()
        sizes = sizes.copy()
        half_w = half_w.copy()
        half_h = half_h.copy()
        n_movable = len(movable_idx)
        if n_movable == 0:
            return pos, sizes
        gap = 0.05

        sep_x, sep_y = _recompute_sep(sizes, n)

        # ── Edge costs ───────────────────────────────────────────────────
        if len(edges) > 0:
            edge_costs = edge_weights * (
                np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0]) +
                np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
            )
            total_wl = edge_costs.sum()
        else:
            edge_costs = np.zeros(0)
            total_wl = 0.0

        # ── Move calibration ─────────────────────────────────────────────
        median_dim = np.median(np.minimum(sizes[movable_idx, 0], sizes[movable_idx, 1]))
        base_step = median_dim * 0.15

        # ── Temperature calibration ──────────────────────────────────────
        test_deltas = []
        for _ in range(min(500, n_movable * 2)):
            i = random.choice(movable_idx)
            ox, oy = pos[i, 0], pos[i, 1]
            s = step_scale[i] * base_step
            pos[i, 0] = np.clip(ox + random.gauss(0, s), half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(oy + random.gauss(0, s), half_h[i], ch - half_h[i])
            delta = 0.0
            for eidx, other in macro_edges[i]:
                delta += edge_weights[eidx] * (
                    abs(pos[i, 0] - pos[other, 0]) + abs(pos[i, 1] - pos[other, 1])
                ) - edge_costs[eidx]
            test_deltas.append(abs(delta))
            pos[i, 0] = ox; pos[i, 1] = oy

        td = max(np.median(test_deltas) if test_deltas else 1.0, 1e-10)
        T_start = td / 0.223
        T_end = td / 4.605

        # ── Best tracking ────────────────────────────────────────────────
        best_proxy, best_ov = _real_proxy_cost(pos, sizes, n, benchmark, plc)
        best_pos = pos.copy()
        best_sizes = sizes.copy()

        # ── Surrogate data collection ────────────────────────────────────
        surrogate = ProxySurrogate(alpha=1.0)
        train_X = []
        train_y = []

        # Collect initial feature/cost pair
        feats = _extract_features(pos, sizes, edges, edge_weights, macro_edges,
                                  half_w, half_h, cw, ch, n,
                                  benchmark.grid_rows, benchmark.grid_cols)
        train_X.append(feats)
        train_y.append(best_proxy)

        movable_with_nbrs = np.array([i for i in movable_idx if len(neighbors[i]) > 0])
        has_rotatable = len(rotatable_idx) > 0
        has_clusters = len(clusters) > 0

        # ── Phase control ────────────────────────────────────────────────
        phase = 1  # 1 = wirelength SA, 2 = surrogate-guided SA
        stale_checkpoints = 0
        step = 0
        max_steps = 10_000_000  # effectively unlimited, controlled by time + convergence

        while step < max_steps:
            # Time check
            if step % 10000 == 0 and step > 0:
                if time.time() - t_start > self.time_budget - 10:
                    break

            frac = min(step / 1_000_000, 1.0)  # anneal over first 1M steps
            T = T_start * (T_end / T_start) ** frac
            step_mult = 1.0 - 0.7 * frac

            # ── Determine cost function for acceptance ───────────────────
            if phase == 2 and surrogate.trained:
                use_surrogate = True
            else:
                use_surrogate = False

            # ── Pick and execute a move ──────────────────────────────────
            r = random.random()

            if r < 0.45:
                # ── SHIFT ──
                i = random.choice(movable_idx)
                ox, oy = pos[i, 0], pos[i, 1]
                s = step_scale[i] * base_step * step_mult
                pos[i, 0] = np.clip(ox + random.gauss(0, s), half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(oy + random.gauss(0, s), half_h[i], ch - half_h[i])

                if _has_overlap_single(pos, i, sep_x[i], sep_y[i], n, gap):
                    pos[i, 0] = ox; pos[i, 1] = oy
                    step += 1; continue

                if use_surrogate:
                    feats_new = _extract_features(pos, sizes, edges, edge_weights, macro_edges,
                                                  half_w, half_h, cw, ch, n,
                                                  benchmark.grid_rows, benchmark.grid_cols)
                    new_cost = surrogate.predict(feats_new)
                    feats_old_pos = pos.copy()
                    feats_old_pos[i] = [ox, oy]
                    # We can cache the old prediction or compute delta
                    # For simplicity, use absolute predictions
                    old_feats = _extract_features(feats_old_pos, sizes, edges, edge_weights,
                                                  macro_edges, half_w, half_h, cw, ch, n,
                                                  benchmark.grid_rows, benchmark.grid_cols)
                    old_cost = surrogate.predict(old_feats)
                    delta = new_cost - old_cost
                else:
                    delta = 0.0
                    for eidx, other in macro_edges[i]:
                        delta += edge_weights[eidx] * (
                            abs(pos[i, 0] - pos[other, 0]) + abs(pos[i, 1] - pos[other, 1])
                        ) - edge_costs[eidx]

                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-15)):
                    # Accept: update edge costs
                    wl_delta = 0.0
                    for eidx, other in macro_edges[i]:
                        new_ec = edge_weights[eidx] * (
                            abs(pos[i, 0] - pos[other, 0]) + abs(pos[i, 1] - pos[other, 1])
                        )
                        wl_delta += new_ec - edge_costs[eidx]
                        edge_costs[eidx] = new_ec
                    total_wl += wl_delta
                else:
                    pos[i, 0] = ox; pos[i, 1] = oy

            elif r < 0.65:
                # ── SWAP ──
                if len(movable_with_nbrs) > 0 and random.random() < 0.7:
                    i = random.choice(movable_with_nbrs)
                    cands = [j for j in neighbors[i] if movable[j]]
                    j = random.choice(cands) if cands else random.choice(movable_idx)
                else:
                    i = random.choice(movable_idx)
                    j = random.choice(movable_idx)
                if i == j:
                    step += 1; continue

                oix, oiy = pos[i, 0], pos[i, 1]
                ojx, ojy = pos[j, 0], pos[j, 1]
                pos[i, 0] = np.clip(ojx, half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(ojy, half_h[i], ch - half_h[i])
                pos[j, 0] = np.clip(oix, half_w[j], cw - half_w[j])
                pos[j, 1] = np.clip(oiy, half_h[j], ch - half_h[j])

                ok = True
                if _has_overlap_single(pos, i, sep_x[i], sep_y[i], n, gap):
                    ok = False
                if ok and _has_overlap_single(pos, j, sep_x[j], sep_y[j], n, gap):
                    ok = False
                if not ok:
                    pos[i, 0] = oix; pos[i, 1] = oiy
                    pos[j, 0] = ojx; pos[j, 1] = ojy
                    step += 1; continue

                # Always compute WL delta for edge cost bookkeeping
                wl_delta = 0.0
                seen = set()
                for eidx, other in macro_edges[i]:
                    wl_delta += edge_weights[eidx] * (
                        abs(pos[i, 0] - pos[other, 0]) + abs(pos[i, 1] - pos[other, 1])
                    ) - edge_costs[eidx]
                    seen.add(eidx)
                for eidx, other in macro_edges[j]:
                    if eidx in seen: continue
                    wl_delta += edge_weights[eidx] * (
                        abs(pos[j, 0] - pos[other, 0]) + abs(pos[j, 1] - pos[other, 1])
                    ) - edge_costs[eidx]

                delta = wl_delta  # use WL for swaps even in surrogate mode (feature extraction too expensive for swaps)

                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-15)):
                    total_wl += wl_delta
                    for eidx, other in macro_edges[i]:
                        edge_costs[eidx] = edge_weights[eidx] * (
                            abs(pos[i, 0] - pos[other, 0]) + abs(pos[i, 1] - pos[other, 1])
                        )
                    for eidx, other in macro_edges[j]:
                        edge_costs[eidx] = edge_weights[eidx] * (
                            abs(pos[j, 0] - pos[other, 0]) + abs(pos[j, 1] - pos[other, 1])
                        )
                else:
                    pos[i, 0] = oix; pos[i, 1] = oiy
                    pos[j, 0] = ojx; pos[j, 1] = ojy

            elif r < 0.80:
                # ── ATTRACTION ──
                if len(movable_with_nbrs) == 0:
                    step += 1; continue
                i = random.choice(movable_with_nbrs)
                nbrs = neighbors[i]
                if not nbrs:
                    step += 1; continue
                j = random.choice(nbrs)
                ox, oy = pos[i, 0], pos[i, 1]
                alpha = random.uniform(0.05, 0.3) * step_scale[i] * step_mult
                pos[i, 0] = np.clip(ox + alpha * (pos[j, 0] - ox), half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(oy + alpha * (pos[j, 1] - oy), half_h[i], ch - half_h[i])

                if _has_overlap_single(pos, i, sep_x[i], sep_y[i], n, gap):
                    pos[i, 0] = ox; pos[i, 1] = oy
                    step += 1; continue

                wl_delta = 0.0
                for eidx, other in macro_edges[i]:
                    wl_delta += edge_weights[eidx] * (
                        abs(pos[i, 0] - pos[other, 0]) + abs(pos[i, 1] - pos[other, 1])
                    ) - edge_costs[eidx]

                if wl_delta < 0 or random.random() < math.exp(-wl_delta / max(T, 1e-15)):
                    total_wl += wl_delta
                    for eidx, other in macro_edges[i]:
                        edge_costs[eidx] = edge_weights[eidx] * (
                            abs(pos[i, 0] - pos[other, 0]) + abs(pos[i, 1] - pos[other, 1])
                        )
                else:
                    pos[i, 0] = ox; pos[i, 1] = oy

            elif r < 0.90 and has_rotatable:
                # ── ROTATION ──
                i = random.choice(rotatable_idx)
                ow, oh = sizes[i, 0], sizes[i, 1]
                sizes[i, 0] = oh; sizes[i, 1] = ow
                half_w[i] = oh / 2; half_h[i] = ow / 2
                sep_x[i, :] = (sizes[i, 0] + sizes[:, 0]) / 2
                sep_x[:, i] = sep_x[i, :]
                sep_y[i, :] = (sizes[i, 1] + sizes[:, 1]) / 2
                sep_y[:, i] = sep_y[i, :]
                pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

                if _has_overlap_single(pos, i, sep_x[i], sep_y[i], n, gap):
                    sizes[i, 0] = ow; sizes[i, 1] = oh
                    half_w[i] = ow / 2; half_h[i] = oh / 2
                    sep_x[i, :] = (sizes[i, 0] + sizes[:, 0]) / 2
                    sep_x[:, i] = sep_x[i, :]
                    sep_y[i, :] = (sizes[i, 1] + sizes[:, 1]) / 2
                    sep_y[:, i] = sep_y[i, :]
                    step += 1; continue

                if random.random() < 0.5 + 0.5 * (1 - frac):
                    pass  # accept
                else:
                    sizes[i, 0] = ow; sizes[i, 1] = oh
                    half_w[i] = ow / 2; half_h[i] = oh / 2
                    sep_x[i, :] = (sizes[i, 0] + sizes[:, 0]) / 2
                    sep_x[:, i] = sep_x[i, :]
                    sep_y[i, :] = (sizes[i, 1] + sizes[:, 1]) / 2
                    sep_y[:, i] = sep_y[i, :]

            elif has_clusters:
                # ── CLUSTER MOVE ──
                cluster = random.choice(clusters)
                s = base_step * step_mult * 0.5
                dx_shift = random.gauss(0, s)
                dy_shift = random.gauss(0, s)
                old_positions = [(pos[idx, 0], pos[idx, 1]) for idx in cluster]
                for idx in cluster:
                    pos[idx, 0] = np.clip(pos[idx, 0] + dx_shift, half_w[idx], cw - half_w[idx])
                    pos[idx, 1] = np.clip(pos[idx, 1] + dy_shift, half_h[idx], ch - half_h[idx])
                ok = True
                for idx in cluster:
                    if _has_overlap_single(pos, idx, sep_x[idx], sep_y[idx], n, gap):
                        ok = False; break
                if not ok:
                    for k2, idx in enumerate(cluster):
                        pos[idx, 0], pos[idx, 1] = old_positions[k2]
                    step += 1; continue

                wl_delta = 0.0
                seen = set()
                for idx in cluster:
                    for eidx, other in macro_edges[idx]:
                        if eidx in seen: continue
                        seen.add(eidx)
                        wl_delta += edge_weights[eidx] * (
                            abs(pos[idx, 0] - pos[other, 0]) + abs(pos[idx, 1] - pos[other, 1])
                        ) - edge_costs[eidx]

                if wl_delta < 0 or random.random() < math.exp(-wl_delta / max(T, 1e-15)):
                    total_wl += wl_delta
                    for idx in cluster:
                        for eidx, other in macro_edges[idx]:
                            edge_costs[eidx] = edge_weights[eidx] * (
                                abs(pos[idx, 0] - pos[other, 0]) + abs(pos[idx, 1] - pos[other, 1])
                            )
                else:
                    for k2, idx in enumerate(cluster):
                        pos[idx, 0], pos[idx, 1] = old_positions[k2]

            step += 1

            # ── Checkpoint: evaluate real proxy, collect data, check convergence ──
            if step % self.checkpoint_interval == 0:
                elapsed = time.time() - t_start
                if elapsed > self.time_budget - 10:
                    break

                proxy, ov_count = _real_proxy_cost(pos, sizes, n, benchmark, plc)

                # Collect training data
                feats = _extract_features(pos, sizes, edges, edge_weights, macro_edges,
                                          half_w, half_h, cw, ch, n,
                                          benchmark.grid_rows, benchmark.grid_cols)
                train_X.append(feats)
                train_y.append(proxy)

                # Best tracking
                if ov_count == 0 and proxy < best_proxy:
                    best_proxy = proxy
                    best_pos = pos.copy()
                    best_sizes = sizes.copy()
                    stale_checkpoints = 0
                else:
                    stale_checkpoints += 1

                # Convergence check → phase transition or stop
                if stale_checkpoints >= self.convergence_patience:
                    if phase == 1:
                        # Transition to phase 2: train surrogate
                        if len(train_X) >= 10:
                            surrogate.fit(train_X, train_y)
                            if surrogate.trained:
                                phase = 2
                                stale_checkpoints = 0
                                # Revert to best position before starting surrogate phase
                                pos = best_pos.copy()
                                sizes = best_sizes.copy()
                                half_w = sizes[:, 0] / 2
                                half_h = sizes[:, 1] / 2
                                sep_x, sep_y = _recompute_sep(sizes, n)
                                # Recompute edge costs from best position
                                if len(edges) > 0:
                                    edge_costs = edge_weights * (
                                        np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0]) +
                                        np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
                                    )
                                    total_wl = edge_costs.sum()
                                # Reset temperature for phase 2
                                T_start = td / 0.223
                                step = 0  # reset annealing
                                continue
                            else:
                                break  # can't train surrogate, give up
                        else:
                            break  # not enough data, give up
                    else:
                        # Phase 2 also converged → done
                        break

                # Retrain surrogate periodically in phase 2
                if phase == 2 and len(train_X) >= 10 and step % (self.checkpoint_interval * 3) == 0:
                    surrogate.fit(train_X, train_y)

        # Final check
        proxy, ov_count = _real_proxy_cost(pos, sizes, n, benchmark, plc)
        if ov_count == 0 and proxy < best_proxy:
            best_pos = pos.copy()
            best_sizes = sizes.copy()

        return best_pos, best_sizes
