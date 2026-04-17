
# LLMEnhancedParallelSO.py
# A Mealpy-free, self-contained "Snake-Optimizer-inspired" multi-group optimizer
# with optional LLM-guided (DeepSeek) cross-group communication strategies.
#
# Author: ChatGPT (GPT-5 Thinking)
# License: MIT
#
# Usage:
#   from LLMEnhancedParallelSO import LLMEnhancedParallelSO
#   opt = LLMEnhancedParallelSO(dim, lb, ub, fit_func, minmax="min", ...)
#   best_pos, best_fit = opt.solve()
#
# Notes:
# - This is a clean-room implementation (no Mealpy dependency).
# - Update rules are SO-inspired (male/female split, temperature q, exploration/exploitation)
#   focusing on practicality and stability rather than exact fidelity to any specific paper.
# - LLM selection is optional; set deepseek_api_key=None to disable network calls.

from __future__ import annotations

import math
import time
import json
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import requests  # used for optional LLM selection
except Exception:  # pragma: no cover
    requests = None


# -------------------------------
# Utilities
# -------------------------------

def _to_array(x, dim):
    arr = np.array(x, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, dim)
    assert arr.size == dim, "lb/ub must be scalar or vector of length dim"
    return arr


def _clip(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


def _levy(n_dim: int, beta: float = 1.5) -> np.ndarray:
    # Mantegna's algorithm for Lévy flights
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=n_dim)
    v = np.random.normal(0, 1, size=n_dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step


@dataclass
class Individual:
    pos: np.ndarray
    fit: float


# -------------------------------
# Optional LLM client (DeepSeek)
# -------------------------------

class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 15.0):
        self.api_key = api_key
        self.timeout = timeout

    def choose_strategy(self, stats: Dict) -> Optional[str]:
        """
        Ask DeepSeek-Chat to pick a communication strategy given current stats.
        Returns a string like 'random_exchange'. If anything fails, returns None.
        """
        if not self.api_key or requests is None:
            return None

        prompt = (
            "You are helping a metaheuristic optimizer decide a cross-group communication strategy. "
            "Given the JSON stats, choose one strategy strictly from this list:\n"
            "['elite_migration','random_exchange','tournament_selection','ring_topology',"
            "'global_broadcast','worst_replacement','cross_group_crossover','lens_opposite_learning',"
            "'adaptive_hybrid','quantum_behavior'].\n"
            "Output ONLY the bare strategy string with no other text.\n"
            f"Stats:\n{json.dumps(stats, ensure_ascii=False)}"
        )
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 8,
            }
            resp = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            resp.raise_for_status()
            out = resp.json()
            content = out["choices"][0]["message"]["content"].strip().lower()
            valid = {
                "elite_migration","random_exchange","tournament_selection","ring_topology",
                "global_broadcast","worst_replacement","cross_group_crossover",
                "lens_opposite_learning","adaptive_hybrid","quantum_behavior"
            }
            return content if content in valid else None
        except Exception:
            return None


# -------------------------------
# Optimizer
# -------------------------------

class LLMEnhancedParallelSO:
    def __init__(
        self,
        dim: int,
        lb,
        ub,
        fit_func: Callable[[np.ndarray], float],
        minmax: str = "min",
        epoch: int = 500,
        pop_size: int = 30,
        n_groups: int = 4,
        male_ratio: float = 0.5,
        c1: float = 0.5,
        c2: float = 0.05,
        c3: float = 2.0,
        seed: Optional[int] = None,
        deepseek_api_key: Optional[str] = None,
        llm_interval: int = 25,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        dim : int
            Dimension of the search space.
        lb, ub : scalar or array-like
            Lower/upper bounds; can be scalar or 1-D array of length dim.
        fit_func : Callable
            Objective function, signature f(x: np.ndarray) -> float
        minmax : {'min','max'}
            Optimization direction.
        epoch : int
            Total iterations.
        pop_size : int
            Population per group (total population = pop_size * n_groups).
        n_groups : int
            Number of groups.
        male_ratio : float
            Fraction of males in each group (0..1).
        c1, c2, c3 : float
            Control parameters used in update rules.
        seed : Optional[int]
            Random seed for reproducibility.
        deepseek_api_key : Optional[str]
            If provided, the optimizer will call DeepSeek to choose a strategy every llm_interval.
        llm_interval : int
            How many iterations between LLM-guided strategy selections (0 disables LLM polling).
        verbose : bool
            Print progress and selected strategies.
        """
        assert 0 < male_ratio < 1, "male_ratio must be in (0,1)"
        self.dim = int(dim)
        self.lb = _to_array(lb, dim)
        self.ub = _to_array(ub, dim)
        self.fit_func = fit_func
        self.minmax = minmax.lower()
        assert self.minmax in {"min", "max"}
        self.epoch = int(epoch)
        self.pop_size = int(pop_size)
        self.n_groups = int(n_groups)
        self.male_ratio = float(male_ratio)
        self.n_males = int(round(self.pop_size * self.male_ratio))
        self.n_females = self.pop_size - self.n_males
        self.c1, self.c2, self.c3 = float(c1), float(c2), float(c3)
        self.seed = seed
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.llm_interval = int(llm_interval)
        self.deepseek = DeepSeekClient(deepseek_api_key) if deepseek_api_key else None

        # State
        self.groups: List[Dict] = []
        self.g_best: Optional[Individual] = None
        self.history: Dict[str, List[float]] = {"best_fitness": []}

        self._init_groups()

    # -------------------------------
    # Core
    # -------------------------------

    def _evaluate(self, pos: np.ndarray) -> float:
        return float(self.fit_func(pos))

    def _init_individual(self) -> Individual:
        pos = np.random.uniform(self.lb, self.ub, size=self.dim)
        fit = self._evaluate(pos)
        return Individual(pos=pos, fit=fit)

    def _init_groups(self):
        self.groups = []
        for _ in range(self.n_groups):
            pop = [self._init_individual() for __ in range(self.pop_size)]
            pop.sort(key=lambda ind: ind.fit, reverse=(self.minmax == "max"))
            g_best = pop[0] if self.minmax == "max" else pop[0]  # sorted already by direction above
            group = {
                "pop": pop,  # full
                "pop_males": pop[: self.n_males].copy(),
                "pop_females": pop[self.n_males :].copy(),
                "g_best": g_best,
            }
            self.groups.append(group)
        # Initialize global best
        self._refresh_global_best()

    def _refresh_global_best(self):
        all_bests = [g["g_best"] for g in self.groups]
        if self.minmax == "min":
            self.g_best = min(all_bests, key=lambda ind: ind.fit)
        else:
            self.g_best = max(all_bests, key=lambda ind: ind.fit)

    def _better(self, a: float, b: float) -> bool:
        return a < b if self.minmax == "min" else a > b

    def _temp(self, t: int) -> float:
        # smooth decay from 1 -> ~0
        return math.exp(-5.0 * (t / max(1, self.epoch)))

    def _diversity(self, pop: List[Individual]) -> float:
        # Use coordinate-wise std as a cheap diversity proxy
        X = np.stack([p.pos for p in pop], axis=0)
        return float(np.mean(np.std(X, axis=0)))

    # -------------------------------
    # Updates (SO-inspired)
    # -------------------------------

    def _update_male(self, x: np.ndarray, best: np.ndarray, temp: float) -> np.ndarray:
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        step = self.c1 * r1 * (best - np.abs(x)) + self.c2 * temp * np.random.randn(self.dim)
        return x + step * (r2 * 2 - 1)  # symmetric push

    def _update_female(self, x: np.ndarray, best: np.ndarray, rnd: np.ndarray, temp: float, exploit: bool) -> np.ndarray:
        r1 = np.random.rand(self.dim)
        if exploit:
            step = self.c3 * r1 * (best - x)
        else:
            step = self.c3 * r1 * (rnd - x)
        step += self.c2 * temp * np.random.randn(self.dim)
        return x + step

    # -------------------------------
    # Communication strategies
    # -------------------------------

    def _apply_elite_migration(self, k: int = 2):
        # Move top-k from the best groups to replace worst-k in the worst groups
        # Rank groups by their current g_best
        ranked = sorted(self.groups, key=lambda g: g["g_best"].fit, reverse=(self.minmax == "max"))
        donors = ranked[: max(1, len(ranked) // 2)]
        receivers = ranked[-max(1, len(ranked) // 2):]

        elites = []
        for g in donors:
            pop_sorted = sorted(g["pop"], key=lambda ind: ind.fit, reverse=(self.minmax == "max"))
            elites.extend(pop_sorted[:k])

        for g in receivers:
            idx_sorted = sorted(range(len(g["pop"])),
                                key=lambda i: g["pop"][i].fit,
                                reverse=(self.minmax == "min"))  # 选出“最差”下标序列
            to_replace_idx = idx_sorted[:k]
            for idx, elite in zip(to_replace_idx, elites[:len(to_replace_idx)]):
                g["pop"][idx] = Individual(pos=elite.pos.copy(), fit=elite.fit)

    def _apply_random_exchange(self, rate: float = 0.2):
        # Randomly swap a fraction of individuals between random pairs of groups
        m = max(1, int(self.pop_size * rate))
        idxs = list(range(self.n_groups))
        random.shuffle(idxs)
        for i in range(0, len(idxs) - 1, 2):
            g1, g2 = self.groups[idxs[i]], self.groups[idxs[i + 1]]
            i1 = np.random.choice(len(g1["pop"]), m, replace=False)
            i2 = np.random.choice(len(g2["pop"]), m, replace=False)
            for a, b in zip(i1, i2):
                g1["pop"][a], g2["pop"][b] = g2["pop"][b], g1["pop"][a]

    def _apply_tournament_selection(self, tour_size: int = 3):
        # Build a new mixed population via tournaments across all groups
        pool = []
        for g in self.groups:
            pool.extend(g["pop"])
        new_groups = [[] for _ in range(self.n_groups)]
        for gi in range(self.n_groups):
            for _ in range(self.pop_size):
                contestants = random.sample(pool, k=min(tour_size, len(pool)))
                if self.minmax == "min":
                    win = min(contestants, key=lambda ind: ind.fit)
                else:
                    win = max(contestants, key=lambda ind: ind.fit)
                new_groups[gi].append(Individual(win.pos.copy(), win.fit))
        for gi in range(self.n_groups):
            self.groups[gi]["pop"] = new_groups[gi]

    def _apply_ring_topology(self, k: int = 2):
        # Send top-k elites to next group in a ring, replacing worst-k
        elites_per_group = []
        for g in self.groups:
            pop_sorted = sorted(g["pop"], key=lambda ind: ind.fit, reverse=(self.minmax == "max"))
            elites_per_group.append([Individual(e.pos.copy(), e.fit) for e in pop_sorted[:k]])
        for i, g in enumerate(self.groups):
            nxt = self.groups[(i + 1) % self.n_groups]
            nxt_sorted = sorted(range(len(nxt["pop"])), key=lambda idx: nxt["pop"][idx].fit, reverse=(self.minmax == "min"))
            # replace worst indices
            for j in range(min(k, len(nxt_sorted))):
                idx = nxt_sorted[j]
                elite = elites_per_group[i][j % len(elites_per_group[i])]
                nxt["pop"][idx] = Individual(elite.pos.copy(), elite.fit)

    def _apply_global_broadcast(self, rate: float = 0.2):
        # Broadcast the global best with slight perturbation to a fraction in each group
        if self.g_best is None:
            return
        m = max(1, int(self.pop_size * rate))
        for g in self.groups:
            idxs = np.random.choice(len(g["pop"]), m, replace=False)
            for idx in idxs:
                noise = 0.01 * (self.ub - self.lb) * np.random.randn(self.dim)
                pos = _clip(self.g_best.pos + noise, self.lb, self.ub)
                fit = self._evaluate(pos)
                g["pop"][idx] = Individual(pos, fit)

    def _apply_worst_replacement(self):
        # Replace worst in each group with mutated best of that group
        for g in self.groups:
            pop_sorted_desc = sorted(range(len(g["pop"])), key=lambda i: g["pop"][i].fit, reverse=(self.minmax == "max"))
            best = g["g_best"]
            worst_idx = pop_sorted_desc[-1] if self.minmax == "min" else pop_sorted_desc[0]
            pos = _clip(best.pos + 0.1 * (self.ub - self.lb) * np.random.randn(self.dim), self.lb, self.ub)
            fit = self._evaluate(pos)
            g["pop"][worst_idx] = Individual(pos, fit)

    def _apply_cross_group_crossover(self, rate: float = 0.3):
        # Simple arithmetic crossover across groups
        m = max(1, int(self.pop_size * rate))
        for _ in range(m):
            g1, g2 = random.sample(self.groups, 2)
            i1 = random.randrange(len(g1["pop"]))
            i2 = random.randrange(len(g2["pop"]))
            a, b = g1["pop"][i1], g2["pop"][i2]
            alpha = np.random.rand(self.dim)
            c1 = _clip(alpha * a.pos + (1 - alpha) * b.pos, self.lb, self.ub)
            c2 = _clip(alpha * b.pos + (1 - alpha) * a.pos, self.lb, self.ub)
            f1, f2 = self._evaluate(c1), self._evaluate(c2)
            # greedy replace worse parents
            if self._better(f1, a.fit):
                g1["pop"][i1] = Individual(c1, f1)
            if self._better(f2, b.fit):
                g2["pop"][i2] = Individual(c2, f2)

    def _apply_lens_opposite_learning(self, rate: float = 0.25):
        # Oppositional learning around the global best
        if self.g_best is None:
            return
        m = max(1, int(self.pop_size * rate))
        for g in self.groups:
            idxs = np.random.choice(len(g["pop"]), m, replace=False)
            for idx in idxs:
                x = g["pop"][idx].pos
                # lens opposite around global best
                x_op = _clip(self.g_best.pos + (self.g_best.pos - x), self.lb, self.ub)
                f_new = self._evaluate(x_op)
                if self._better(f_new, g["pop"][idx].fit):
                    g["pop"][idx] = Individual(x_op, f_new)

    def _apply_adaptive_hybrid(self):
        # Choose exploration vs exploitation per group using diversity and recent progress
        for g in self.groups:
            div = self._diversity(g["pop"])
            # exploration if diversity is low
            if div < 0.05 * float(np.mean(self.ub - self.lb)):
                self._apply_random_exchange(rate=0.25)
            else:
                self._apply_global_broadcast(rate=0.15)

    def _apply_quantum_behavior(self, scale: float = 0.1):
        # Lévy flight around the global best
        if self.g_best is None:
            return
        for g in self.groups:
            idx = random.randrange(len(g["pop"]))
            step = _levy(self.dim) * scale * (self.ub - self.lb)
            pos = _clip(self.g_best.pos + step, self.lb, self.ub)
            fit = self._evaluate(pos)
            if self._better(fit, g["pop"][idx].fit):
                g["pop"][idx] = Individual(pos, fit)

    def _apply_communication_strategy(self, name: str):
        if name == "elite_migration":
            self._apply_elite_migration()
        elif name == "random_exchange":
            self._apply_random_exchange()
        elif name == "tournament_selection":
            self._apply_tournament_selection()
        elif name == "ring_topology":
            self._apply_ring_topology()
        elif name == "global_broadcast":
            self._apply_global_broadcast()
        elif name == "worst_replacement":
            self._apply_worst_replacement()
        elif name == "cross_group_crossover":
            self._apply_cross_group_crossover()
        elif name == "lens_opposite_learning":
            self._apply_lens_opposite_learning()
        elif name == "adaptive_hybrid":
            self._apply_adaptive_hybrid()
        elif name == "quantum_behavior":
            self._apply_quantum_behavior()
        # after changing group['pop'], we must re-split sexes and refresh group bests
        for g in self.groups:
            # recompute fitness for safety (in case external edits haven't evaluated)
            for i in range(len(g["pop"])):
                if not np.isfinite(g["pop"][i].fit):
                    g["pop"][i].fit = self._evaluate(g["pop"][i].pos)
            # split
            pop_sorted = sorted(g["pop"], key=lambda ind: ind.fit, reverse=(self.minmax == "max"))
            g["pop"] = pop_sorted
            g["pop_males"] = pop_sorted[: self.n_males].copy()
            g["pop_females"] = pop_sorted[self.n_males :].copy()
            # g_best
            g["g_best"] = pop_sorted[0] if self.minmax == "min" else pop_sorted[0]

    # -------------------------------
    # LLM selection + fallback heuristic
    # -------------------------------

    def _choose_strategy(self, t: int, recent_best: List[float]) -> str:
        valid = [
            "elite_migration","random_exchange","tournament_selection","ring_topology",
            "global_broadcast","worst_replacement","cross_group_crossover",
            "lens_opposite_learning","adaptive_hybrid","quantum_behavior"
        ]
        # Stats for LLM
        stats = {
            "t": t,
            "epoch": self.epoch,
            "global_best": None if self.g_best is None else float(self.g_best.fit),
            "group_bests": [float(g["g_best"].fit) for g in self.groups],
            "group_diversities": [self._diversity(g["pop"]) for g in self.groups],
            "recent_best_mean": float(np.mean(recent_best[-10:])) if recent_best else None,
        }
        # Ask LLM if available
        if self.deepseek and self.llm_interval > 0 and (t % self.llm_interval == 0) and t > 0:
            s = self.deepseek.choose_strategy(stats)
            if s in valid:
                if self.verbose:
                    print(f"[LLM] t={t} chose strategy: {s}")
                return s
        # Fallback heuristic
        div = float(np.mean(stats["group_diversities"]))
        stagnated = False
        if len(recent_best) > 20:
            window = np.array(recent_best[-20:])
            stagnated = np.allclose(window.min() if self.minmax=="min" else window.max(),
                                    recent_best[-1], rtol=0, atol=1e-12)
        if stagnated and div < 0.05 * float(np.mean(self.ub - self.lb)):
            return "cross_group_crossover"
        if div < 0.03 * float(np.mean(self.ub - self.lb)):
            return "random_exchange"
        if div > 0.2 * float(np.mean(self.ub - self.lb)):
            return "global_broadcast"
        return random.choice(valid)

    # -------------------------------
    # Main solve loop
    # -------------------------------

    def solve(self) -> Tuple[np.ndarray, float]:
        recent = []
        for t in range(self.epoch):
            temp = self._temp(t)
            q = np.random.rand()

            # LLM-guided strategy (or heuristic), applied before updates
            strategy = self._choose_strategy(t, recent)
            self._apply_communication_strategy(strategy)

            # Group-wise updates
            for g in self.groups:
                best = g["g_best"].pos.copy()
                # males
                new_males: List[Individual] = []
                for ind in g["pop_males"]:
                    x_new = _clip(self._update_male(ind.pos, best, temp), self.lb, self.ub)
                    f_new = self._evaluate(x_new)
                    new_males.append(Individual(x_new, f_new) if self._better(f_new, ind.fit) else ind)
                # females
                new_females: List[Individual] = []
                for ind in g["pop_females"]:
                    rnd = np.random.uniform(self.lb, self.ub, size=self.dim)
                    x_new = _clip(self._update_female(ind.pos, best, rnd, temp, exploit=(q >= 0.5)), self.lb, self.ub)
                    f_new = self._evaluate(x_new)
                    new_females.append(Individual(x_new, f_new) if self._better(f_new, ind.fit) else ind)

                # merge and refresh group best
                g["pop_males"], g["pop_females"] = new_males, new_females
                g["pop"] = (g["pop_males"] + g["pop_females"])
                # sort by fitness
                g["pop"].sort(key=lambda ind: ind.fit, reverse=(self.minmax == "max"))
                g["g_best"] = g["pop"][0]

            # Refresh global best & history
            self._refresh_global_best()
            recent.append(self.g_best.fit)
            self.history["best_fitness"].append(self.g_best.fit)

            if self.verbose and (t % max(1, self.epoch // 10) == 0):
                print(f"[t={t:04d}] best = {self.g_best.fit:.6e} (strategy: {strategy}, temp={temp:.3f})")

        return self.g_best.pos.copy(), float(self.g_best.fit)
