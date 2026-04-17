# run.py (MATLAB-like batch runner for PyCharm) - LLM-Parallel SO
# ------------------------------------------------------------------------------
# Paper:
#   A Novel Parallel Snake Optimizer with LLM-Enhanced Cross-Population
#   Communication for KELM Hyperparameter Optimization and Applied in
#   Power Load Forecasting
# Authors:
#   Tian-Yu Gao, Gao-Yuan Liu, Joel J. P. C. Rodrigues, Jeng-Shyang Pan,
#   Hui-Qi Zhao, Ying Yu and Ru-Yu Wang
#
# Algorithm:
#   - LLM-Parallel SO : LLMEnhancedParallelSO
#
# Parallelization: inner tasks across runs using ProcessPoolExecutor.
# Benchmarks: opfunu==1.0.4 CEC (F{idx}{year}) or fallback to built-ins.
# Results: one CSV under ./results/.
#
# License: MIT

import importlib
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict

import numpy as np

# --------------- CPU/BLAS threading hygiene ---------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

# ---- Silence the noisy warning coming from opfunu.cec_based.cec (UserWarning) ----
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# =============== Parameter settings (edit these like MATLAB) ===============
lb = -100.0
ub = 100.0
SearchAgents_no = 60         # Population size passed to LLM-Parallel SO
fun_num = 12                 # Number of benchmark functions
Max_iteration = 1000         # Number of iterations
dim = 20
runs = 20                    # Independent runs per function

# CEC settings (recommended for opfunu 1.0.4 cec_based)
USE_CEC = True
CEC_YEAR = 2022              # Available: 2005/2008/2010/2013/2014/2015/2017/2019/2020/2021/2022

# Only keep LLM-Parallel SO for the open-source release
ALGO_NAME = "LLM-Parallel SO"
use_llm = True
llm_interval = 20
LLM_SO_GROUPS = 4
LLM_SO_MALE_RATIO = 0.5
LLM_SO_PARAMS = dict(c1=0.5, c2=0.05, c3=2.0)

# Parallel execution settings
PARALLEL_RUNS = True
N_WORKERS = max(1, (os.cpu_count() or 2))


# ==========================================================
# =============== Built-in fallback benchmark functions ===============
def builtin_eval(x: np.ndarray, name: str) -> float:
    x = np.asarray(x, dtype=float).ravel()
    if name == "rastrigin":
        a = 10.0
        return float(a * x.size + np.sum(x**2 - a * np.cos(2 * np.pi * x)))
    if name == "ackley":
        a, b, c = 20.0, 0.2, 2 * np.pi
        d = x.size
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x * x) / d))
        term2 = -np.exp(np.sum(np.cos(c * x)) / d)
        return float(term1 + term2 + a + np.e)
    raise ValueError(f"Unknown builtin function: {name}")


BUILTIN_LIST = [
    ("builtin_rastrigin", "rastrigin", -5.12, 5.12),
    ("builtin_ackley", "ackley", -32.768, 32.768),
]


# =============== Build benchmark descriptors in the main process ===============
def _import_cec_class(year: int, idx: int):
    class_name = f"F{idx}{year}"
    try:
        mod = importlib.import_module("opfunu.cec_based")
        return getattr(mod, class_name)
    except Exception:
        mod = importlib.import_module(f"opfunu.cec_based.cec{year}")
        return getattr(mod, class_name)


def build_cec_benchmarks(dim: int, desired_num: int, year: int):
    funcs = []
    try:
        importlib.import_module("opfunu")
    except Exception:
        return funcs

    for idx in range(1, desired_num + 1):
        try:
            cls = _import_cec_class(year, idx)
            try:
                inst = cls(ndim=dim)
            except TypeError:
                inst = cls()

            if hasattr(inst, "lb") and hasattr(inst, "ub"):
                lb_arr = np.array(inst.lb, dtype=float).ravel()
                ub_arr = np.array(inst.ub, dtype=float).ravel()
            elif hasattr(inst, "bounds"):
                bounds = np.array(inst.bounds, dtype=float)
                lb_arr, ub_arr = bounds[0], bounds[1]
            else:
                lb_arr = np.full(dim, lb, dtype=float)
                ub_arr = np.full(dim, ub, dtype=float)

            if lb_arr.size != dim:
                lb_arr = np.resize(lb_arr, dim)
                ub_arr = np.resize(ub_arr, dim)

            desc = {"kind": "cec", "year": year, "idx": idx, "name": f"F{idx}{year}"}
            funcs.append((desc, lb_arr, ub_arr))
        except Exception as exc:
            print(f"[WARN] CEC{year} F{idx}: {exc}")
    return funcs


def build_benchmark_list(dim: int, desired_num: int):
    benchmarks = []
    if USE_CEC:
        cec_list = build_cec_benchmarks(dim, desired_num, CEC_YEAR)
        if cec_list:
            benchmarks.extend(cec_list[:desired_num])

    if not benchmarks:
        print("[INFO] Falling back to built-in benchmark functions because opfunu CEC is unavailable.")
        for name, builtin_name, lower, upper in BUILTIN_LIST:
            desc = {"kind": "builtin", "bname": builtin_name, "name": name}
            benchmarks.append((desc, np.full(dim, lower), np.full(dim, upper)))

    if len(benchmarks) < desired_num:
        base = benchmarks.copy()
        while len(benchmarks) < desired_num and base:
            src_desc, lb_arr, ub_arr = base[len(benchmarks) % len(base)]
            desc = src_desc.copy()
            desc["name"] = f"{desc['name']}_{len(benchmarks) - len(base) + 1}"
            benchmarks.append((desc, lb_arr, ub_arr))

    return benchmarks[:desired_num]


# =============== Rebuild evaluator inside worker processes ===============
def _make_evaluator(desc: Dict[str, Any], dim: int):
    kind = desc["kind"]
    if kind == "cec":
        year = desc["year"]
        idx = desc["idx"]
        class_name = f"F{idx}{year}"
        try:
            mod = importlib.import_module("opfunu.cec_based")
            cls = getattr(mod, class_name)
        except Exception:
            mod = importlib.import_module(f"opfunu.cec_based.cec{year}")
            cls = getattr(mod, class_name)

        try:
            inst = cls(ndim=dim)
        except TypeError:
            inst = cls()

        if hasattr(inst, "evaluate"):
            def evaluator(x):
                return float(inst.evaluate(np.asarray(x, dtype=float).ravel()))
        else:
            def evaluator(x):
                return float(inst(np.asarray(x, dtype=float).ravel()))
        return evaluator

    if kind == "builtin":
        builtin_name = desc["bname"]

        def evaluator(x):
            return float(builtin_eval(np.asarray(x, dtype=float).ravel(), builtin_name))

        return evaluator

    raise ValueError(f"Unknown desc kind: {kind}")


# =============== Single-run worker ===============
def _single_run_worker(args):
    (
        run_idx,
        dim,
        lb_arr,
        ub_arr,
        desc,
        max_iteration,
        pop_size,
        seed,
        algo_params,
        use_llm,
        api_key,
        llm_interval,
        llm_groups,
        llm_male_ratio,
    ) = args

    try:
        from LLMEnhancedParallelSO import LLMEnhancedParallelSO

        evaluator = _make_evaluator(desc, dim)
        opt = LLMEnhancedParallelSO(
            dim=dim,
            lb=lb_arr,
            ub=ub_arr,
            fit_func=evaluator,
            minmax="min",
            epoch=max_iteration,
            pop_size=pop_size,
            n_groups=llm_groups,
            male_ratio=llm_male_ratio,
            c1=algo_params.get("c1", 0.5),
            c2=algo_params.get("c2", 0.05),
            c3=algo_params.get("c3", 2.0),
            seed=seed,
            verbose=False,
            deepseek_api_key=(api_key if use_llm else None),
            llm_interval=llm_interval,
        )

        _, best_fit = opt.solve()
        return run_idx - 1, best_fit, None
    except Exception as exc:
        return run_idx - 1, np.nan, str(exc)


def main():
    bench_list = build_benchmark_list(dim, fun_num)
    real_fun_num = len(bench_list)
    if real_fun_num != fun_num:
        print(f"[INFO] fun_num={fun_num} differs from available benchmarks {real_fun_num}; using {real_fun_num}.")

    api_key = os.getenv("DEEPSEEK_API_KEY") if use_llm else None
    if use_llm and not api_key:
        print("[WARN] use_llm=True but DEEPSEEK_API_KEY is not set; falling back to heuristic strategy selection.")

    results = np.zeros((real_fun_num, runs), dtype=float)

    t0_all = time.time()
    for func_idx, (desc, lb_arr, ub_arr) in enumerate(bench_list, start=1):
        fname = desc["name"]
        print(f"\n========== Function {func_idx}/{real_fun_num}: {fname} (dim={dim}) ==========")
        t0 = time.time()

        tasks = []
        for run_idx in range(1, runs + 1):
            seed = (func_idx * 100000) + run_idx
            tasks.append(
                (
                    run_idx,
                    dim,
                    lb_arr,
                    ub_arr,
                    desc,
                    Max_iteration,
                    SearchAgents_no,
                    seed,
                    LLM_SO_PARAMS,
                    use_llm,
                    api_key,
                    llm_interval,
                    LLM_SO_GROUPS,
                    LLM_SO_MALE_RATIO,
                )
            )

        if PARALLEL_RUNS and N_WORKERS > 1:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = [executor.submit(_single_run_worker, task) for task in tasks]
                for future in as_completed(futures):
                    col, best_fit, err = future.result()
                    results[func_idx - 1, col] = best_fit
                    if err:
                        print(f"  [{ALGO_NAME}] run {col + 1:02d}/{runs} ERR: {err}")
        else:
            for task in tasks:
                col, best_fit, err = _single_run_worker(task)
                results[func_idx - 1, col] = best_fit
                if err:
                    print(f"  [{ALGO_NAME}] run {col + 1:02d}/{runs} ERR: {err}")

        elapsed = time.time() - t0
        vals = results[func_idx - 1]
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        best = np.nanmin(vals)
        print(f"  -> {ALGO_NAME:<12s} | mean={mean:.6e}  std={std:.6e}  best={best:.6e}")
        print(f"  [time] {elapsed:.1f}s")

    total_elapsed = time.time() - t0_all
    print(f"\nAll done in {total_elapsed:.1f}s.")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ALGO_NAME}_results.csv")
    np.savetxt(out_path, results, delimiter=",")
    print(f"Saved results to {out_path}")

    print(f"\nPer-function summary (mean | best) for {ALGO_NAME}:")
    means = np.nanmean(results, axis=1)
    bests = np.nanmin(results, axis=1)
    for func_idx, (desc, _, _) in enumerate(bench_list, start=1):
        print(f"  {func_idx:02d} {desc['name']:<20s}  mean={means[func_idx - 1]:.6e}  best={bests[func_idx - 1]:.6e}")


if __name__ == "__main__":
    main()
