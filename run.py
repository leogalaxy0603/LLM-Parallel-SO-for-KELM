# --------------------------------------------------------------------------------
# Algorithms:
#   - PSO           : OriginalPSO (PSO.py, standalone, no Mealpy)
#   - SO            : OriginalSO  (SO.py,  standalone, no Mealpy)
#   - PARALLEL_SO   : ParallelSO  (ParallelSO.py, standalone, no Mealpy; multi-group Snake)
#   - LLM_SO        : LLMEnhancedParallelSO (optional, if you enable use_llm)
#
# Parallelization: inner tasks across (algorithm × runs) using ProcessPoolExecutor.
# Benchmarks: opfunu==1.0.4 CEC (F{idx}{year}) or fallback to built-ins.
# Results: one CSV per algorithm under ./results/.
#
# Author: Gao-Yuan Liu
# License: MIT

import os
import time
import importlib
import warnings
from typing import Callable, List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    category=UserWarning
)

# =============== 参数设置 (edit these like MATLAB) ===============
lb = -100.0
ub =  100.0
SearchAgents_no = 60         # 种群规模 pop_size
fun_num = 12                 # 函数数量
Max_iteration = 1000         # 最大代数
dim = 20
runs = 20                    # 每个函数重复次数

# CEC 配置（opfunu 1.0.4 建议使用 cec_based）
USE_CEC = True
CEC_YEAR = 2022              # 可选: 2005/2008/2010/2013/2014/2015/2017/2019/2020/2021/2022
# 完全对齐 CEC2013： fun_num=28

# 算法选择（按需要启用）
ALGOS_TO_RUN = ["PSO", "SO", "PARALLEL_SO", "PPSO", "PCSO", "PMVO", "LLM_SO"]
use_llm = True                       # 仅 LLM_SO 生效
llm_interval = 20

# PSO 超参
PSO_PARAMS = dict(c1=2.05, c2=2.05, w_min=0.4, w_max=0.9)

# SO 超参（蛇优化器单组）
SO_PARAMS = dict(c1=0.5, c2=0.05, c3=2.0)

# PARALLEL_SO 超参（多组蛇优化器）
PARALLEL_SO_GROUPS = 4
PARALLEL_SO_PARAMS = dict(c1=0.5, c2=0.05, c3=2.0, use_enhanced_exploration=True)

# PPSO 超参（并行PSO，多组）
PPSO_GROUPS = 4
PPSO_PARAMS = dict(c1=1.2, c2=1.2, v_clamp_frac=0.1, use_inertia=False)

# PCSO 超参（并行猫群优化）
PCSO_GROUPS = 4
PCSO_PARAMS = dict(mr=0.15, srd=0.2, smp=5, c=2.0, vel_min=-1.0, vel_max=1.0)

# PMVO 超参（并行多元宇宙优化）
PMVO_GROUPS = 4
PMVO_PARAMS = dict(wep_min=0.2, wep_max=1.0)

# LLM-SO（分组 + 可选 DeepSeek 决策）
LLM_SO_GROUPS = 4
LLM_SO_MALE_RATIO = 0.5
LLM_SO_PARAMS = dict(c1=0.5, c2=0.05, c3=2.0)

# 并行配置
PARALLEL_RUNS = True
N_WORKERS = max(1, (os.cpu_count() or 2))   # 也可以手动指定，如 16 或 24




# ==========================================================
# =============== 内置函数（当无法使用 opfunu 时） ===============
def builtin_eval(x: np.ndarray, name: str) -> float:
    x = np.asarray(x, dtype=float).ravel()
    if name == "rastrigin":
        A = 10.0
        return float(A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
    elif name == "ackley":
        a, b, c = 20.0, 0.2, 2*np.pi
        d = x.size
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x*x)/d))
        term2 = -np.exp(np.sum(np.cos(c*x))/d)
        return float(term1 + term2 + a + np.e)
    else:
        raise ValueError(f"Unknown builtin function: {name}")

BUILTIN_LIST = [
    ("builtin_rastrigin", "rastrigin", -5.12, 5.12),
    ("builtin_ackley",    "ackley",   -32.768, 32.768),
]

# =============== 构建基准列表（主进程，仅做描述，不传闭包） ===============
def _import_cec_class(year: int, idx: int):
    class_name = f"F{idx}{year}"
    try:
        mod = importlib.import_module("opfunu.cec_based")
        cls = getattr(mod, class_name)
        return cls
    except Exception:
        mod = importlib.import_module(f"opfunu.cec_based.cec{year}")
        cls = getattr(mod, class_name)
        return cls

def build_cec_benchmarks(dim: int, desired_num: int, year: int):
    funcs = []
    try:
        importlib.import_module("opfunu")
    except Exception:
        return funcs

    for i in range(1, desired_num + 1):
        try:
            cls = _import_cec_class(year, i)
            try:
                inst = cls(ndim=dim)
            except TypeError:
                inst = cls()
            if hasattr(inst, "lb") and hasattr(inst, "ub"):
                lb_arr = np.array(inst.lb, dtype=float).ravel()
                ub_arr = np.array(inst.ub, dtype=float).ravel()
            elif hasattr(inst, "bounds"):
                b = np.array(inst.bounds, dtype=float)
                lb_arr, ub_arr = b[0], b[1]
            else:
                lb_arr = np.full(dim, lb, dtype=float)
                ub_arr = np.full(dim, ub, dtype=float)
            if lb_arr.size != dim:
                lb_arr = np.resize(lb_arr, dim)
                ub_arr = np.resize(ub_arr, dim)
            desc = {"kind": "cec", "year": year, "idx": i, "name": f"F{i}{year}"}
            funcs.append((desc, lb_arr, ub_arr))
        except Exception as e:
            print(f"[WARN] CEC{year} F{i}: {e}")
    return funcs

def build_benchmark_list(dim: int, desired_num: int):
    lst = []
    if USE_CEC:
        ceclist = build_cec_benchmarks(dim, desired_num, CEC_YEAR)
        if ceclist:
            lst.extend(ceclist[:desired_num])
    if not lst:
        print("[INFO] 使用内置基准函数（未安装或无法导入 opfunu CEC 套件）。")
        for name, bname, l, u in BUILTIN_LIST:
            desc = {"kind": "builtin", "bname": bname, "name": name}
            lst.append((desc, np.full(dim, l), np.full(dim, u)))

    if len(lst) < desired_num:
        base = lst.copy()
        while len(lst) < desired_num and base:
            idx = len(lst) - len(base)
            desc, lb_arr, ub_arr = base[(len(lst)) % len(base)]
            desc2 = desc.copy()
            desc2["name"] = f'{desc2["name"]}_{idx+1}'
            lst.append((desc2, lb_arr, ub_arr))
    return lst[:desired_num]

# =============== 子进程内重建 evaluator（可被 pickle 调用） ===============
def _make_evaluator(desc: Dict[str, Any], dim: int):
    kind = desc["kind"]
    if kind == "cec":
        year = desc["year"]; idx = desc["idx"]
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
            def f(x): return float(inst.evaluate(np.asarray(x, dtype=float).ravel()))
        else:
            def f(x): return float(inst(np.asarray(x, dtype=float).ravel()))
        return f
    elif kind == "builtin":
        bname = desc["bname"]
        def f(x): return float(builtin_eval(np.asarray(x, dtype=float).ravel(), bname))
        return f
    else:
        raise ValueError(f"Unknown desc kind: {kind}")

# =============== 单次运行 Worker（必须顶层定义） ===============
def _single_run_worker(args):

    (algo, i, j, dim, lb_arr, ub_arr, desc,
     Max_iteration, pop_size, seed, algo_params,
     use_llm, api_key, llm_interval,
     llm_groups, llm_male_ratio,
     pso_params, so_params, pso_parallel_groups, pso_parallel_extra,
     pcso_groups, pcso_params, pmvo_groups, pmvo_params) = args
    try:
        f = _make_evaluator(desc, dim)

        if algo == "PSO":
            from PSO import OriginalPSO
            opt = OriginalPSO(
                dim=dim, lb=lb_arr, ub=ub_arr, fit_func=f, minmax="min",
                epoch=Max_iteration, pop_size=pop_size,
                c1=pso_params.get("c1", 2.05),
                c2=pso_params.get("c2", 2.05),
                w_min=pso_params.get("w_min", 0.4),
                w_max=pso_params.get("w_max", 0.9),
                seed=seed, verbose=False
            )
        elif algo == "SO":
            from SO import OriginalSO
            opt = OriginalSO(
                dim=dim, lb=lb_arr, ub=ub_arr, fit_func=f, minmax="min",
                epoch=Max_iteration, pop_size=pop_size,
                c1=so_params.get("c1", 0.5),
                c2=so_params.get("c2", 0.05),
                c3=so_params.get("c3", 2.0),
                seed=seed, verbose=False
            )
        elif algo == "PARALLEL_SO":
            from ParallelSO import ParallelSO
            opt = ParallelSO(
                dim=dim, lb=lb_arr, ub=ub_arr, fit_func=f, minmax="min",
                epoch=Max_iteration, pop_size=pop_size,
                c1=pso_parallel_extra.get("c1", 0.5),
                c2=pso_parallel_extra.get("c2", 0.05),
                c3=pso_parallel_extra.get("c3", 2.0),
                n_groups=pso_parallel_groups,
                seed=seed, verbose=False,
                use_enhanced_exploration=pso_parallel_extra.get("use_enhanced_exploration", True)
            )
        elif algo == "PPSO":
            from PPSO import PPSO
            opt = PPSO(
                dim=dim, lb=lb_arr, ub=ub_arr, fit_func=f, minmax="min",
                epoch=Max_iteration, pop_size=pop_size,
                groups=algo_params.get("groups", 4),
                c1=algo_params.get("c1", 1.2),
                c2=algo_params.get("c2", 1.2),
                v_clamp_frac=algo_params.get("v_clamp_frac", 0.1),
                use_inertia=algo_params.get("use_inertia", False),
                seed=seed, verbose=False
            )
        elif algo == "PCSO":
            from PCSO import PCSO
            opt = PCSO(
                dim=dim, lb=lb_arr, ub=ub_arr, fit_func=f, minmax="min",
                epoch=Max_iteration, pop_size=pop_size,
                n_groups=pcso_groups,
                mr=pcso_params.get("mr", 0.15),
                srd=pcso_params.get("srd", 0.2),
                smp=pcso_params.get("smp", 5),
                c=pcso_params.get("c", 2.0),
                vel_min=pcso_params.get("vel_min", -1.0),
                vel_max=pcso_params.get("vel_max", 1.0),
                seed=seed, verbose=False
            )
        elif algo == "PMVO":
            from PMVO import PMVO
            opt = PMVO(
                dim=dim, lb=lb_arr, ub=ub_arr, fit_func=f, minmax="min",
                epoch=Max_iteration, pop_size=pop_size,
                n_groups=pmvo_groups,
                wep_min=pmvo_params.get("wep_min", 0.2),
                wep_max=pmvo_params.get("wep_max", 1.0),
                seed=seed, verbose=False
            )
        elif algo == "LLM_SO":
            from LLMEnhancedParallelSO import LLMEnhancedParallelSO
            opt = LLMEnhancedParallelSO(
                dim=dim, lb=lb_arr, ub=ub_arr, fit_func=f, minmax="min",
                epoch=Max_iteration, pop_size=pop_size, n_groups=llm_groups,
                male_ratio=llm_male_ratio,
                c1=algo_params.get("c1", 0.5),
                c2=algo_params.get("c2", 0.05),
                c3=algo_params.get("c3", 2.0),
                seed=seed, verbose=False,
                deepseek_api_key=(api_key if use_llm else None),
                llm_interval=llm_interval
            )
        else:
            raise ValueError(f"Unknown algo: {algo}")

        _, best_fit = opt.solve()
        return (algo, j-1, best_fit, None)
    except Exception as e:
        return (algo, j-1, np.nan, str(e))

# =============== 主循环（函数串行，算法×runs 并行） ===============
def main():
    bench_list = build_benchmark_list(dim, fun_num)
    real_fun_num = len(bench_list)
    if real_fun_num != fun_num:
        print(f"[INFO] fun_num={fun_num} 与可用函数数 {real_fun_num} 不一致，已采用 {real_fun_num}.")

    # DeepSeek API (仅 LLM_SO 用)
    api_key = os.getenv("DEEPSEEK_API_KEY") if use_llm else None
    if ("LLM_SO" in ALGOS_TO_RUN) and use_llm and not api_key:
        print("[WARN] use_llm=True 但未设置 DEEPSEEK_API_KEY，将退回启发式策略。")

    # 准备结果矩阵
    results = {}
    for algo in ALGOS_TO_RUN:
        results[algo] = np.zeros((real_fun_num, runs), dtype=float)

    t0_all = time.time()
    for i, (desc, lb_arr, ub_arr) in enumerate(bench_list, start=1):
        fname = desc["name"]
        print(f"\n========== Function {i}/{real_fun_num}: {fname} (dim={dim}) ==========")
        t0 = time.time()

        # 构建所有 (algo × runs) 任务
        tasks = []
        for algo in ALGOS_TO_RUN:
            for j in range(1, runs + 1):
                # 给不同算法加偏移，避免完全相同的随机序列（简单可控）
                base = {"PSO": 0, "SO": 1, "PARALLEL_SO": 2, "PPSO": 3, "PCSO": 4, "PMVO": 5, "LLM_SO": 6}.get(algo, 9)
                seed = (i * 100000) + (j * 10) + base
                if algo == "PSO":
                    aparams = PSO_PARAMS
                elif algo == "SO":
                    aparams = SO_PARAMS
                elif algo == "PARALLEL_SO":
                    aparams = PARALLEL_SO_PARAMS
                elif algo == "PPSO":
                    aparams = PPSO_PARAMS.copy()
                    aparams["groups"] = PPSO_GROUPS
                elif algo == "PCSO":
                    aparams = PCSO_PARAMS
                elif algo == "PMVO":
                    aparams = PMVO_PARAMS
                else:
                    aparams = LLM_SO_PARAMS
                tasks.append((algo, i, j, dim, lb_arr, ub_arr, desc,
                              Max_iteration, SearchAgents_no, seed, aparams,
                              use_llm, api_key, llm_interval,
                              LLM_SO_GROUPS, LLM_SO_MALE_RATIO,
                              PSO_PARAMS, SO_PARAMS, PARALLEL_SO_GROUPS, PARALLEL_SO_PARAMS,
                              PCSO_GROUPS, PCSO_PARAMS, PMVO_GROUPS, PMVO_PARAMS))

        if PARALLEL_RUNS and N_WORKERS > 1:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                futures = [ex.submit(_single_run_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    algo, col, best_fit, err = fut.result()
                    results[algo][i-1, col] = best_fit
                    if err:
                        print(f"  [{algo}] run {col+1:02d}/{runs} ERR: {err}")
        else:
            for t in tasks:
                algo, col, best_fit, err = _single_run_worker(t)
                results[algo][i-1, col] = best_fit
                if err:
                    print(f"  [{algo}] run {col+1:02d}/{runs} ERR: {err}")

        elapsed = time.time() - t0
        # 打印这一函数在每个算法下的统计
        for algo in ALGOS_TO_RUN:
            vals = results[algo][i-1]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            best = np.nanmin(vals)
            print(f"  -> {algo:<12s} | mean={mean:.6e}  std={std:.6e}  best={best:.6e}")
        print(f"  [time] {elapsed:.1f}s")

    total_elapsed = time.time() - t0_all
    print(f"\nAll done in {total_elapsed:.1f}s.")

    # 结果保存：每个算法各存一个 CSV 到 ./results/
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    for algo in ALGOS_TO_RUN:
        out_path = os.path.join(out_dir, f"{algo}_results.csv")
        np.savetxt(out_path, results[algo], delimiter=",")
        print(f"Saved results to {out_path}")

    # 总结（逐算法）
    print("\nPer-function summary (mean | best) by algorithm:")
    for algo in ALGOS_TO_RUN:
        means = np.nanmean(results[algo], axis=1)
        bests = np.nanmin(results[algo], axis=1)
        print(f"== {algo} ==")
        for i, (desc, _, _) in enumerate(bench_list, start=1):
            print(f"  {i:02d} {desc['name']:<20s}  mean={means[i-1]:.6e}  best={bests[i-1]:.6e}")

if __name__ == "__main__":
    main()