# LLM-Parallel-SO-for-KELM

This repository contains the open-source implementation of the method presented in the paper:

**A Novel Parallel Snake Optimizer with LLM-Enhanced Cross-Population Communication for KELM Hyperparameter Optimization and Applied in Power Load Forecasting**

Accepted by **IEEE Industry Applications Magazine**.

## Overview

This project provides a standalone implementation of **LLM-Parallel SO**, a parallel snake-optimizer-style method with optional LLM-enhanced cross-population communication.

Current release highlights:

- Only the `LLM-Parallel SO` algorithm is retained in the open-source batch runner.
- The optimizer implementation is self-contained in `LLMEnhancedParallelSO.py`.
- Benchmark evaluation supports `opfunu==1.0.4` CEC functions and falls back to built-in functions when CEC benchmarks are unavailable.
- Batch execution is handled by `run.py`, and results are saved as CSV files under `./results/`.

## Paper Information

- Title: **A Novel Parallel Snake Optimizer with LLM-Enhanced Cross-Population Communication for KELM Hyperparameter Optimization and Applied in Power Load Forecasting**
- Journal: **IEEE Industry Applications Magazine**
- Authors:
  Tian-Yu Gao, Gao-Yuan Liu, Joel J. P. C. Rodrigues, Jeng-Shyang Pan, Hui-Qi Zhao, Ying Yu, and Ru-Yu Wang

## Project Structure

```text
.
|-- LLMEnhancedParallelSO.py
|-- run.py
|-- requirements.txt
`-- README.md
```

- `run.py`: batch runner for benchmark experiments.
- `LLMEnhancedParallelSO.py`: core optimizer implementation.
- `requirements.txt`: Python dependencies.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Dependencies currently include:

- `numpy`
- `requests`
- `opfunu==1.0.4`

## Usage

Run the project with:

```bash
python run.py
```

The default script configuration is defined directly in `run.py`, including:

- search bounds
- population size
- benchmark dimension
- number of runs
- maximum iterations
- CEC benchmark year
- LLM communication interval

If needed, you can edit these parameters directly near the top of `run.py` before execution.

## API Key Configuration

This project optionally uses a **DeepSeek API key** for LLM-guided communication strategy selection.

The code reads the API key from the environment variable:

```text
DEEPSEEK_API_KEY
```

### PowerShell

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
python run.py
```

### CMD

```cmd
set DEEPSEEK_API_KEY=your_api_key
python run.py
```

### Bash

```bash
export DEEPSEEK_API_KEY="your_api_key"
python run.py
```

If no API key is provided, the program will still run and automatically fall back to heuristic strategy selection.

## Output

After execution, the program creates a `results` directory and saves the benchmark results as:

```text
./results/LLM-Parallel SO_results.csv
```

The script also prints per-function statistics in the terminal, including:

- mean fitness
- standard deviation
- best fitness
- execution time

## Notes

- This repository is focused on the `LLM-Parallel SO` implementation only.
- The optimizer is implemented without relying on Mealpy.
- CEC benchmarks are loaded through `opfunu`. If `opfunu` is unavailable, the runner falls back to built-in benchmark functions.
- LLM guidance is optional rather than mandatory.

## Citation

Official citation information is coming soon.

If you use this code for now, please cite the project repository:

```text
https://github.com/leogalaxy0603/LLM-Parallel-SO-for-KELM
```

## License

This project is released under the MIT License.
