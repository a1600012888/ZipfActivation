"""Example: load saved statistics from a pipeline run and inspect shapes."""

import numpy as np

results_dir = "results/full_run_v2/statistics"

for layer_name in ["embed", "layer_18", "layer_34"]:
    path = f"{results_dir}/{layer_name}.npz"
    data = np.load(path)

    print(f"=== {layer_name} ===")
    print(f"  Keys: {list(data.keys())}")
    for key in data:
        arr = data[key]
        print(f"  {key:25s}  shape={str(arr.shape):20s}  dtype={arr.dtype}  min={arr.min():.4g}  max={arr.max():.4g}")
    print()
