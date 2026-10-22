"""Run the frozen GPU matrix in reference/clean/augmented order, with resumable results."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from turkish_punctuation.gpu_evaluation import write_in_place

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads((ROOT / "configs/robustness-grid.json").read_text())
    output = args.storage_root / matrix["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    for config_path in matrix["configuration_order"]:
        config = json.loads((ROOT / config_path).read_text())
        write_in_place(output / "progress.json", {"status": "running", "model": config["run_name"],
                       "completed_models": completed, "total_models": 3})
        destination = args.storage_root / config["output_directory"]
        command = [sys.executable, "-u", "scripts/evaluate_noise.py", "--config", config_path,
                   "--input-root", str(args.input_root), "--output-root", str(args.storage_root),
                   "--storage-root", str(args.storage_root), "--calibration", str(args.calibration)]
        if destination.exists():
            command.append("--resume")
        print(f"STARTING_MODEL {config['run_name']}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        summary = json.loads((destination / "summary.json").read_text())
        if len(summary["conditions"]) != 38:
            raise ValueError("Expected all 38 prespecified noisy conditions.")
        completed.append(config["run_name"])
        print(f"COMPLETED_MODEL {config['run_name']}", flush=True)
    subprocess.run([sys.executable, "scripts/analyze_robustness_grid.py",
                    "--input-root", str(args.input_root), "--storage-root", str(args.storage_root)],
                   cwd=ROOT, check=True)
    write_in_place(output / "progress.json", {"status": "complete", "completed_models": completed,
                   "model_evaluations": 117, "training_performed": False, "test_evaluated": False})
    print("ROBUSTNESS_GRID_COMPLETE — 3 models × 39 conditions; verified report ready.", flush=True)


if __name__ == "__main__":
    main()
