#!/usr/bin/env python3
"""
High-level CLI wrapper around app.train.train()

Examples
========
# vanilla run
scripts/train_cli.py --batch-size 32 --lr 4e-4 --max-epochs 12

# hyper-param sweep loop
for lr in 3e-4 4e-4 5e-4; do
  scripts/train_cli.py --run-name "sweep_lr_${lr}" --lr $lr --min-delta 0.001
done
"""
import importlib
import argparse
from pathlib import Path
import json

train_mod = importlib.import_module("app.train")  # dynamic import keeps package layout intact
train_fn  = train_mod.train

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
for k, v in train_mod.DEFAULTS.items():
    parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
parser.add_argument("--run-name", type=str, default="default")
parser.add_argument("--resume",   type=str, help="Checkpoint to resume from (.pt)")
parser.add_argument("--cfg-json", type=str, help="Path to JSON file of overrides")

args = vars(parser.parse_args())

# optional JSON overrides
if args["cfg_json"]:
    overrides = json.loads(Path(args["cfg_json"]).read_text())
    args.update(overrides)

print("⇢  Launching training run", args["run_name"])
train_fn(**args)