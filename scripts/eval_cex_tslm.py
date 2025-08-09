#!/usr/bin/env python3
"""
Evaluate CEX-TSLM on validation set (MSE and optional explanation perplexity).
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from cex_tslm.models.cex_tslm import CEXTSLM, CEXConfig
from cex_tslm.data.time_mmd import TimeMMDDataset, collate_time_mmd
from cex_tslm.utils.metrics import mse_metric


def load_config(path: str):
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "inherit" in cfg:
        base = cfg["inherit"]
        with open(base, "r") as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit"})
        cfg = base_cfg
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = TimeMMDDataset(root=cfg["data"]["root"], split=cfg["data"].get("val_split", "val"),
                        max_docs=cfg["data"].get("max_docs", 16),
                        max_tokens=cfg["data"].get("max_tokens", 64))
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_time_mmd)

    model = CEXTSLM(CEXConfig(**cfg["model"]))
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    mses = []
    with torch.no_grad():
        for batch in dl:
            out = model(batch["ts"], batch["docs_tokens"])
            mses.append(mse_metric(out["forecast"], batch["target"]))
    print(f"Val MSE: {sum(mses)/len(mses):.6f}")


if __name__ == "__main__":
    main()
