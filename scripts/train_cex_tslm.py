#!/usr/bin/env python3
"""
Train CEX-TSLM on Time-MMD (or synthetic).
Supports both the lightweight (non-HF) encoders and HuggingFace BERT/GPT-2 via flags/config.
"""
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from cex_tslm.models.cex_tslm import CEXTSLM, CEXConfig
from cex_tslm.models.hf_text import HFTextEncoder, GPT2Decoder
from cex_tslm.data.time_mmd import TimeMMDDataset, collate_time_mmd
from cex_tslm.trainers.trainer import CEXTrainer, TrainConfig as CEXTrainCfg
from ct_gssn.utils.helpers import seed_all


def load_config(path: str):
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
    ap.add_argument("--use_hf", action="store_true", help="use HuggingFace encoders/decoder")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg["train"].get("seed", 1337))

    # Data
    train = TimeMMDDataset(root=cfg["data"]["root"], split=cfg["data"].get("split", "train"),
                           max_docs=cfg["data"].get("max_docs", 16),
                           max_tokens=cfg["data"].get("max_tokens", 64))
    val = TimeMMDDataset(root=cfg["data"]["root"], split=cfg["data"].get("val_split", "val"),
                         max_docs=cfg["data"].get("max_docs", 16),
                         max_tokens=cfg["data"].get("max_tokens", 64))
    dl_tr = DataLoader(train, batch_size=cfg["train"]["batch_size"], shuffle=True,
                       num_workers=cfg["train"]["num_workers"], collate_fn=collate_time_mmd)
    dl_va = DataLoader(val, batch_size=cfg["train"]["batch_size"], shuffle=False,
                       num_workers=cfg["train"]["num_workers"], collate_fn=collate_time_mmd)

    # Model
    if args.use_hf:
        # base backbone for TS encoder still from CEXConfig; text encoder/decoder replaced
        model = CEXTSLM(CEXConfig(**cfg["model"]))
        hf_cfg = cfg.get("hf", {})
        model.txt_enc = HFTextEncoder(
            model_name=hf_cfg.get("text_encoder_model", "bert-base-uncased"),
            cache_dir=hf_cfg.get("cache_dir", None),
            freeze=hf_cfg.get("freeze_text_encoder", True),
            use_auth_token=hf_cfg.get("use_auth_token", None),
            d_out=model.cfg.d_model,
        )
        model.decoder = GPT2Decoder(
            model_name=hf_cfg.get("decoder_model", "gpt2"),
            cache_dir=hf_cfg.get("cache_dir", None),
            freeze=hf_cfg.get("freeze_decoder", False),
            use_auth_token=hf_cfg.get("use_auth_token", None),
        )
    else:
        model = CEXTSLM(CEXConfig(**cfg["model"]))

    trainer = CEXTrainer(
        model=model,
        cfg=CEXTrainCfg(
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
            max_epochs=cfg["train"]["max_epochs"],
            device=cfg["train"]["device"],
            seed=cfg["train"]["seed"],
            alpha_explain=cfg["train"].get("alpha_explain", 0.5),
            beta_causal=cfg["train"].get("beta_causal", 1.0),
            tau=cfg["train"].get("tau", 0.07),
        ),
    )

    trainer.train(dl_tr, dl_va)


if __name__ == "__main__":
    main()
