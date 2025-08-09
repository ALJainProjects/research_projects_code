# CT-GSSN & CEX-TSLM

This repository contains two research-ready deep learning pipelines:

- **CT-GSSN** – Causal-Temporal Graph Spatio-Sequence Network for forecasting and causal attribution on multivariate time series.
- **CEX-TSLM** – Causal-Explanation Time Series + Language Model for joint forecasting and natural language explanation.

The codebase supports:
- PyTorch implementations with modular encoders, decoders, and loss functions
- Synthetic and real-world datasets (e.g., METR-LA, MIMIC-III, TIME-MMD)
- Training & evaluation scripts
- TensorBoard & Weights & Biases logging
- Dockerized GPU-ready environment

---

## 📦 Installation

### **Local Install**
```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
pip install --upgrade pip
pip install -e ".[all]"
```

### **Docker Build**
Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU passthrough.

```bash
docker build -t ctgssn:latest .
```

---

## 🚀 Quick Start

### **Run CT-GSSN Training**
```bash
docker run --rm --gpus all \
  -v "$PWD":/workspace \
  --shm-size=1g \
  ctgssn:latest \
  python scripts/train_ct_gssn.py --config configs/ct_gssn/metr_la.yaml
```

### **Run CEX-TSLM Training**
```bash
docker run --rm --gpus all \
  -v "$PWD":/workspace \
  --shm-size=1g \
  ctgssn:latest \
  python scripts/train_cex_tslm.py --config configs/cex_tslm/time_mmd.yaml
```

### **Preprocess Datasets**
```bash
docker run --rm \
  -v "$PWD":/workspace \
  ctgssn:latest \
  python scripts/preprocess_templates.py --dataset all
```

---

## 📂 Project Structure

```
.
├── configs/                     # YAML configs for training runs
│   ├── ct_gssn/                  # CT-GSSN configs
│   └── cex_tslm/                 # CEX-TSLM configs
├── scripts/                      # Entry point scripts
│   ├── train_ct_gssn.py
│   ├── train_cex_tslm.py
│   ├── preprocess_templates.py
├── src/
│   ├── ct_gssn/                   # CT-GSSN model, utils, data
│   └── cex_tslm/                  # CEX-TSLM model, utils, data
├── Dockerfile
└── README.md
```

---

## 📊 Logging & Visualization

- **TensorBoard**  
  Launch:
  ```bash
  tensorboard --logdir runs
  ```
- **Weights & Biases**  
  Enable with `use_wandb: true` in the config or pass `--use_wandb` at runtime.

---

## 📜 Config Overview

Configs define:
- Model architecture (`d_model`, `num_layers`, etc.)
- Dataset paths & preprocessing
- Training hyperparameters (`lr`, `batch_size`, `max_epochs`)
- Logging & checkpointing options

Example: `configs/ct_gssn/metr_la.yaml`
```yaml
model:
  d_model: 256
  num_layers: 4
train:
  lr: 1e-4
  batch_size: 8
  max_epochs: 50
logging:
  use_wandb: true
  log_dir: runs/ct_gssn
```

---

## 🛠 Development Notes

- All random seeds are controlled via `seed_all()` for reproducibility.
- Models are modular—swap encoders/decoders easily.
- Causal contrastive loss supervises cross-attention alignment.
- Datasets are expected to be preprocessed into `.npz` or cached formats.

---

## 📄 Citation

If you use this code, please cite:

```
@misc{ctgssn_cextslm2025,
  author = Arnav Jain,
  title = {CT-GSSN and CEX-TSLM: Joint Causal Forecasting and Explanation},
  year = {2025},
  url = https://github.com/ALJainProjects/research_projects_code
}
```

---

## 🧠 Related Work
- CT-GSSN builds on spatio-temporal graph learning.
- CEX-TSLM combines multimodal causal attention with sequence-to-sequence language modeling for explanations.

---

## 📬 Contact
For questions or issues, open a GitHub issue
