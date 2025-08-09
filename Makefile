.PHONY: setup train-ct synth cee train-cex hf test fmt

setup:
	pip install -e ".[all]"

train-ct:
	python scripts/train_ct_gssn.py --config configs/ct_gssn/synthetic.yaml

train-ct-metr:
	python scripts/convert_metr_la.py --out_root data/metr-la_proc
	python scripts/train_ct_gssn.py --config configs/ct_gssn/metr_la.yaml

train-cex:
	python scripts/convert_time_mmd.py --synthesize
	python scripts/train_cex_tslm.py --config configs/cex_tslm/time_mmd.yaml

train-cex-hf:
	python scripts/convert_time_mmd.py --synthesize
	python scripts/train_cex_tslm.py --config configs/cex_tslm/hf_bert_gpt2.yaml --use_hf

test:
	pytest -q

fmt:
	ruff check --fix || true
