# experment_deepsparse

## Install instruction

### Make conda env

```bash
conda create -n exp_py38 python=3.8
conda activate exp_py38
pip install torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Download file weight

```bash
./get_weight.sh
```

## Run experiment

```bash
conda activate exp_py38
python exp1.py
```
