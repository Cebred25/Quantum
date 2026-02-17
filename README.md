Quick Start

1. Build chemistry cache
python -m qsar3d.cli.prep \
  --input data/raw/smiles.smi \
  --policies qsar/configs/policies

    Output:
data/cache/conformers/<policy_hash>/
data/cache/graphs_3d/<policy_hash>/

2. Train model
python -m qsar3d.cli.train \
  --policies qsar/configs/policies \
  --model qsar/configs/policies/model \
  --train-config qsar/configs/train/base.yaml
Output:
artifacts/runs/<run_id>/
artifacts/models/<run_id>/

3. Calibrate Uncertianity
python -m qsar3d.cli.calibrate \
  --run artifacts/models/<run_id> \
  --config qsar/configs/calibrate/base.yaml

4. Predict New Molecules
python -m qsar3d.cli.predict \
  --model artifacts/models/<run_id> \
  --input new_molecules.smi
