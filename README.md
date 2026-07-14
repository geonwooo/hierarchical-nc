# Hierarchical Classification & Neural Collapse Experiments

Research code for studying how hierarchical classifiers, intermediate features, and classifier geometry interact on image classification benchmarks.

The repository is an **active experiment workspace**, not a polished library. It contains the training code, configuration sweeps, Neural Collapse measurements, and ablations used to compare flat and hierarchy-aware classification designs.

## Research questions

The current experiments focus on questions such as:

- When does an explicit coarse-to-fine classifier help over a flat classifier?
- Does passing coarse information into the fine classifier improve discrimination, or only add capacity?
- Which gains survive capacity-matched and structure-matched controls?
- How do hierarchical objectives change Neural Collapse statistics and representation geometry?

## Experimental families

The codebase includes variants for:

- **Flat classification** — standard single-head baseline.
- **Direct / sequential hierarchy** — coarse prediction followed by fine prediction.
- **Residual hierarchy** — fine prediction with a residual hierarchical path.
- **Factorized / per-group classifiers** — class structure encoded in the classifier parameterization.
- **Capacity controls** — MLP width, enlarged backbones, concatenated intermediate features.
- **Optimization and objective ablations** — auxiliary losses, scheduled sampling, cosine classifiers, label smoothing, FiLM, ETF variants, and joint 100-way objectives.
- **Geometry analysis** — Neural Collapse metrics and representation/classifier statistics.

Recent experiment commits include a 40+ run factorized ablation suite and structural fixes for the concat/enlarged/intermediate-channel controls. Treat earlier results produced before those fixes as historical unless reproduced with the current code.

## Repository layout

```text
configs/        Experiment configurations
main/           Training entry points
models/         Backbones and classifier variants
utils/          Training / analysis utilities
RunAllExperiments.sh
                Multi-GPU experiment sweep used for current CIFAR-100 studies
```

The exact layout may evolve while experiments are active. Configuration files are the most reliable record of the condition used for each run.

## Environment

A compatible baseline environment is:

```bash
conda create -n hierarchical-nc python=3.9
conda activate hierarchical-nc
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

Because this is research code, some experiments may depend on the GPU count, local dataset paths, or additional packages used by a specific configuration. Check the target config and run script before launching a long sweep.

## Running experiments

The training entry point used by the current sweep is:

```bash
python main/train.py --cfg <config.yaml> <key value overrides>
```

For the current multi-GPU CIFAR-100 sweep:

```bash
bash RunAllExperiments.sh 0
```

The argument is the random seed. Review the script before running: it launches many experiments concurrently and assumes four visible GPUs.

## Reproducibility notes

For a result to be treated as current, record at least:

- commit SHA,
- configuration file and command-line overrides,
- random seed,
- dataset version / hierarchy definition,
- backbone and feature dimensions,
- whether the run predates a structural bug fix.

This matters especially for architecture comparisons: a hierarchy-aware model can appear better simply because it has a larger feature path or classifier capacity. The repository therefore includes explicit capacity and structure controls rather than comparing only the final headline variants.

## Status

**Active research.** Results, hypotheses, and model definitions may change as controls expose confounds or implementation bugs. The commit history is part of the experimental record.
