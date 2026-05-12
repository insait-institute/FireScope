# FireScope: Wildfire Risk Raster Prediction with a Chain-of-Thought Oracle

Official code repository for the paper:

**FireScope: Wildfire Risk Raster Prediction with a Chain-of-Thought Oracle**\
Accepted to **CVPR 2026**

📄 Paper: https://arxiv.org/abs/2511.17171
🤗 Dataset:
https://huggingface.co/datasets/INSAIT-Institute/FireScope-Bench
🔥 Website:
www.firescope.ai

------------------------------------------------------------------------

## Overview

FireScope introduces a framework for **wildfire risk raster prediction** that
leverages **Chain-of-Thought (CoT) reasoning via an Oracle model** to
improve predictive performance and interpretability.

This repository contains the full codebase used for:

-   Oracle training
-   Subsequent Encoder-Decoder trainings
-   Ablations
-   Evaluation and benchmarking

The dataset used in the experiments (**FireScope-Bench**) combines
Sentinel‑2 imagery, climate data, expert-defined wildfire risk rasters, and real wildfire events to
enable reasoning‑based wildfire risk modeling and cross‑continental
generalization experiments.

------------------------------------------------------------------------

## Dataset

The **FireScope-Bench** dataset is available on Hugging Face:

https://huggingface.co/datasets/INSAIT-Institute/FireScope-Bench

Please download the dataset and place it inside your configured
`DATA_DIR` (see configuration section below).

------------------------------------------------------------------------

## Repository Structure

    .
    ├── custom_datasets/     # datasets used for training and evaluating models
    ├── data_generation/     # code for downloading alphaearth embeddings and generating Oracle predictions
    ├── evaluation/          # generation and evaluation of results
    ├── models/              # models used for all experiments
    ├── prompts/             # prompts used for all experiments
    ├── rewards/             # GRPO reward components
    ├── training/            # all trainings
    └── config.py            # configuration (dataset paths etc.)

------------------------------------------------------------------------

## Configuration

The repository uses a **hardcoded data directory**.

In `config.py` you will find:

``` python
DATA_DIR = ...
```

Some parts of the codebase also directly reference the following path:

    /work/wildfirerisk/

### Important

To reproduce the experiments you will likely need to:

1.  Modify `DATA_DIR` in `config.py`
2.  Update any hardcoded references to `/work/wildfirerisk/`
3.  Ensure dataset folder names match those expected by the code

Note that some dataset folders may have **different naming
conventions** than the official huggingface data, so you may need to adjust paths accordingly after
downloading the datasets.

------------------------------------------------------------------------

## Citation

If you find this work useful in your research, please cite:

``` bibtex
@article{markov2025firescope,
  title={FireScope: Wildfire Risk Raster Prediction with a Chain-of-Thought Oracle},
  author={Markov, Mario and Ailuro, Stefan Maria and Van Gool, Luc and Schindler, Konrad and Paudel, Danda Pani},
  journal={arXiv preprint arXiv:2511.17171},
  year={2025}
}
```

------------------------------------------------------------------------

## Contact

For questions or issues, please open a GitHub issue in this repository.
