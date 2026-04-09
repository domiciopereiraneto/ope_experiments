# Inference-Time Optimization of Prompt Embeddings in Diffusion Models: A Comparison of sep-CMA-ES and Adam

This work searches directly in prompt-embedding space for SDXL Turbo while keeping the generation pipeline frozen. The optimization target combines two complementary proxy signals: LAION Aesthetic Predictor V2 for visual quality and CLIP Score for prompt-image alignment.

This repository contains the required experimental code used to compare the gradient-based Adam optimizer with the evolutionary sep-CMA-ES in that setting. The central question is whether gradient-free search is better suited to the noisy, non-convex objective landscape induced by stochastic image generation and external evaluators. In the experiments described in the accompanying paper, sep-CMA-ES consistently achieved higher final objective values than Adam while using fewer computational resources, indicating that evolutionary optimization is a strong approach for inference-time embedding-space exploration in diffusion models.

The repository keeps one compact backend in [`src/utils.py`](/src/utils.py) and two notebooks:

- [`notebook/single_prompt_example.ipynb`](/notebook/single_prompt_example.ipynb) for experimenting with a single prompt.
- [`notebook/sampled_prompts_experiments.ipynb`](/notebook/sampled_prompts_experiments.ipynb) to replicate the experiments found in the paper, where a set of prompts are sampled from the Parti Prompts (P2) dataset.

## Structure

```text
ope_experiments/
├── README.md
├── environment.yml
├── notebook/
│   ├── sampled_prompts_experiments.ipynb
│   └── single_prompt_example.ipynb
├── src/
│   └── utils.py
├── results/
│   └── sample_outputs.png
└── .gitignore
```

## Setup

```bash
conda env create -f environment.yml
conda activate ope-experiments
```

To update an existing Conda environment:

```bash
conda env update -f environment.yml --prune
```

## Notes

Exact replication of the results reported in the paper may not be feasible due to inherent sources of non-determinism in the SDXL Turbo pipeline. In particular, components such as stochastic sampling procedures, parallel GPU operations, and low-level library implementations (e.g., CUDA kernels) can introduce variability across runs, even when using identical seeds and configurations. As a result, small differences in the generated outputs—and consequently in the evaluation metrics—are expected, which can lead to deviations from the originally reported results.