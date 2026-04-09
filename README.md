# OPE Experiments

Reduced repository for comparing sep-CMA-ES and Adam on prompt-embedding optimization with:

- `stabilityai/sdxl-turbo`
- LAION Aesthetic Score V2 (`predictor=2`)
- CLIP Score

The repository keeps one compact backend in [`src/utils.py`](/home/posgrad/phd2025/dneto/ope_experiments/src/utils.py) and two notebooks:

- [`notebook/single_prompt.ipynb`](/home/posgrad/phd2025/dneto/ope_experiments/notebook/single_prompt.ipynb)
- [`notebook/sampled_prompts.ipynb`](/home/posgrad/phd2025/dneto/ope_experiments/notebook/sampled_prompts.ipynb)

## Structure

```text
ope_experiments/
├── README.md
├── requirements.txt
├── notebook/
│   ├── sampled_prompts.ipynb
│   └── single_prompt.ipynb
├── src/
│   └── utils.py
├── results/
│   └── sample_outputs.png
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebook Coverage

- `single_prompt.ipynb`: single-prompt optimization with either sep-CMA-ES or Adam.
- `sampled_prompts.ipynb`: Parti Prompts sampling in the style of `p2_experiments.py`, batch runs, and aggregated summaries.

## Notes

- The reduced code path only supports `predictor=2`.
- The reduced CMA-ES path only supports the `sep` variant.
- SDXL Turbo, CLIP, and the LAION V2 aesthetic weights are downloaded automatically on first use.
