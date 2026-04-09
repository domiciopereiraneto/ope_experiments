from __future__ import annotations

import contextlib
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import cma
import clip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline
from PIL import Image


DEFAULT_CONFIG = {
    "selected_prompt": "an astronaut riding a horse on Mars",
    "optimization_method": "cmaes",
    "seed": 42,
    "cuda": 0,
    "predictor": 2,
    "num_inference_steps": 1,
    "guidance_scale": 0.0,
    "height": 512,
    "width": 512,
    "aesthetic_score_weight": 0.5,
    "clip_score_weight": 0.5,
    "max_aesthetic_score": 10.0,
    "max_clip_score": 0.5,
    "model_id": "stabilityai/sdxl-turbo",
    "torch_dtype": "float32",
    "use_safetensors": True,
    "results_folder": "results",
    "prompt_per_category": 3,
    "prompt_sample_seed": 42,
    "single_prompt_per_seed": True,
    "num_generations": 100,
    "pop_size": 20,
    "cmaes_variant": "sep",
    "sigma": 0.4,
    "save_gens": False,
    "num_iterations": 250,
    "adam_lr": 1e-4,
    "adam_weight_decay": 0.0,
    "adam_eps": 1e-8,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_max_grad_norm": None,
    "time_limit_seconds": None,
}


def make_config(**overrides):
    config = dict(DEFAULT_CONFIG)
    config.update(overrides)
    return config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(cuda_value):
    if cuda_value == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{cuda_value}"


def _model_id_tag(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model_id.lower().split("/")[-1])


def _to_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(dtype_name).lower()
    if key not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return dtype_map[key]


def _tensor_to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image_np = image_tensor.detach().to(torch.float32).cpu().numpy()
    image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
    image_np = np.clip(image_np, 0.0, 1.0)
    return (image_np * 255).astype(np.uint8)


def _save_image(image_tensor: torch.Tensor, path: str | Path) -> None:
    Image.fromarray(_tensor_to_uint8_image(image_tensor)).save(path)


def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _first_non_empty(series: pd.Series):
    for value in series.dropna():
        text = str(value).strip()
        if text:
            return text
    return None


def _l2_normalize(arr: torch.Tensor, axis: int = -1) -> torch.Tensor:
    norm = arr.norm(p=2, dim=axis, keepdim=True).clamp(min=1e-8)
    return arr / norm


def _download_laion_v2_weights(target: Path, variant: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    base_url = "https://huggingface.co/camenduru/improved-aesthetic-predictor/resolve/main/"
    url = base_url + variant
    print(f"Downloading LAION V2 weights to {target}")
    urlretrieve(url, target)


class _ImprovedMLP(nn.Module):
    def __init__(self, embedding_dim: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LAIONV2Aesthetic:
    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        clip_model: str = "ViT-L/14",
        weight_variant: str = "sac+logos+ava1-l14-linearMSE.pth",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.clip_model_name = clip_model
        self.clip_model, self._preprocess_pil = clip.load(self.clip_model_name, device=self.device)
        self.clip_model.eval().requires_grad_(False)

        cache_dir = Path(cache_dir) if cache_dir is not None else Path(os.path.expanduser("~/.cache/emb_reader"))
        weights_path = cache_dir / weight_variant
        if not weights_path.exists():
            _download_laion_v2_weights(weights_path, weight_variant)

        self.mlp = _ImprovedMLP(embedding_dim=768).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.mlp.load_state_dict(state_dict, strict=True)
        self.mlp.eval().requires_grad_(False)

        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(-1, 1, 1)
        self._std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(-1, 1, 1)

    @staticmethod
    def _resize_shorter_side(img: torch.Tensor, target: int = 224) -> torch.Tensor:
        _, h, w = img.shape
        scale = target / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        img = img.unsqueeze(0)
        img = F.interpolate(img, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
        return img.squeeze(0)

    def _preprocess_tensor(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(self.device)
        img = self._resize_shorter_side(img)
        _, h, w = img.shape
        top, left = (h - 224) // 2, (w - 224) // 2
        img = img[:, top:top + 224, left:left + 224]
        return (img - self._mean) / self._std

    def _embed(self, img: torch.Tensor, *, no_grad: bool = True) -> torch.Tensor:
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            emb = self.clip_model.encode_image(img).float()
        return _l2_normalize(emb, axis=-1)

    def predict_from_pil(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self._preprocess_pil(image).unsqueeze(0).to(self.device)
        emb = self._embed(img_tensor, no_grad=True)
        with torch.no_grad():
            score = self.mlp(emb)
        return score.squeeze()

    def predict_from_tensor(self, img_tensor: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        img_tensor = self._preprocess_tensor(img_tensor).unsqueeze(0)
        img_tensor.requires_grad_(True)
        emb = self._embed(img_tensor, no_grad=False)
        score = self.mlp(emb)
        return score.to(dtype).squeeze()


@dataclass
class PromptExample:
    prompt: str
    category: str


def sample_parti_prompts(
    prompt_per_category: int = 3,
    sample_seed: int = 42,
    *,
    single_prompt_per_seed: bool = False,
    run_seed: int | None = None,
) -> list[PromptExample]:
    dataset = load_dataset("nateraw/parti-prompts")["train"]
    category_prompts = defaultdict(list)
    all_prompts = []
    for item in dataset:
        category = item.get("Category", "Uncategorized")
        prompt = item["Prompt"]
        category_prompts[category].append(prompt)
        all_prompts.append(PromptExample(prompt=prompt, category=category))

    if single_prompt_per_seed:
        seed = sample_seed if run_seed is None else run_seed
        rng = random.Random(seed)
        return [rng.choice(all_prompts)]

    rng = random.Random(sample_seed)
    selected = []
    for category, prompts in category_prompts.items():
        if len(prompts) >= prompt_per_category:
            chosen = rng.sample(prompts, prompt_per_category)
        else:
            chosen = list(prompts)
        selected.extend(PromptExample(prompt=p, category=category) for p in chosen)
    return selected


class EmbeddingOptimizer:
    _MODEL_CACHE = {}
    _ACTIVE_CACHE_KEY = None

    def __init__(self, config: dict):
        self.config = dict(config)
        self.device = get_device(self.config.get("cuda", 0))
        self.model_dtype = _to_dtype(self.config.get("torch_dtype", "float32"))
        self.guidance_scale = float(self.config.get("guidance_scale", 0.0))
        self._clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self._clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.results_root = Path(self.config.get("results_folder", "results"))
        self.results_root.mkdir(parents=True, exist_ok=True)

        if int(self.config.get("predictor", 2)) != 2:
            raise ValueError("This reduced repository only supports predictor=2 (LAION Aesthetic Score V2).")
        if self.config.get("optimization_method") == "cmaes" and self.config.get("cmaes_variant", "sep") != "sep":
            raise ValueError("This reduced repository only supports sep-CMA-ES for CMA-ES runs.")

        cache_key = (
            self.config["model_id"],
            str(self.model_dtype),
            self.device,
            bool(self.config.get("use_safetensors", True)),
        )
        cls = type(self)
        if cls._ACTIVE_CACHE_KEY is not None and cls._ACTIVE_CACHE_KEY != cache_key:
            cls.clear_model_cache()
        if cache_key not in cls._MODEL_CACHE:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.config["model_id"],
                torch_dtype=self.model_dtype,
                use_safetensors=bool(self.config.get("use_safetensors", True)),
            ).to(self.device)
            pipe.set_progress_bar_config(disable=True)
            for component in pipe.components.values():
                params = getattr(component, "parameters", None)
                if callable(params):
                    for parameter in params():
                        parameter.requires_grad_(False)

            clip_model, clip_preprocess = clip.load("ViT-L/14", device=self.device)
            clip_model.eval().requires_grad_(False)
            aesthetic_model = LAIONV2Aesthetic(self.device, clip_model="ViT-L/14")
            cls._MODEL_CACHE[cache_key] = {
                "pipe": pipe,
                "clip_model": clip_model,
                "clip_preprocess": clip_preprocess,
                "aesthetic_model": aesthetic_model,
            }
            cls._ACTIVE_CACHE_KEY = cache_key

        cached = cls._MODEL_CACHE[cache_key]
        self.pipe = cached["pipe"]
        self.clip_model = cached["clip_model"]
        self.clip_preprocess = cached["clip_preprocess"]
        self.aesthetic_model = cached["aesthetic_model"]
        wrapped_call = getattr(self.pipe.__class__.__call__, "__wrapped__", None)
        self.call_with_grad = wrapped_call.__get__(self.pipe, self.pipe.__class__) if wrapped_call is not None else self.pipe.__call__

    @classmethod
    def clear_model_cache(cls):
        cls._MODEL_CACHE.clear()
        cls._ACTIVE_CACHE_KEY = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _pipeline_input_device(self):
        execution_device = getattr(self.pipe, "_execution_device", None)
        if execution_device is not None:
            return str(execution_device)
        return self.device

    def _adam_autocast_context(self):
        if self.model_dtype not in (torch.float16, torch.bfloat16):
            return contextlib.nullcontext()
        if self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=self.model_dtype)
        if self.device.startswith("cpu") and self.model_dtype == torch.bfloat16:
            return torch.autocast(device_type="cpu", dtype=self.model_dtype)
        return contextlib.nullcontext()

    def _encode_prompt_embeddings(self, prompt: str):
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            device=self._pipeline_input_device(),
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.guidance_scale > 1.0,
        )
        return prompt_embeds, pooled_prompt_embeds

    def _build_generation_kwargs(self, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor, generator):
        prompt_device = self._pipeline_input_device()
        return {
            "prompt_embeds": prompt_embeds.to(device=prompt_device, dtype=self.model_dtype),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(device=prompt_device, dtype=self.model_dtype),
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": int(self.config["num_inference_steps"]),
            "generator": generator,
            "height": int(self.config["height"]),
            "width": int(self.config["width"]),
            "output_type": "pt",
        }

    def generate_image_from_embeddings_cmaes(self, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor, seed: int):
        generator = torch.Generator(device=self._pipeline_input_device()).manual_seed(seed)
        out = self.pipe(**self._build_generation_kwargs(prompt_embeds, pooled_prompt_embeds, generator))["images"]
        image = out.clamp(0, 1).squeeze(0).permute(1, 2, 0)
        return image.to(self.device)

    def generate_image_from_embeddings_adam(self, text_embeddings: list[torch.Tensor], seed: int):
        generator = torch.Generator(device=self._pipeline_input_device()).manual_seed(seed)
        prompt_embeds, pooled_prompt_embeds = text_embeddings
        out = self.call_with_grad(**self._build_generation_kwargs(prompt_embeds, pooled_prompt_embeds, generator))["images"]
        image = out.clamp(0, 1).squeeze(0).permute(1, 2, 0)
        return image.to(self.device)

    def evaluate_aesthetic(self, image_tensor: torch.Tensor):
        return self.aesthetic_model.predict_from_tensor(image_tensor.permute(2, 0, 1).to(torch.float32))

    def evaluate_clip_score_cmaes(self, image_tensor: torch.Tensor, prompt: str):
        image = Image.fromarray(_tensor_to_uint8_image(image_tensor))
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input).float()
            text_features = self.clip_model.encode_text(text_input).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).squeeze()

    def evaluate_clip_score_adam(self, image_tensor: torch.Tensor, text_features: torch.Tensor):
        img = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        img = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=False)
        mean = self._clip_mean.to(img.device, img.dtype)
        std = self._clip_std.to(img.device, img.dtype)
        img = (img - mean) / std
        image_features = self.clip_model.encode_image(img).float()
        image_features = F.normalize(image_features, dim=-1, eps=1e-6)
        return (image_features @ text_features.T).squeeze()

    def _combine_metric_components(self, aesthetic_score, clip_score):
        aesthetic_component = float(self.config["aesthetic_score_weight"]) * aesthetic_score / float(self.config["max_aesthetic_score"])
        clip_component = float(self.config["clip_score_weight"]) * clip_score / float(self.config["max_clip_score"])
        total = aesthetic_component + clip_component
        return total, {
            "aesthetic_component": aesthetic_component,
            "clip_component": clip_component,
        }

    def _save_run_config(self, results_folder: Path, seed: int, prompt: str, category: str | None = None):
        run_config = dict(self.config)
        run_config["seed"] = int(seed)
        run_config["selected_prompt"] = prompt
        if category is not None:
            run_config["category"] = category
        with open(results_folder / "config.json", "w", encoding="utf-8") as handle:
            json.dump(run_config, handle, indent=2, ensure_ascii=False)

    def _make_run_folder(self, method: str, seed: int, prompt_number: int | None = None):
        model_tag = _model_id_tag(self.config["model_id"])
        base_name = f"{method}_laionv2_sdxl_{model_tag}_seed{seed}"
        if prompt_number is not None:
            base_name += f"_p{prompt_number}"
        results_folder = self.results_root / base_name
        results_folder.mkdir(parents=True, exist_ok=True)
        return results_folder

    def evaluate_candidate(self, flat_embedding: np.ndarray, seed: int, embedding_shape, prompt: str, save_path: str | Path | None = None):
        split = int(np.prod(embedding_shape[0]))
        prompt_embeds = torch.tensor(flat_embedding[:split], dtype=torch.float32, device=self.device).view(embedding_shape[0])
        pooled_prompt_embeds = torch.tensor(flat_embedding[split:], dtype=torch.float32, device=self.device).view(embedding_shape[1])
        with torch.no_grad():
            image = self.generate_image_from_embeddings_cmaes(prompt_embeds, pooled_prompt_embeds, seed)
            aesthetic_score = float(self.evaluate_aesthetic(image).item())
            clip_score = float(self.evaluate_clip_score_cmaes(image, prompt).item())
        fitness, components = self._combine_metric_components(aesthetic_score, clip_score)
        if save_path is not None:
            _save_image(image, save_path)
        return -fitness, aesthetic_score, clip_score, fitness, components

    def _plot_cmaes_results(self, results: pd.DataFrame, results_folder: Path) -> None:
        def plot_mean_std(x, mean_values, std_values, label):
            lower = [m - s for m, s in zip(mean_values, std_values)]
            upper = [m + s for m, s in zip(mean_values, std_values)]
            plt.plot(x, mean_values, "--", label=f"{label} Avg.")
            plt.fill_between(x, lower, upper, alpha=0.3, label=f"{label} Avg. ± SD")

        plt.figure(figsize=(10, 6))
        plot_mean_std(results["generation"], results["avg_fitness"], results["std_fitness"], "Fitness")
        plt.plot(results["generation"], results["max_fitness"], "r-", label="Best Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder / "fitness_evolution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results["generation"], results["avg_aesthetic_score"], results["std_aesthetic_score"], "Population")
        plt.plot(results["generation"], results["max_aesthetic_score"], "r-", label="Best")
        plt.xlabel("Generation")
        plt.ylabel("Aesthetic Score")
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder / "aesthetic_score_evolution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results["generation"], results["avg_clip_score"], results["std_clip_score"], "Population")
        plt.plot(results["generation"], results["max_clip_score"], "r-", label="Best")
        plt.xlabel("Generation")
        plt.ylabel("CLIP Score")
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder / "clip_score_evolution.png")
        plt.close()

    def _plot_adam_results(self, results: pd.DataFrame, results_folder: Path) -> None:
        for filename, column, ylabel, title in [
            ("aesthetic_evolution.png", "aesthetic_score", "Aesthetic Score", "Aesthetic Score Evolution"),
            ("clip_evolution.png", "clip_score", "CLIP Score", "CLIP Score Evolution"),
            ("loss_evolution.png", "combined_loss", "Loss", "Loss Evolution"),
        ]:
            plt.figure(figsize=(10, 6))
            plt.plot(results["iteration"], results[column], label=ylabel)
            plt.xlabel("Iteration")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(results_folder / filename)
            plt.close()

    def run_cmaes(self, prompt: str, seed: int | None = None, *, category: str | None = None, prompt_number: int | None = None):
        seed = int(self.config["seed"] if seed is None else seed)
        set_seed(seed)
        results_folder = self._make_run_folder("sepcmaes", seed, prompt_number=prompt_number)
        self._save_run_config(results_folder, seed, prompt, category)

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt_embeddings(prompt)
            initial_image = self.generate_image_from_embeddings_cmaes(prompt_embeds.clone(), pooled_prompt_embeds.clone(), seed)
            _save_image(initial_image, results_folder / "it_0.png")

        trainable_params_init = torch.cat([prompt_embeds.flatten(), pooled_prompt_embeds.flatten()]).to(torch.float32).cpu().numpy()
        text_embeddings_init_shape = [prompt_embeds.shape, pooled_prompt_embeds.shape]

        initial_loss, initial_aesthetic, initial_clip, initial_fitness, _ = self.evaluate_candidate(
            trainable_params_init,
            seed,
            text_embeddings_init_shape,
            prompt,
        )

        es_options = {
            "seed": seed,
            "popsize": int(self.config["pop_size"]),
            "maxiter": int(self.config["num_generations"]),
            "verb_filenameprefix": str(results_folder / "outcmaes"),
            "verb_log": 0,
            "verbose": -9,
            "CMA_diagonal": True,
        }
        es = cma.CMAEvolutionStrategy(trainable_params_init, float(self.config["sigma"]), es_options)

        start_time = time.time()
        time_list = [0.0]
        best_fitness_overall = float(initial_fitness)
        best_text_embeddings_overall = trainable_params_init.copy()
        max_fit_list = [float(initial_fitness)]
        avg_fit_list = [float(initial_fitness)]
        std_fit_list = [0.0]
        max_aesthetic_list = [float(initial_aesthetic)]
        avg_aesthetic_list = [float(initial_aesthetic)]
        std_aesthetic_list = [0.0]
        max_clip_list = [float(initial_clip)]
        avg_clip_list = [float(initial_clip)]
        std_clip_list = [0.0]

        generation = 0
        while not es.stop():
            elapsed_time = time.time() - start_time
            if self.config.get("time_limit_seconds") is not None and elapsed_time >= float(self.config["time_limit_seconds"]):
                break

            generation += 1
            print(f"Generation {generation}/{self.config['num_generations']}")
            if self.config.get("save_gens", False):
                (results_folder / f"gen_{generation}").mkdir(exist_ok=True)

            solutions = es.ask()
            tmp_fitnesses = []
            aesthetic_scores = []
            clip_scores = []
            for ind_id, x in enumerate(solutions, start=1):
                save_path = None
                if self.config.get("save_gens", False):
                    save_path = results_folder / f"gen_{generation}" / f"id_{ind_id}.png"
                loss, aesthetic_score, clip_score, _, _ = self.evaluate_candidate(
                    x,
                    seed,
                    text_embeddings_init_shape,
                    prompt,
                    save_path=save_path,
                )
                tmp_fitnesses.append(loss)
                aesthetic_scores.append(aesthetic_score)
                clip_scores.append(clip_score)
            es.tell(solutions, tmp_fitnesses)

            fitnesses = np.array([-f for f in tmp_fitnesses], dtype=float)
            max_fit = float(np.max(fitnesses))
            avg_fit = float(np.mean(fitnesses))
            std_fit = float(np.std(fitnesses))
            max_aesthetic = float(np.max(aesthetic_scores))
            avg_aesthetic = float(np.mean(aesthetic_scores))
            std_aesthetic = float(np.std(aesthetic_scores))
            max_clip = float(np.max(clip_scores))
            avg_clip = float(np.mean(clip_scores))
            std_clip = float(np.std(clip_scores))

            best_idx = int(np.argmax(fitnesses))
            best_x = solutions[best_idx]
            if max_fit > best_fitness_overall:
                best_fitness_overall = max_fit
                best_text_embeddings_overall = np.array(best_x, copy=True)

            with torch.no_grad():
                split = int(np.prod(text_embeddings_init_shape[0]))
                best_prompt_embeds = torch.tensor(best_x[:split], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[0])
                best_pooled_embeds = torch.tensor(best_x[split:], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[1])
                best_image = self.generate_image_from_embeddings_cmaes(best_prompt_embeds, best_pooled_embeds, seed)
                _save_image(best_image, results_folder / f"best_{generation}.png")

            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)
            max_fit_list.append(max_fit)
            avg_fit_list.append(avg_fit)
            std_fit_list.append(std_fit)
            max_aesthetic_list.append(max_aesthetic)
            avg_aesthetic_list.append(avg_aesthetic)
            std_aesthetic_list.append(std_aesthetic)
            max_clip_list.append(max_clip)
            avg_clip_list.append(avg_clip)
            std_clip_list.append(std_clip)

            results = pd.DataFrame(
                {
                    "generation": list(range(0, generation + 1)),
                    "prompt": [prompt] + [""] * generation,
                    "avg_fitness": avg_fit_list,
                    "std_fitness": std_fit_list,
                    "max_fitness": max_fit_list,
                    "avg_aesthetic_score": avg_aesthetic_list,
                    "std_aesthetic_score": std_aesthetic_list,
                    "max_aesthetic_score": max_aesthetic_list,
                    "avg_clip_score": avg_clip_list,
                    "std_clip_score": std_clip_list,
                    "max_clip_score": max_clip_list,
                    "elapsed_time": time_list,
                }
            )
            if category is not None:
                results["category"] = [category] + [""] * generation
            results.to_csv(results_folder / "fitness_results.csv", index=False)
            self._plot_cmaes_results(results, results_folder)

            generations_left = int(self.config["num_generations"]) - generation
            eta = (elapsed_time / generation) * generations_left if generation else 0.0
            print(
                f"Generation {generation}/{self.config['num_generations']}: "
                f"Max fitness={max_fit:.4f}, Max aesthetic={max_aesthetic:.4f}, "
                f"Max clip={max_clip:.4f}, ETA={_format_time(eta)}"
            )

        with torch.no_grad():
            split = int(np.prod(text_embeddings_init_shape[0]))
            best_prompt_embeds = torch.tensor(best_text_embeddings_overall[:split], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[0])
            best_pooled_embeds = torch.tensor(best_text_embeddings_overall[split:], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[1])
            best_image = self.generate_image_from_embeddings_cmaes(best_prompt_embeds, best_pooled_embeds, seed)
            _save_image(best_image, results_folder / "best_all.png")

        return str(results_folder)

    def run_adam(self, prompt: str, seed: int | None = None, *, category: str | None = None, prompt_number: int | None = None):
        seed = int(self.config["seed"] if seed is None else seed)
        set_seed(seed)
        results_folder = self._make_run_folder("adam", seed, prompt_number=prompt_number)
        self._save_run_config(results_folder, seed, prompt, category)

        with torch.no_grad():
            text_tokens = clip.tokenize([prompt]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens).float()
            text_features = F.normalize(text_features, dim=-1, eps=1e-6)

        prompt_embeds, pooled_prompt_embeds = self._encode_prompt_embeddings(prompt)
        text_embeddings_init = [
            prompt_embeds.detach().clone().to(torch.float32),
            pooled_prompt_embeds.detach().clone().to(torch.float32),
        ]
        text_embeddings = [
            torch.nn.Parameter(text_embeddings_init[0].clone()),
            torch.nn.Parameter(text_embeddings_init[1].clone()),
        ]

        with torch.no_grad():
            with self._adam_autocast_context():
                initial_image = self.generate_image_from_embeddings_adam(text_embeddings_init, seed)
            _save_image(initial_image, results_folder / "it_0.png")

        aesthetic_score = self.evaluate_aesthetic(initial_image)
        clip_score = self.evaluate_clip_score_adam(initial_image, text_features)
        initial_combined_score, _ = self._combine_metric_components(aesthetic_score, clip_score)
        initial_combined_loss = 1 - initial_combined_score
        if not torch.isfinite(initial_combined_loss):
            raise RuntimeError("Initial Adam objective is non-finite.")

        optimizer = torch.optim.AdamW(
            text_embeddings,
            lr=float(self.config["adam_lr"]),
            betas=(float(self.config["adam_beta1"]), float(self.config["adam_beta2"])),
            weight_decay=float(self.config["adam_weight_decay"]),
            eps=float(self.config["adam_eps"]),
        )

        aesthetic_score_list = [float(aesthetic_score.item())]
        clip_score_list = [float(clip_score.item())]
        combined_score_list = [float(initial_combined_score.item())]
        combined_loss_list = [float(initial_combined_loss.item())]
        time_list = [0.0]
        best_score = float(initial_combined_score.item())
        best_text_embeddings = [t.detach().clone() for t in text_embeddings_init]

        start_time = time.time()
        elapsed_time = 0.0
        max_grad_norm = self.config.get("adam_max_grad_norm")
        if max_grad_norm is not None and float(max_grad_norm) <= 0:
            max_grad_norm = None

        for iteration in range(1, int(self.config["num_iterations"]) + 1):
            if self.config.get("time_limit_seconds") is not None and elapsed_time >= float(self.config["time_limit_seconds"]):
                break
            print(f"Iteration {iteration}/{self.config['num_iterations']}")
            optimizer.zero_grad()

            with self._adam_autocast_context():
                image = self.generate_image_from_embeddings_adam(text_embeddings, seed)
            aesthetic_score = self.evaluate_aesthetic(image)
            clip_score = self.evaluate_clip_score_adam(image, text_features)
            combined_score, _ = self._combine_metric_components(aesthetic_score, clip_score)
            combined_loss = 1 - combined_score
            if not torch.isfinite(combined_loss):
                print("Non-finite Adam objective detected. Stopping early.")
                break

            combined_loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(text_embeddings, max_norm=float(max_grad_norm))
            optimizer.step()

            current_score = float(combined_score.item())
            if current_score > best_score:
                best_score = current_score
                best_text_embeddings = [t.detach().clone() for t in text_embeddings]

            aesthetic_score_list.append(float(aesthetic_score.item()))
            clip_score_list.append(float(clip_score.item()))
            combined_score_list.append(current_score)
            combined_loss_list.append(float(combined_loss.item()))
            _save_image(image, results_folder / f"it_{iteration}.png")

            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)
            results = pd.DataFrame(
                {
                    "iteration": list(range(0, iteration + 1)),
                    "prompt": [prompt] + [""] * iteration,
                    "combined_score": combined_score_list,
                    "combined_loss": combined_loss_list,
                    "aesthetic_score": aesthetic_score_list,
                    "clip_score": clip_score_list,
                    "elapsed_time": time_list,
                }
            )
            if category is not None:
                results["category"] = [category] + [""] * iteration
            results.to_csv(results_folder / "score_results.csv", index=False)
            self._plot_adam_results(results, results_folder)

            iterations_left = int(self.config["num_iterations"]) - iteration
            eta = (elapsed_time / iteration) * iterations_left if iteration else 0.0
            print(
                f"Iteration {iteration}/{self.config['num_iterations']}: "
                f"Combined score={current_score:.4f}, Aesthetic={aesthetic_score.item():.4f}, "
                f"CLIP={clip_score.item():.4f}, ETA={_format_time(eta)}"
            )

        with torch.no_grad():
            with self._adam_autocast_context():
                best_image = self.generate_image_from_embeddings_adam(best_text_embeddings, seed)
            _save_image(best_image, results_folder / "best_all.png")

        return str(results_folder)


def run_single_prompt_experiment(config: dict):
    config = dict(config)
    optimizer = EmbeddingOptimizer(config)
    prompt = config["selected_prompt"]
    method = config.get("optimization_method", "cmaes")
    if method == "cmaes":
        return optimizer.run_cmaes(prompt, seed=int(config["seed"]))
    if method == "adam":
        return optimizer.run_adam(prompt, seed=int(config["seed"]))
    raise ValueError("optimization_method must be either 'cmaes' or 'adam'")


def run_batch_experiment(
    base_config: dict,
    *,
    methods: Iterable[str] = ("cmaes", "adam"),
    seeds: Iterable[int] | None = None,
):
    base_config = dict(base_config)
    seeds = list(seeds or [int(base_config["seed"])])
    all_run_folders = []
    for method in methods:
        method_config = dict(base_config)
        method_config["optimization_method"] = method
        optimizer = EmbeddingOptimizer(method_config)
        for seed in seeds:
            prompt_examples = sample_parti_prompts(
                prompt_per_category=int(method_config["prompt_per_category"]),
                sample_seed=int(method_config["prompt_sample_seed"]),
                single_prompt_per_seed=bool(method_config.get("single_prompt_per_seed", False)),
                run_seed=int(seed),
            )
            for prompt_number, example in enumerate(prompt_examples, start=1):
                print(f"Running {method} | seed={seed} | prompt={example.prompt}")
                if method == "cmaes":
                    folder = optimizer.run_cmaes(
                        example.prompt,
                        seed=int(seed),
                        category=example.category,
                        prompt_number=prompt_number,
                    )
                elif method == "adam":
                    folder = optimizer.run_adam(
                        example.prompt,
                        seed=int(seed),
                        category=example.category,
                        prompt_number=prompt_number,
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")
                all_run_folders.append(folder)
    return all_run_folders


def _find_result_dirs(output_folder: str | Path):
    output_folder = Path(output_folder)
    result_dirs = []
    for candidate in sorted(output_folder.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "fitness_results.csv").exists() or (candidate / "score_results.csv").exists():
            result_dirs.append(candidate)
    return result_dirs


def aggregate_results(output_folder: str | Path):
    output_folder = Path(output_folder)
    rows = []
    for run_dir in _find_result_dirs(output_folder):
        if (run_dir / "fitness_results.csv").exists():
            df = pd.read_csv(run_dir / "fitness_results.csv")
            df = df.sort_values("generation")
            rows.append(
                {
                    "run_dir": run_dir.name,
                    "method": "sepcmaes",
                    "prompt": _first_non_empty(df["prompt"]) if "prompt" in df.columns else run_dir.name,
                    "category": _first_non_empty(df["category"]) if "category" in df.columns else "",
                    "final_aesthetic_score": float(df["max_aesthetic_score"].iloc[-1]),
                    "final_clip_score": float(df["max_clip_score"].iloc[-1]),
                    "final_objective": float(df["max_fitness"].iloc[-1]),
                    "elapsed_time": float(df["elapsed_time"].iloc[-1]),
                }
            )
        elif (run_dir / "score_results.csv").exists():
            df = pd.read_csv(run_dir / "score_results.csv")
            df = df.sort_values("iteration")
            rows.append(
                {
                    "run_dir": run_dir.name,
                    "method": "adam",
                    "prompt": _first_non_empty(df["prompt"]) if "prompt" in df.columns else run_dir.name,
                    "category": _first_non_empty(df["category"]) if "category" in df.columns else "",
                    "final_aesthetic_score": float(df["aesthetic_score"].iloc[-1]),
                    "final_clip_score": float(df["clip_score"].iloc[-1]),
                    "final_objective": float((1.0 - df["combined_loss"].iloc[-1])),
                    "elapsed_time": float(df["elapsed_time"].iloc[-1]),
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary.to_csv(output_folder / "aggregate_summary.csv", index=False)

    method_summary = (
        summary.groupby("method", dropna=False)[["final_aesthetic_score", "final_clip_score", "final_objective", "elapsed_time"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    method_summary.to_csv(output_folder / "aggregate_method_summary.csv")
    return summary
