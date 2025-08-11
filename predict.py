# predict.py
import os
import random
from typing import Optional

import torch
from PIL import Image
import cog
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
)

SCHEDULERS = {
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "ddim": DDIMScheduler,
    "pndm": PNDMScheduler,
}

def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # macOS fallback
        return "mps"
    return "cpu"


class Predictor(cog.BasePredictor):
    def setup(self):
        """
        One-time model setup. Adjust the default base model here if you want.
        You can always override it at runtime via the 'base_model' input.
        """
        self.device = _select_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # A sensible default; override via input if you like.
        self.default_base_model = os.environ.get(
            "BASE_MODEL_ID", "stabilityai/stable-diffusion-2-1"
        )

        # Create a lightweight placeholder; we’ll (re)load per request if base_model changes.
        self.pipe = None
        self.pipe_base_id = None

    def _ensure_pipeline(self, base_model: str, scheduler_name: str, use_safety_checker: bool):
        """
        (Re)load the pipeline if the base model or safety checker setting changed,
        and set the requested scheduler.
        """
        if (self.pipe is None) or (self.pipe_base_id != base_model):
            self.pipe = DiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=self.dtype,
                safety_checker=None if not use_safety_checker else None,  # keep None; add checker if you have one
            )
            self.pipe_base_id = base_model

            # Move to device once.
            self.pipe = self.pipe.to(self.device)

        # Swap schedulers if needed
        scheduler_cls = SCHEDULERS.get(scheduler_name, DPMSolverMultistepScheduler)
        if not isinstance(self.pipe.scheduler, scheduler_cls):
            self.pipe.scheduler = scheduler_cls.from_config(self.pipe.scheduler.config)

    @cog.input("prompt", type=str, help="Text prompt for image generation")
    @cog.input(
        "negative_prompt",
        type=str,
        default="",
        help="Negative prompt to steer away from undesired content",
    )
    @cog.input(
        "base_model",
        type=str,
        default=None,
        help="Diffusers base model repo ID (e.g. 'stabilityai/sd-turbo'). Defaults to env BASE_MODEL_ID or SD2.1",
    )
    @cog.input(
        "scheduler",
        type=str,
        default="dpmpp_2m",
        options=list(SCHEDULERS.keys()),
        help="Sampling scheduler",
    )
    @cog.input(
        "steps",
        type=int,
        default=25,
        min=1,
        max=150,
        help="Number of diffusion steps",
    )
    @cog.input(
        "guidance",
        type=float,
        default=7.0,
        min=0.0,
        max=20.0,
        help="Classifier-free guidance scale (a.k.a. CFG)",
    )
    @cog.input(
        "width",
        type=int,
        default=768,
        help="Output width (must be multiple of 8 for most models)",
    )
    @cog.input(
        "height",
        type=int,
        default=768,
        help="Output height (must be multiple of 8 for most models)",
    )
    @cog.input(
        "seed",
        type=int,
        default=None,
        help="Random seed (leave empty for random)",
    )
    @cog.input(
        "use_safety_checker",
        type=bool,
        default=False,
        help="Enable model safety checker if available in the pipeline",
    )
    # ----- LoRA controls -----
    @cog.input(
        "lora_file",
        type=cog.Path,
        default=None,
        help="Local LoRA file (.safetensors/.bin) uploaded with the job",
    )
    @cog.input(
        "lora_repo",
        type=str,
        default=None,
        help="Hugging Face repo ID that contains LoRA weights (alternative to lora_file)",
    )
    @cog.input(
        "lora_weight_name",
        type=str,
        default=None,
        help="Specific filename inside the LoRA repo, e.g. 'lora.safetensors' (only used with lora_repo)",
    )
    @cog.input(
        "lora_scale",
        type=float,
        default=1.0,
        min=0.0,
        max=2.0,
        help="Strength of the LoRA adapter (some pipelines expose this as adapter_scale/weight)",
    )
    def predict(
        self,
        prompt: str,
        negative_prompt: str = "",
        base_model: Optional[str] = None,
        scheduler: str = "dpmpp_2m",
        steps: int = 25,
        guidance: float = 7.0,
        width: int = 768,
        height: int = 768,
        seed: Optional[int] = None,
        use_safety_checker: bool = False,
        lora_file: Optional[cog.Path] = None,
        lora_repo: Optional[str] = None,
        lora_weight_name: Optional[str] = None,
        lora_scale: float = 1.0,
    ) -> cog.Path:
        """
        Generate a single image and return it as a file.
        You can change the LoRA in 3 ways:
          1) Pass a new lora_file (uploaded with the request)
          2) Pass lora_repo (HF repo) + lora_weight_name
          3) Pass nothing to use the base model only
        """
        # Seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        # Ensure/load pipeline
        base_id = base_model or self.default_base_model
        self._ensure_pipeline(base_id, scheduler, use_safety_checker)

        # Clear any previously loaded adapters to avoid stacking across requests
        if hasattr(self.pipe, "unload_lora_weights"):
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass

        # Load LoRA if provided
        if lora_file is not None:
            # Local file uploaded with the request
            self.pipe.load_lora_weights(str(lora_file))
            # Some pipelines support setting scale via set_adapters or similar:
            if hasattr(self.pipe, "set_adapters"):
                try:
                    self.pipe.set_adapters(["default"], adapter_weights=[lora_scale])
                except Exception:
                    pass
        elif lora_repo is not None:
            # From a Hugging Face repo
            kwargs = {}
            if lora_weight_name:
                kwargs["weight_name"] = lora_weight_name
            self.pipe.load_lora_weights(lora_repo, **kwargs)
            if hasattr(self.pipe, "set_adapters"):
                try:
                    self.pipe.set_adapters(["default"], adapter_weights=[lora_scale])
                except Exception:
                    pass
        # else: no LoRA, just the base model

        # Run inference
        extra = {}
        # Many text2img pipelines use these argument names; safe to pass if present.
        if "guidance_scale" in self.pipe.__call__.__code__.co_varnames:
            extra["guidance_scale"] = guidance

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator,
            **extra,
        ).images

        image: Image.Image = images[0]
        out_path = "output.png"
        image.save(out_path, format="PNG")
        return cog.Path(out_path)
