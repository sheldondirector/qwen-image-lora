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
        One-time model setup. You can override the base model per-request
        via the 'base_model' input or with the BASE_MODEL_ID env var.
        """
        self.device = _select_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Sensible default; override via env or input.
        self.default_base_model = os.environ.get(
            "BASE_MODEL_ID", "stabilityai/stable-diffusion-2-1"
        )

        self.pipe = None
        self.pipe_base_id = None
        self.safety_enabled = None

    def _ensure_pipeline(self, base_model: str, scheduler_name: str, use_safety_checker: bool):
        """
        (Re)load the pipeline if base model or safety setting changed,
        and set the requested scheduler.
        """
        need_reload = (
            self.pipe is None
            or self.pipe_base_id != base_model
            or self.safety_enabled != use_safety_checker
        )
        if need_reload:
            kwargs = {"torch_dtype": self.dtype}
            # Only disable the checker when asked; otherwise let the pipeline default handle it.
            if not use_safety_checker:
                kwargs["safety_checker"] = None

            self.pipe = DiffusionPipeline.from_pretrained(base_model, **kwargs)
            self.pipe = self.pipe.to(self.device)
            self.pipe_base_id = base_model
            self.safety_enabled = use_safety_checker

        # Swap scheduler if needed.
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
        help="Classifier-free guidance scale (CFG)",
    )
    @cog.input(
        "width",
        type=int,
        default=768,
        help="Output width (multiple of 8 for most models)",
    )
    @cog.input(
        "height",
        type=int,
        default=768,
        help="Output height (multiple of 8 for most models)",
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
        help="Specific filename inside the LoRA repo, e.g. 'lora.safetensors' (used with lora_repo)",
    )
    @cog.input(
        "lora_scale",
        type=float,
        default=1.0,
        min=0.0,
        max=2.0,
        help="Strength of the LoRA adapter",
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
        # Validate sizes
        if (width % 8) or (height % 8):
            raise ValueError("width and height must be multiples of 8")

        # Seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)

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
            self.pipe.load_lora_weights(str(lora_file))
            if hasattr(self.pipe, "set_adapters"):
                try:
                    self.pipe.set_adapters(["default"], adapter_weights=[lora_scale])
                except Exception:
                    pass
        elif lora_repo is not None:
            kwargs = {}
            if lora_weight_name:
                kwargs["weight_name"] = lora_weight_name
            self.pipe.load_lora_weights(lora_repo, **kwargs)
            if hasattr(self.pipe, "set_adapters"):
                try:
                    self.pipe.set_adapters(["default"], adapter_weights=[lora_scale])
                except Exception:
                    pass

        # Build call kwargs
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator,
        )
        if "guidance_scale" in self.pipe.__call__.__code__.co_varnames:
            call_kwargs["guidance_scale"] = guidance

        # Run inference
        images = self.pipe(**call_kwargs).images
        image: Image.Image = images[0]
        out_path = "output.png"
        image.save(out_path, format="PNG")
        return cog.Path(out_path)
