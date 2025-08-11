from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
import torch, os, tempfile, requests
from typing import Union

# Base model
MODEL_ID = "Qwen/Qwen-Image"

# Your default LoRA on the Hub
DEFAULT_LORA_REPO = "sheldondirector/Kay021000"
DEFAULT_LORA_FILENAME = "pytorch_lora_weights.safetensors"


class Predictor(BasePredictor):
    def setup(self):
        """Load the base pipeline once per container."""
        use_cuda = torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.bfloat16 if use_cuda else torch.float32

        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        ).to(self.device)

        # You can uncomment these if VRAM is tight on the runner
        # self.pipe.enable_sequential_cpu_offload()
        # self.pipe.enable_attention_slicing()

    # ---------- LoRA loaders ----------
    def _load_lora_from_hub(self, repo_id: str, filename: str, adapter_name: str, weight: float):
        """Load a LoRA from Hugging Face Hub. Uses HF_TOKEN if present for private repos."""
        token = os.getenv("HF_TOKEN")
        # Prefer repo+weight_name path (robust across diffusers versions)
        try:
            self.pipe.load_lora_weights(
                repo_id,
                weight_name=filename,
                adapter_name=adapter_name,
                token=token,
            )
        except TypeError:
            # Fallback: download the file then load from local path
            local_fp = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
            self.pipe.load_lora_weights(local_fp, adapter_name=adapter_name)

        # Apply weight
        if hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters([adapter_name], adapter_weights=[weight])

    def _load_lora_from_url(self, url: str, adapter_name: str, weight: float):
        """Download a LoRA file from a direct URL (e.g., presigned S3) and load it."""
        with tempfile.TemporaryDirectory() as td:
            dst = os.path.join(td, "lora.safetensors")
            headers = {}
            if os.getenv("HF_TOKEN"):
                headers["Authorization"] = f"Bearer {os.getenv('HF_TOKEN')}"
            with requests.get(url, stream=True, headers=headers or None) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            self.pipe.load_lora_weights(dst, adapter_name=adapter_name)

        if hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters([adapter_name], adapter_weights=[weight])

    def _load_lora_from_file(self, file_path: str, adapter_name: str, weight: float):
        """Load a LoRA from an uploaded file path."""
        self.pipe.load_lora_weights(file_path, adapter_name=adapter_name)
        if hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters([adapter_name], adapter_weights=[weight])

    # ---------- Prediction ----------
    def predict(
        self,
        prompt: str = Input(description="Prompt text"),
        negative_prompt: str = Input(default="", description="Negative prompt"),
        width: int = Input(default=1024, ge=64, le=1792),
        height: int = Input(default=1024, ge=64, le=1792),
        num_inference_steps: int = Input(default=40, ge=1, le=100),
        true_cfg_scale: float = Input(default=4.0, description="Qwen-Image 'true CFG' (fallbacks to guidance_scale if unsupported)"),
        seed: int = Input(default=1234, description="Random seed"),

        # LoRA controls
        use_base_only: bool = Input(default=False, description="Ignore LoRA; use base model only"),
        lora_repo_id: str = Input(default=DEFAULT_LORA_REPO, description="HF repo id for LoRA (if not uploading)"),
        lora_filename: str = Input(default=DEFAULT_LORA_FILENAME, description="Filename in the HF repo"),
        lora_weight: float = Input(default=0.8, description="LoRA strength (typical 0.6–1.0)"),
        adapter_name: str = Input(default="lora", description="Adapter name"),

        # Alternative LoRA sources
        lora_url: str = Input(default="", description="Direct URL to a .safetensors file"),
        lora_file: Union[Path, None] = Input(default=None, description="Upload a .safetensors file"),
    ) -> Path:
        """
        Returns a single PNG image saved as output.png
        """
        # Load/clear LoRA
        if use_base_only:
            # Clear any adapters if the pipeline provides that API
            try:
                if hasattr(self.pipe, "set_adapters"):
                    self.pipe.set_adapters([], adapter_weights=[])
            except Exception:
                pass
        else:
            if lora_file is not None:
                self._load_lora_from_file(str(lora_file), adapter_name, lora_weight)
            elif lora_url:
                self._load_lora_from_url(lora_url, adapter_name, lora_weight)
            else:
                self._load_lora_from_hub(lora_repo_id, lora_filename, adapter_name, lora_weight)

        # Deterministic seed
        g = torch.Generator(device=self.device).manual_seed(int(seed))

        # Some diffusers pipelines accept `true_cfg_scale`; others use `guidance_scale`.
        # Try true_cfg first; gracefully fall back if not supported.
        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(num_inference_steps),
                true_cfg_scale=float(true_cfg_scale),
                generator=g,
            )
        except TypeError:
            # Fallback to guidance_scale if true_cfg_scale isn't recognized
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(true_cfg_scale),
                generator=g,
            )

        image = result.images[0]
        out = Path("output.png")
        image.save(out)
        return out
