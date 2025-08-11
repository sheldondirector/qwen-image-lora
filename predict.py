from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
import torch, os, tempfile, requests
from typing import Optional

MODEL_ID = "Qwen/Qwen-Image"

# Defaults for your LoRA
DEFAULT_LORA_REPO = "sheldondirector/Kay021000"
DEFAULT_LORA_FILENAME = "pytorch_lora_weights.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        use_cuda = torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.bfloat16 if use_cuda else torch.float32

        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype
        ).to(self.device)

        # If VRAM is tight on your chosen hardware, you can enable:
        # self.pipe.enable_sequential_cpu_offload()
        # self.pipe.enable_attention_slicing()

    def _load_lora_from_hub(self, repo_id: str, filename: str, adapter_name: str, weight: float):
        token = os.getenv("HF_TOKEN")  # set as a Replicate model secret if repo is private
        fp = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        self.pipe.load_lora_weights(fp, adapter_name=adapter_name)
        self.pipe.set_adapters([adapter_name], adapter_weights=[weight])

    def _load_lora_from_url(self, url: str, adapter_name: str, weight: float):
        with tempfile.TemporaryDirectory() as td:
            dst = os.path.join(td, "lora.safetensors")
            headers = {}
            if os.getenv("HF_TOKEN"):
                headers["Authorization"] = f"Bearer {os.getenv('HF_TOKEN')}"
            with requests.get(url, stream=True, headers=headers) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            self.pipe.load_lora_weights(dst, adapter_name=adapter_name)
        self.pipe.set_adapters([adapter_name], adapter_weights=[weight])

    def predict(
        self,
        prompt: str = Input(description="Prompt text"),
        negative_prompt: str = Input(default="", description="Negative prompt"),
        width: int = Input(default=1024, ge=64, le=1792),
        height: int = Input(default=1024, ge=64, le=1792),
        num_inference_steps: int = Input(default=40, ge=1, le=100),
        true_cfg_scale: float = Input(default=4.0, description="Qwen-Image uses true_cfg_scale (not guidance_scale)"),
        seed: int = Input(default=1234, description="Random seed"),

        # LoRA options (defaults to your HF repo)
        use_base_only: bool = Input(default=False, description="Ignore LoRA and use base model only"),
        lora_repo_id: str = Input(default=DEFAULT_LORA_REPO, description="HF repo id for LoRA (if using Hub)"),
        lora_filename: str = Input(default=DEFAULT_LORA_FILENAME, description="Filename in that repo"),
        lora_weight: float = Input(default=0.8, description="LoRA strength, typically 0.6–1.0"),
        adapter_name: str = Input(default="lora"),

        # Alternatives to HF repo:
        lora_url: str = Input(default="", description="Direct URL to .safetensors (e.g. S3 presigned)"),
        lora_file: Optional[Path] = Input(default=None, description="Upload a .safetensors file directly"),
    ) -> Path:
        if not use_base_only:
            if lora_file is not None:
                self.pipe.load_lora_weights(str(lora_file), adapter_name=adapter_name)
                self.pipe.set_adapters([adapter_name], adapter_weights=[lora_weight])
            elif lora_url:
                self._load_lora_from_url(lora_url, adapter_name, lora_weight)
            else:
                self._load_lora_from_hub(lora_repo_id, lora_filename, adapter_name, lora_weight)
        else:
            self.pipe.set_adapters([], adapter_weights=[])

        g = torch.Generator(device=self.device).manual_seed(int(seed))
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            true_cfg_scale=float(true_cfg_scale),
            generator=g,
        ).images[0]

        out = Path("output.png")
        image.save(out)
        return out
