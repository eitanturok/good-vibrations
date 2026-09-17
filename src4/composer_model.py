"""ComposerModel wrapping SonicDiffusion (CLAP audio encoder -> Adapter -> gated
cross-attention UNet, see https://arxiv.org/abs/2405.00878) for training on our
vibration data.

Trainable parameters: the audio Adapter (model/audio_projector.py) + the UNet's
gated adapter layers (model/unet2d.py's `use_adapter_list`-selected blocks) --
everything else (VAE, CLIP text encoder, CLAP audio encoder, the rest of the UNet)
stays frozen. Both trainable pieces are initialized from SonicDiffusion's released
'landscape' checkpoint (continued fine-tuning, not training from scratch) -- see
model/clap_audio_encoder.py and ckpts/.

Unlike src3 (AudioToken), audio conditioning here is NOT spliced into the text
token sequence -- it gets its own parallel gated cross-attention pathway in the
UNet (10 of its attention blocks), with the frozen CLIP text encoder still handling
the (usually empty, see dataset.py:DEFAULT_PROMPT) text prompt separately.
"""
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from composer import ComposerModel
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from src4.model.audio_projector import Adapter  # noqa: E402
from src4.model.clap_audio_encoder import load_clap_audio_encoder  # noqa: E402
from src4.model.unet2d import UNet2DConditionModel  # noqa: E402

SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"
CKPTS = REPO / "src4" / "ckpts"


@contextmanager
def _timed(label: str):
    """Prints wall time + CPU RSS delta (+ CUDA memory, if available) for one load
    stage -- added to find where SonicDiffusionModel.__init__ actually spends time
    (network fetch vs. disk read vs. GPU transfer), not to keep permanently."""
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e9
    cuda0 = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    rss1 = proc.memory_info().rss / 1e9
    msg = f"[timing] {label}: {dt:.2f}s, cpu_rss {rss0:.2f}->{rss1:.2f}GB (+{rss1 - rss0:.2f}GB)"
    if torch.cuda.is_available():
        cuda1 = torch.cuda.memory_allocated() / 1e9
        msg += f", cuda_mem {cuda0:.2f}->{cuda1:.2f}GB (+{cuda1 - cuda0:.2f}GB)"
    print(msg)


class SonicDiffusionModel(ComposerModel):
    def __init__(self, gate_ckpt: Path = CKPTS / "landscape.pt",
                 adapter_ckpt: Path = CKPTS / "audio_projector_landscape.pth",
                 clap_ckpt: Path = CKPTS / "CLAP_weights_2022.pth"):
        super().__init__()
        t_start = time.perf_counter()

        with _timed("tokenizer"):
            self.tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
        with _timed("text_encoder"):
            self.text_encoder = CLIPTextModel.from_pretrained(SD_MODEL_ID, subfolder="text_encoder")
        with _timed("vae"):
            self.vae = AutoencoderKL.from_pretrained(SD_MODEL_ID, subfolder="vae")
        with _timed("unet (859M frozen + 34.6M adapter, random-init)"):
            self.unet = UNet2DConditionModel.from_pretrained(
                SD_MODEL_ID, subfolder="unet", use_adapter_list=[False, True, True],
                low_cpu_mem_usage=False, device_map=None)
        with _timed("noise_scheduler"):
            self.noise_scheduler = DDIMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")

        with _timed("unet adapter gate weights (landscape.pt, 109MB)"):
            gate_dict = torch.load(gate_ckpt, map_location="cpu")
            for name, param in self.unet.named_parameters():
                if "adapter" in name:
                    param.data = gate_dict[name]

        with _timed("CLAP audio encoder (CLAP_weights_2022.pth, 2.3GB)"):
            self.audio_encoder = load_clap_audio_encoder(clap_ckpt)
        with _timed("audio_projector (audio_projector_landscape.pth, 25MB)"):
            self.audio_projector = Adapter(audio_token_count=77, transformer_layer_count=4)
            self.audio_projector.load_state_dict(torch.load(adapter_ckpt, map_location="cpu"))

        print(f"[timing] SonicDiffusionModel.__init__ total: {time.perf_counter() - t_start:.2f}s")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.audio_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        for name, param in self.unet.named_parameters():
            if "adapter" in name:
                param.requires_grad_(True)
        self.audio_projector.requires_grad_(True)

        self.vae.eval(); self.text_encoder.eval(); self.audio_encoder.eval()

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.unet.parameters() if p.requires_grad] + list(self.audio_projector.parameters())

    def trainable_state_dict(self) -> dict:
        """Just the ~34.6M trained params (unet adapter layers + audio_projector),
        not Composer's default full-model checkpoint -- see train.py: saving all
        1.1B params (frozen VAE/text_encoder/CLAP/UNet included) writes a 5GB file
        per save, dwarfing what actually changed during training."""
        return {"unet_adapter": {n: p for n, p in self.unet.named_parameters() if "adapter" in n},
                "audio_projector": self.audio_projector.state_dict()}

    def load_trainable_state_dict(self, state: dict) -> None:
        for name, param in self.unet.named_parameters():
            if name in state["unet_adapter"]:
                param.data = state["unet_adapter"][name]
        self.audio_projector.load_state_dict(state["audio_projector"])

    def audio_context(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, n_samples) at CLAP's 44.1kHz/10s (see
        model/clap_audio_encoder.py:preprocess_waveform) -> (B,77,768) audio tokens."""
        with torch.no_grad():
            audio_emb = self.audio_encoder(waveform)  # (B,1024)
        return self.audio_projector(audio_emb.unsqueeze(1))  # (B,77,768)

    def encode_prompt(self, prompts: list[str]) -> torch.Tensor:
        input_ids = self.tokenizer(prompts, padding="max_length", truncation=True,
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt").input_ids.to(self.text_encoder.device)
        with torch.no_grad():
            return self.text_encoder(input_ids)[0]

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor

    def forward(self, batch: dict) -> dict:
        """Standard SD training step: predict the noise added to the target image's
        latent (epsilon-prediction)."""
        audio_context = self.audio_context(batch["waveform"])
        latents = self.encode_image(batch["pixel_values"])
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                   device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self.encode_prompt(batch["prompt"])
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, audio_context=audio_context).sample
        return {"model_pred": model_pred, "noise": noise}

    def loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        return F.mse_loss(outputs["model_pred"].float(), outputs["noise"].float())

    def get_metrics(self, is_train: bool = False) -> dict:
        return {}  # no cheap per-step metric; evaluation is qualitative, see callbacks.py

    def eval_forward(self, batch: dict, outputs: dict | None = None) -> dict:
        return outputs if outputs is not None else self.forward(batch)

    @torch.no_grad()
    def generate(self, waveform: torch.Tensor, prompt: str, start_image: Image.Image | None = None,
                 start_strength: float = 0.75, num_inference_steps: int = 30,
                 generator: torch.Generator | None = None) -> Image.Image:
        """Sample one image: SDEdit-style from start_image (typically the box's
        empty-box photo, see dataset.py:empty_box_image) if given, partially noised
        to start_strength and denoised the rest of the way; pure-noise txt2img
        otherwise. No classifier-free guidance -- this is a minimal sampler for
        periodic training-time visualization (callbacks.py), not final-quality
        generation."""
        device = next(self.unet.parameters()).device
        audio_context = self.audio_context(waveform.unsqueeze(0).to(device))
        encoder_hidden_states = self.encode_prompt([prompt])

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

        if start_image is not None:
            arr = np.array(start_image.resize((512, 512))).astype(np.float32) / 127.5 - 1.0
            pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            start_latents = self.encode_image(pixel_values)
            n_denoise_steps = max(1, int(num_inference_steps * start_strength))
            timesteps = self.noise_scheduler.timesteps[-n_denoise_steps:]
            noise = torch.randn(start_latents.shape, generator=generator, device=device)
            latents = self.noise_scheduler.add_noise(start_latents, noise, timesteps[:1])
        else:
            latents = torch.randn((1, self.unet.config.in_channels, 64, 64), generator=generator, device=device)
            timesteps = self.noise_scheduler.timesteps

        for t in timesteps:
            model_pred = self.unet(latents, t, encoder_hidden_states, audio_context=audio_context).sample
            latents = self.noise_scheduler.step(model_pred, t, latents).prev_sample

        image = (self.vae.decode(latents / self.vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)
        image = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image)
