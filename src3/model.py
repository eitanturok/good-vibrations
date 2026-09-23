"""AudioTokenModel: wires our BEATs seam (beats_input.py) + the released embedder
into a frozen CompVis/stable-diffusion-v1-4, the way
vendor/modules/AudioToken/AudioToken.py:AudioTokenWrapper does for real audio.

Trainable parameters: the embedder MLP only, matching AudioTokenWrapper's own
`requires_grad_(False)` calls on vae/unet/text_encoder/aud_encoder -- no LoRA, no
UNet fine-tuning. `--finetune-beats` additionally unfreezes BEATs. This keeps the
pretrained SD prior (and BEATs' pretrained weights, up to --finetune-beats) fully
intact; only the audio-to-pseudo-word adapter is learned/continued on our data.

Two different text-encoder call paths, both loading the SAME frozen SD weights:
  - training: vendor's custom modules/clip_text_model/modeling_clip.py:CLIPTextModel,
    called as `text_encoder(audio_token, input_ids=...)` -- properly batched, each
    row's placeholder token position gets that row's own audio_token (see
    CLIPTextEmbeddings.forward: `inputs_embeds[indices] = audio_e`).
  - sampling (generate()): a plain transformers.CLIPTextModel, whose placeholder
    embedding row is overwritten in place before each call -- batch=1, matching
    vendor/inference.py's own approach exactly (fine for periodic eval sampling).
Both start from the exact same frozen weights; the only "trained" state that ever
differs between them is the placeholder row, which is always set explicitly right
before use, so there's nothing to keep in sync between the two.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "src3" / "vendor"
if str(VENDOR) not in sys.path: sys.path.insert(0, str(VENDOR))
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from modules.BEATs.BEATs import BEATs, BEATsConfig
from modules.AudioToken.embedder import FGAEmbedder
from modules.clip_text_model.modeling_clip import CLIPTextModel as AudioCLIPTextModel
from diffusers import (AutoencoderKL, UNet2DConditionModel, DDPMScheduler,
                        StableDiffusionPipeline, StableDiffusionImg2ImgPipeline)
from transformers import CLIPTextModel, CLIPTokenizer

from src3.beats_input import beats_extract_features

SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"
PLACEHOLDER_TOKEN = "<*>"
BEATS_CKPT = VENDOR / "models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
EMBEDDER_CKPT = VENDOR / "output/embedder_learned_embeds.bin"


class AudioTokenModel(torch.nn.Module):
    def __init__(self, finetune_beats: bool = False, embedder_ckpt: Path = EMBEDDER_CKPT):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
        self.tokenizer.add_tokens(PLACEHOLDER_TOKEN)
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)

        self.text_encoder = AudioCLIPTextModel.from_pretrained(SD_MODEL_ID, subfolder="text_encoder")
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.vae = AutoencoderKL.from_pretrained(SD_MODEL_ID, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(SD_MODEL_ID, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")

        ckpt = torch.load(BEATS_CKPT, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        self.aud_encoder = BEATs(cfg)
        self.aud_encoder.load_state_dict(ckpt["model"])
        self.aud_encoder.predictor = None

        self.embedder = FGAEmbedder(input_size=768 * 3, output_size=768)
        self.embedder.load_state_dict(torch.load(embedder_ckpt, map_location="cpu"))

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.aud_encoder.requires_grad_(finetune_beats)
        self.embedder.requires_grad_(True)
        self.finetune_beats = finetune_beats

        self.vae.eval(); self.unet.eval(); self.text_encoder.eval()
        self.aud_encoder.train(finetune_beats)
        self.embedder.train()

        # lazily built + cached on first generate() call (see generate()) -- previously
        # every single generate() call reconstructed the eval text encoder AND the full
        # SD pipeline from scratch via from_pretrained, which is pure overhead: nothing
        # about them changes call to call except the placeholder embedding row, which is
        # cheap to overwrite in place. Caching cut a real, measured slowdown during eval
        # (10 generate() calls per eval interval with the train_viz/eval_viz split).
        self.eval_text_encoder = None
        self.eval_pipe_txt2img = None
        self.eval_pipe_img2img = None

    def clear_eval_cache(self) -> None:
        """Free the cached eval text encoder/pipelines (see generate()) after an eval
        pass finishes. They must NOT stay resident across training steps -- caching
        them within one eval pass (many generate() calls back to back) is the win;
        holding them permanently instead just adds a second copy of ~everything SD-
        related on top of training's own memory and OOMs (measured: pushed a 16GB GPU
        over budget during a normal training step, well after eval had returned)."""
        self.eval_text_encoder = None
        self.eval_pipe_txt2img = None
        self.eval_pipe_img2img = None
        torch.cuda.empty_cache()

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        params = list(self.embedder.parameters())
        if self.finetune_beats:
            params += list(self.aud_encoder.parameters())
        return params

    def audio_token(self, fbank: torch.Tensor) -> torch.Tensor:
        """fbank: (B,T,128) from beats_input.to_beats_fbank, batched -> (B,768)."""
        _, layers_sum, _ = beats_extract_features(self.aud_encoder, fbank)
        return self.embedder(layers_sum)

    def encode_prompt(self, prompts: list[str], audio_token: torch.Tensor) -> torch.Tensor:
        input_ids = self.tokenizer(prompts, padding="max_length", truncation=True,
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt").input_ids.to(audio_token.device)
        return self.text_encoder(audio_token, input_ids=input_ids)[0]

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor

    def forward(self, pixel_values: torch.Tensor, fbank: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Standard SD training step: predict the noise added to the target
        image's latent, MSE loss (matches vendor/train.py's default prediction_type
        == 'epsilon' path)."""
        audio_token = self.audio_token(fbank)
        with torch.no_grad():
            latents = self.encode_image(pixel_values)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.encode_prompt(prompts, audio_token)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return F.mse_loss(model_pred.float(), noise.float())

    @torch.no_grad()
    def generate(self, fbank: torch.Tensor, prompt: str, start_image: Image.Image | None = None,
                 start_strength: float = 0.75, num_inference_steps: int = 50, guidance_scale: float = 7.5,
                 generator: torch.Generator | None = None) -> Image.Image:
        """Sample one image. fbank: (1,T,128), one sample. start_image=None -> vanilla
        AudioToken txt2img (pure random latent, start_strength ignored); otherwise an
        SDEdit-style img2img: encode start_image, partially noise it to
        `start_strength` (1.0 == pure noise, 0.0 == the start image itself), denoise
        from there for the remaining steps."""
        device = next(self.unet.parameters()).device
        audio_token = self.audio_token(fbank)  # (1,768)

        if self.eval_text_encoder is None:
            self.eval_text_encoder = CLIPTextModel.from_pretrained(SD_MODEL_ID, subfolder="text_encoder").to(device)
            self.eval_text_encoder.resize_token_embeddings(len(self.tokenizer))
            self.eval_text_encoder.requires_grad_(False)
            self.eval_text_encoder.eval()
        # overwritten in place every call -- this (not the encoder/pipeline objects) is
        # the only thing that actually changes between samples
        self.eval_text_encoder.get_input_embeddings().weight.data[self.placeholder_token_id] = audio_token[0]

        if start_image is None:
            # vanilla AudioToken: pure txt2img from a random latent
            if self.eval_pipe_txt2img is None:
                self.eval_pipe_txt2img = StableDiffusionPipeline.from_pretrained(
                    SD_MODEL_ID, tokenizer=self.tokenizer, text_encoder=self.eval_text_encoder, vae=self.vae, unet=self.unet,
                ).to(device)
                self.eval_pipe_txt2img.set_progress_bar_config(disable=True)
            image = self.eval_pipe_txt2img(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                            generator=generator).images[0]
        else:
            # SDEdit-style: encode start_image, partially noise to start_strength, denoise the rest
            if self.eval_pipe_img2img is None:
                self.eval_pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    SD_MODEL_ID, tokenizer=self.tokenizer, text_encoder=self.eval_text_encoder, vae=self.vae, unet=self.unet,
                ).to(device)
                self.eval_pipe_img2img.set_progress_bar_config(disable=True)
            image = self.eval_pipe_img2img(prompt, image=start_image, strength=start_strength, num_inference_steps=num_inference_steps,
                                            guidance_scale=guidance_scale, generator=generator).images[0]
        return image
