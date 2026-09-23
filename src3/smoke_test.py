"""Run AudioToken's actual pretrained model (BEATs encoder + released embedder +
CompVis/stable-diffusion-v1-4) end to end on one real audio clip, to confirm the
downloaded checkpoints and vendored code work before we touch anything.

Reuses vendor/modules/AudioToken/AudioToken.py:AudioTokenWrapper unmodified -- this
script only supplies the args/audio/accelerator scaffolding that vendor/inference.py
normally gets from its VGGSound dataloader + argparse (which needs a full VGGSound-
format data_dir we don't have), replicating vendor/inference.py's core forward pass
(lines ~178-196) directly on one waveform instead.

Usage:
    python src3/smoke_test.py [path/to/audio.wav]
    (with no argument, uses a short synthesized chirp as a stand-in "audio" clip)
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "src3" / "vendor"
sys.path.insert(0, str(VENDOR))  # vendor code imports itself as `modules.AudioToken...`
os.chdir(VENDOR)  # AudioTokenWrapper hardcodes a `models/BEATs/...`-relative checkpoint path

import torch
import torchaudio
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer

from modules.AudioToken.AudioToken import AudioTokenWrapper

SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"
PLACEHOLDER_TOKEN = "<*>"
INPUT_LENGTH_SEC = 5
AUDIO_SAMPLE_RATE = 16000


def load_audio(path: Path | None) -> torch.Tensor:
    if path is not None:
        wav, sr = torchaudio.load(str(path))
        return wav[0]
    sys.path.insert(0, str(REPO / "src"))
    from data.audio import make_chirp
    audio = make_chirp(T_sec=INPUT_LENGTH_SEC, T_start=0, T_end=0, fs=AUDIO_SAMPLE_RATE, f_start=200, f_end=2000)
    return torch.from_numpy(audio.astype("float32") / 32767.0)


def main(audio_path: Path | None, out_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = argparse.Namespace(
        pretrained_model_name_or_path=SD_MODEL_ID, revision=None, data_set="test",
        learned_embeds=str(VENDOR / "output/embedder_learned_embeds.bin"), lora=False,
    )
    accelerator = argparse.Namespace(device=device)

    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
    tokenizer.add_tokens(PLACEHOLDER_TOKEN)
    placeholder_token_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)

    model = AudioTokenWrapper(args, accelerator).to(device).eval()
    model.text_encoder.resize_token_embeddings(len(tokenizer))

    audio = load_audio(audio_path).to(device).unsqueeze(0)  # (1, samples)
    with torch.no_grad():
        aud_features = model.aud_encoder.extract_features(audio)[1]
        audio_token = model.embedder(aud_features)
        model.text_encoder.get_input_embeddings().weight.data[placeholder_token_id] = audio_token.clone()

        pipeline = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID, tokenizer=tokenizer, text_encoder=model.text_encoder,
            vae=model.vae, unet=model.unet,
        ).to(device)
        image = pipeline(f"a photo of {PLACEHOLDER_TOKEN}, 4k, high resolution",
                          num_inference_steps=30, guidance_scale=7.5).images[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="?", type=Path, default=None)
    p.add_argument("--out", type=Path, default=REPO / "src3" / "smoke_test_output.png")
    parsed = p.parse_args()
    main(parsed.audio, parsed.out)
