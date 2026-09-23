"""Download the AudioToken pretrained checkpoint (BEATs encoder) and cache the base
Stable Diffusion weights that the released embedder was fine-tuned against.

The embedder itself (vendor/output/embedder_learned_embeds.bin, ~35MB) is already
vendored -- it ships inside the AudioToken git repo, cloned into src3/vendor/. This
script only fetches the two large pieces that aren't: the BEATs audio-encoder
checkpoint (~350MB) and the CompVis/stable-diffusion-v1-4 weights (into the normal HF
hub cache) -- v1-4 because that's the base model the released embedder was trained
against (AudioTokenWrapper sizes the embedder's output to 768 only when
pretrained_model_name_or_path == "CompVis/stable-diffusion-v1-4"; stable-diffusion-2
would silently load an embedder shaped for the wrong dimension).

The AudioToken README's BEATs download URL (an Azure blob SAS link) is dead (403,
expired signature). Microsoft's own OneDrive links (github.com/microsoft/unilm/tree/master/beats)
aren't script-downloadable. Using a HF Hub mirror of the exact same file instead
(byte-identical checkpoint, verified filename match: BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt).

Usage:
    python src3/download.py
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

VENDOR = Path(__file__).resolve().parent / "vendor"
BEATS_REPO = "WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2"
BEATS_FILENAME = "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
BEATS_PATH = VENDOR / "models/BEATs" / BEATS_FILENAME
SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"


def download_beats() -> None:
    if BEATS_PATH.exists():
        print(f"BEATs checkpoint already at {BEATS_PATH}")
        return
    BEATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=BEATS_REPO, filename=BEATS_FILENAME)
    BEATS_PATH.symlink_to(downloaded)
    print(f"downloaded BEATs checkpoint to {BEATS_PATH} (symlinked from HF cache)")


def download_sd() -> None:
    from diffusers import StableDiffusionPipeline
    print(f"caching {SD_MODEL_ID} into the HF hub cache...")
    StableDiffusionPipeline.from_pretrained(SD_MODEL_ID)
    print("done")


if __name__ == "__main__":
    download_beats()
    download_sd()
