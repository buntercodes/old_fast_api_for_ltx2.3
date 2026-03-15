import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, GatedRepoError

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("models").resolve()

def download_required_models() -> dict[str, str]:
    """
    Downloads the required models for LTX-2.3 Distilled Pipeline if they are not already present.
    Returns a dictionary of paths to be used for loading the pipeline.
    """
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Checking/Downloading LTX-2.3 Checkpoint...")
    checkpoint_path = hf_hub_download(
        repo_id="Lightricks/LTX-2.3",
        filename="ltx-2.3-22b-distilled.safetensors",
        local_dir=str(DEFAULT_MODELS_DIR),
    )
    
    logger.info("Checking/Downloading Spatial Upsampler...")
    upsampler_path = hf_hub_download(
        repo_id="Lightricks/LTX-2.3",
        filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        local_dir=str(DEFAULT_MODELS_DIR),
    )
    
    logger.info("Checking/Downloading Gemma 3 12B IT (QAT Q4_0)...")
    try:
        gemma_dir = snapshot_download(
            repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
            local_dir=str(DEFAULT_MODELS_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"),
        )
    except (GatedRepoError, HfHubHTTPError) as e:
        logger.error(
            "\n" + "="*80 + "\n"
            "🚨 AUTHENTICATION REQUIRED FOR GEMMA 3 🚨\n"
            "The model 'google/gemma-3-12b-it-qat-q4_0-unquantized' is gated.\n"
            "1. Visit https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized and accept the terms.\n"
            "2. Generate an access token at https://huggingface.co/settings/tokens\n"
            "3. Set the token as an environment variable before running the server:\n"
            "   Windows (PowerShell): $env:HF_TOKEN=\"your_token\"\n"
            "   Windows (CMD): set HF_TOKEN=your_token\n"
            "   Linux/macOS: export HF_TOKEN=\"your_token\"\n"
            "   Or log in via CLI: huggingface-cli login\n"
            + "="*80 + "\n"
        )
        raise RuntimeError("Failed to download Gemma 3 due to missing or invalid Hugging Face token.") from e
        
    logger.info("All models are successfully downloaded and ready.")
    
    return {
        "checkpoint_path": checkpoint_path,
        "upsampler_path": upsampler_path,
        "gemma_path": gemma_dir,
    }
