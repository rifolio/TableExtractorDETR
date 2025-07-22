import logging
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s %(levelname)-5s %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def ensure_local_model(repo_id: str, target_dir: Path) -> Path:
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info("✅ Local model already present at %s", target_dir)
    else:
        logger.info("📥 Local model not found. Downloading %s → %s", repo_id, target_dir)
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False
            )
            logger.info("✅ Download complete")
        except Exception:
            logger.exception("❌ Failed to download model %s", repo_id)
            sys.exit(1)
    return target_dir

if __name__ == "__main__":
    MODEL_REPOS = [
        "microsoft/table-transformer-detection",
        "microsoft/table-structure-recognition-v1.1-all"
    ]

    BASE_DIR = Path(__file__).resolve().parent / "models"

    for repo_id in MODEL_REPOS:
        repo_name = repo_id.split("/")[-1]
        target = BASE_DIR / repo_name
        ensure_local_model(repo_id, target)
