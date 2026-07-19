from pathlib import Path

PIPELINE_VERSION = "ensemble-v1"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts", ".flv"}

MLX_LARGE_REPO = "mlx-community/whisper-large-v3-mlx"
MLX_TURBO_REPO = "mlx-community/whisper-large-v3-turbo"
FASTER_WHISPER_MODEL = "large-v3"
CLASSLA_MODEL = "classla/wav2vec2-xls-r-juznevesti-sr"

# The ensemble forced-alignment/text profile is only calibrated for Serbian.
# Other languages need their own aligner and evaluation before being supported.
SUPPORTED_LANGUAGES = {"sr"}

BRANCH_FAMILIES = {
    "original": (
        "original_mlx_large",
        "original_mlx_turbo",
        "original_faster_whisper",
    ),
    "slowed": ("slow70_mlx", "slow50_mlx"),
    "resolve": ("resolve21_vi50_mlx",),
}

DEFAULT_RESOLVE_APP = Path("/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/MacOS/Resolve")
