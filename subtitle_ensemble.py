#!/usr/bin/env python3

# Several ASR backends (MLX, faster-whisper/CTranslate2, torch) each ship their
# own OpenMP runtime; loading them in one process aborts with "OMP Error #15".
# Allow the duplicate runtime before any of them import.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from subtitle_extractor.cli import main

raise SystemExit(main())
