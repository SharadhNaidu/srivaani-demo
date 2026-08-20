from __future__ import annotations

import sys

from .config import MODEL_REPO
from .engine import resolve_token


def main():
    from huggingface_hub import snapshot_download

    token = resolve_token()
    try:
        path = snapshot_download(
            MODEL_REPO,
            token=token,
            allow_patterns=["*.ts", "*.pt", "*.model", "*.json", "*.py"],
        )
    except Exception as exc:
        text = str(exc)
        low = text.lower()
        print("[!!] Could not download %s" % MODEL_REPO)
        if "401" in text or "403" in text or "gated" in low:
            print("     The model is gated. Accept the terms at")
            print("     https://huggingface.co/%s" % MODEL_REPO)
            print("     then put a valid token in .env as HF_PAT")
        else:
            print("     %s" % text[:300])
        return 1

    print("[ok] Model ready at %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
