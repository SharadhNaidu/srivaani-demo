from __future__ import annotations

import sys

from .config import MODEL_REPOS, UPSTREAM_REPO
from .engine import resolve_token


def main():
    from huggingface_hub import snapshot_download

    token = resolve_token()
    last = None
    for repo in MODEL_REPOS:
        try:
            kwargs = dict(allow_patterns=["*.ts", "*.pt", "*.model", "*.json", "*.py"])
            if token and repo == UPSTREAM_REPO:
                kwargs["token"] = token
            path = snapshot_download(repo, **kwargs)
            print("[ok] model ready (%s)" % repo)
            print("     %s" % path)
            return 0
        except Exception as exc:
            last = exc
            print("[..] could not fetch %s (%s)" % (repo, str(exc)[:90]))

    print("[!!] could not download the model from any source.")
    text = str(last or "")
    low = text.lower()
    if "connect" in low or "resolve" in low or "network" in low or "timeout" in low:
        print("     no internet connection. connect and run setup.bat again.")
    else:
        print("     %s" % text[:250])
    return 1


if __name__ == "__main__":
    sys.exit(main())
