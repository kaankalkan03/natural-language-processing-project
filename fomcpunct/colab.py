"""Move the transformer runs to Google Colab and back.

Usage (from the repository root):
    python -m fomcpunct.colab bundle [--out ../colab/p2_colab_bundle.zip]
    python -m fomcpunct.colab import <p2_transformer_results.zip> [--force]

`bundle` packs the code (fomcpunct/), the processed data and a manifest with
SHA-256 hashes into one zip to upload to Google Drive; the notebook
notebooks/transformers_colab.ipynb unpacks and checks it.

`import` reads the results zip that the notebook writes and copies its files
into results/ (and nowhere else) after checking that every run card was made
from the same data and the same transformer code as this repository, and with
the model, pinned revision and seed that its file name (<key>_s<seed>) says.
It warns about runs with a non-default Config or runs that did not use a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .transformer import CODE_HASH, MODELS, Config

DEFAULT_CONFIG = asdict(Config())
PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
DATA_FILES = ("data/processed/train.jsonl", "data/processed/validation.jsonl", "data/processed/test.jsonl",
              "data/processed/stats.json")
REQUIREMENTS = "transformers==5.20.2\nsentencepiece==0.2.2\n"
CODE_FILES = ("transformer.py", "metrics.py", "__init__.py")
RESULT_PARTS = ("predictions", "latency", "training", "tuning", "params")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle(out: Path) -> None:
    files = {f"fomcpunct/{p.name}": p for p in sorted(PACKAGE.glob("*.py"))}
    files.update({name: REPO / name for name in DATA_FILES})
    manifest = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "files": {name: sha256_bytes(path.read_bytes()) for name, path in files.items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("requirements-colab.txt", REQUIREMENTS)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files, created {manifest['created_utc']}, "
          f"transformer code hash {CODE_HASH})")


def local_fingerprint() -> dict:
    code = hashlib.sha256(b"".join((PACKAGE / name).read_bytes() for name in CODE_FILES)).hexdigest()[:16]
    data = {s: sha256_bytes((REPO / "data" / "processed" / f"{s}.jsonl").read_bytes())
            for s in ("train", "validation", "test")}
    return {"code": code, "data": data}


def import_results(archive_path: Path, force: bool) -> None:
    expected = local_fingerprint()
    results_root = (REPO / "results").resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        for name in names:
            target = (REPO / name).resolve()
            if not name.startswith("results/") or results_root not in target.parents:
                raise SystemExit(f"refusing a path outside results/: {name}")
        cards = {Path(n).stem: n for n in names if n.startswith("results/training/") and n.endswith(".json")}
        if not cards:
            raise SystemExit("no run cards (results/training/*.json) in the archive")
        for stem, name in cards.items():
            try:
                card = json.loads(archive.read(name))
                run, environment = card["identity"], card.get("environment", {})
                key, seed = stem.rsplit("_s", 1)
                model, revision = MODELS[key]
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise SystemExit(f"{name}: not a valid run card ({error!r})")
            if run["data"] != expected["data"]:
                raise SystemExit(f"{name}: made from different data than data/processed")
            if run["code"] != expected["code"]:
                raise SystemExit(f"{name}: made with different transformer code (code hash {run['code']} "
                                 f"vs local {expected['code']})")
            if (run.get("model"), run.get("revision"), run.get("seed")) != (model, revision, int(seed)):
                raise SystemExit(f"{name}: card is for {run.get('model')}@{run.get('revision')} seed "
                                 f"{run.get('seed')}, not {model}@{revision} seed {seed}")
            if run.get("config") != DEFAULT_CONFIG:
                print(f"warning: {name} used a non-default config: {run.get('config')}")
            if str(environment.get("device", "cpu")).startswith("cpu"):
                print(f"warning: {name} was trained on a CPU, not a GPU")
        for name in names:
            part = name.split("/")[1]
            if part in ("predictions", "latency") and Path(name).name.split(".")[0] not in cards:
                raise SystemExit(f"{name} has no run card in the archive")
        targets = [(n, REPO / n) for n in names if n.split("/")[1] in RESULT_PARTS]
        clashes = [str(target) for name, target in targets
                   if target.exists() and target.read_bytes() != archive.read(name)]
        if clashes and not force:
            raise SystemExit("would overwrite different files (use --force):\n  " + "\n  ".join(clashes))
        for name, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    print(f"imported {len(targets)} files from {archive_path} ({len(cards)} run cards checked)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--out", type=Path, default=REPO.parent / "colab" / "p2_colab_bundle.zip")
    i = sub.add_parser("import")
    i.add_argument("archive", type=Path)
    i.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "bundle":
        bundle(args.out)
    else:
        import_results(args.archive, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
