"""Check the delivery using only Python's standard library; never download data."""
import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "train_indices.npy": "24520246ef0988e61bb0fe0f5c66da3a1f944be0569ccb5f0afc2536b92dfc15",
    "val_indices.npy": "2c80060ed0fa8eba98b2023be25aa56794982c9d6bf762dd1eb29205835d67c2",
}
ACTIVE_DOCS = [
    "README.md", "docs/running.md", "docs/experiment_protocol.md",
    "augmentations/README.md", "models/README.md", "data/README.md",
    "configs/README.md", "scripts/README.md", "tests/README.md",
    "analysis/README.md", "results/README.md", "results/figures/README.md",
    "report/README.md", "presentation/README.md", "third_party/README.md",
    "data/splits/README.md", "report/figures/README.md", "presentation/assets/README.md",
    "notebooks/README.md", "experiments/README.md", "CONTRIBUTING.md", "AGENTS.md",
    "README_B.md", "README_C.md", "docs/submission_validation.md",
]


def verify():
    manifest = json.loads((ROOT / "third_party/cutmix_provenance.json").read_text())
    upstream = ROOT / "third_party/CutMix-PyTorch"
    for item in manifest["files"]:
        path = upstream / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Upstream snapshot changed: {path.relative_to(ROOT)}")
    for name, expected in SPLITS.items():
        if hashlib.sha256((ROOT / "data/splits" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Fixed split changed: {name}")
    for name in ["train.py", "evaluate.py", "requirements.txt", "requirements-dev.txt",
                 "configs/smoke_cpu.yaml", "scripts/smoke_test.py", "analysis/reproduce_results.py"]:
        if not (ROOT / name).is_file():
            raise ValueError(f"Missing entry point: {name}")
    sources = [ROOT / "train.py", ROOT / "evaluate.py"]
    for folder in ("augmentations", "models", "data", "scripts", "analysis", "tests"):
        sources.extend((ROOT / folder).glob("*.py"))
    for source in sources:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source.relative_to(ROOT)))
    # Check file targets in current user-facing docs. Historical reports retain their original context.
    link_pattern = re.compile(r"\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")
    for name in ACTIVE_DOCS:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        if re.search(r"^(<<<<<<< |=======\s*$|>>>>>>> )", text, re.M):
            raise ValueError(f"Unresolved conflict in {name}")
        for match in link_pattern.finditer(text):
            target = match.group(1).strip("<>")
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if target and not (path.parent / target).exists():
                raise ValueError(f"Broken local link in {name}: {target}")
    return {"status": "passed", "upstream_files_verified": len(manifest["files"]),
            "fixed_split_files_verified": len(SPLITS), "python_files_parsed": len(sources),
            "current_documents_checked": len(ACTIVE_DOCS)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
