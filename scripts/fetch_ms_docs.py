"""
Step 2a: Download a curated slice of official Microsoft Learn documentation.

What this does, in plain language:
  1. For each public Microsoft docs GitHub repo in config.MS_DOCS_SOURCES,
     it does a "sparse" git clone -- it only downloads the specific
     sub-folders we listed (e.g. Intune's "device-enrollment" docs), not
     the whole repo. These repos have thousands of pages; we want a
     representative slice, not everything.
  2. For each Markdown page in those folders, it strips the YAML
     "frontmatter" header (title/description/metadata that Microsoft's
     publishing pipeline uses internally) and some Learn-specific markdown
     syntax that isn't useful as plain text (image embeds, moniker tags).
  3. It saves the cleaned Markdown to data/raw/ms_docs/<product>/ and
     records metadata (title, product, exact source file on GitHub) in a
     manifest so later steps -- and citations in the final answers -- know
     where every chunk came from.

Why sparse clone instead of the requests+BeautifulSoup scraping approach:
learn.microsoft.com pages are rendered from these same Markdown files, so
cloning gets us clean source text directly, with no HTML-scraping,
rate-limiting, or "did I strip the nav bar correctly" problems.

Run it with:
    venv\\Scripts\\python.exe scripts\\fetch_ms_docs.py
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n", re.DOTALL)
IMAGE_DIRECTIVE_RE = re.compile(r":::image[^:]*?:::", re.DOTALL)
MONIKER_LINE_RE = re.compile(r"^:::moniker(-end)?.*$\n?", re.MULTILINE)
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
WHITESPACE_ONLY_LINE_RE = re.compile(r"^[ \t]+$", re.MULTILINE)
EXTRA_BLANK_LINES_RE = re.compile(r"\n{3,}")


def run_git(args, cwd=None):
    """Run a git command, raising with its stderr if it fails."""
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


def sparse_clone(repo_url: str, folders: list[str], dest: Path) -> None:
    """Clone only `folders` from `repo_url` into `dest` (no full checkout)."""
    run_git(["clone", "--no-checkout", "--depth", "1", "--filter=blob:none", repo_url, str(dest)])
    run_git(["sparse-checkout", "init", "--cone"], cwd=dest)
    run_git(["sparse-checkout", "set", *folders], cwd=dest)
    run_git(["checkout", "main"], cwd=dest)


def clean_markdown(raw_text: str) -> tuple[dict, str]:
    """Strip YAML frontmatter + Learn-specific markup. Returns (frontmatter_dict, body)."""
    frontmatter = {}
    match = FRONTMATTER_RE.match(raw_text)
    body = raw_text
    if match:
        try:
            frontmatter = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            frontmatter = {}
        body = raw_text[match.end():]

    body = IMAGE_DIRECTIVE_RE.sub("", body)
    body = MONIKER_LINE_RE.sub("", body)
    body = MD_IMAGE_RE.sub("", body)
    # Removing an indented image directive often leaves a line of bare
    # spaces behind (e.g. inside a numbered list) -- flatten those to
    # truly-empty lines so the next step can collapse them.
    body = WHITESPACE_ONLY_LINE_RE.sub("", body)
    body = EXTRA_BLANK_LINES_RE.sub("\n\n", body)
    return frontmatter, body.strip()


def first_markdown_heading(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def collect_markdown_files(clone_dir: Path, folder: str) -> list[Path]:
    """Find .md files under `folder`, skipping non-content sub-dirs, sorted for determinism."""
    folder_path = clone_dir / folder
    if not folder_path.exists():
        print(f"  ! folder not found in repo: {folder}")
        return []

    files = []
    for path in folder_path.rglob("*.md"):
        if any(part in config.MS_DOCS_SKIP_DIR_NAMES for part in path.relative_to(folder_path).parts):
            continue
        files.append(path)

    files.sort()
    return files[: config.MS_DOCS_MAX_FILES_PER_FOLDER]


def fetch_source(source: dict, manifest: list[dict]) -> int:
    repo_url = source["repo"]
    product = source["product"]
    folders = source["folders"]
    repo_name = repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")

    out_dir = config.RAW_MS_DOCS_DIR / product
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {product} ({repo_name}) ===")
    with tempfile.TemporaryDirectory(prefix="msdocs_") as tmp:
        clone_dir = Path(tmp)
        print(f"  cloning (sparse: {len(folders)} folder(s))...")
        sparse_clone(repo_url, folders, clone_dir)

        saved = 0
        for folder in folders:
            files = collect_markdown_files(clone_dir, folder)
            print(f"  {folder}: {len(files)} file(s)")

            for src_path in files:
                raw_text = src_path.read_text(encoding="utf-8", errors="ignore")
                frontmatter, body = clean_markdown(raw_text)
                if len(body) < 200:
                    # Skip near-empty pages (redirect stubs, includes-only pages).
                    continue

                title = frontmatter.get("title") or first_markdown_heading(body) or src_path.stem
                relative_path = src_path.relative_to(clone_dir).as_posix()

                # Flatten into a single file per doc; the manifest keeps the
                # real hierarchy (product/folder/filename) for citations.
                safe_name = relative_path.replace("/", "__")
                dest_path = out_dir / safe_name
                dest_path.write_text(body, encoding="utf-8")

                manifest.append({
                    "id": f"msdoc_{product}_{safe_name.removesuffix('.md')}",
                    "product": product,
                    "title": title.strip(),
                    "local_path": dest_path.relative_to(config.PROJECT_ROOT).as_posix(),
                    "source_url": f"https://github.com/MicrosoftDocs/{repo_name}/blob/main/{relative_path}",
                })
                saved += 1

        print(f"  -> saved {saved} page(s) to {out_dir}")
        return saved


def main():
    config.ensure_dirs()
    manifest: list[dict] = []
    total = 0

    for source in config.MS_DOCS_SOURCES:
        total += fetch_source(source, manifest)

    manifest_path = config.RAW_MS_DOCS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. {total} Microsoft Learn pages saved under {config.RAW_MS_DOCS_DIR}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
