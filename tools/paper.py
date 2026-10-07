#!/usr/bin/env python3
"""Create and list one-paper projects inside papers/."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"
TEMPLATE = ROOT / "tools" / "template"
CATALOG = ROOT / "PAPERS.md"
WORKSPACE = ROOT / "paper-readings.code-workspace"
STATUSES = ("unread", "reading", "read")
REQUIRED_FILES = ("README.md", "notes.md", "questions.md", "summary.md", "paper.json")


def slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")[:60].rstrip("-")
    return slug or "paper"


def load_projects() -> list[dict]:
    projects: list[dict] = []
    if not PAPERS.is_dir():
        return projects
    for child in PAPERS.iterdir():
        meta_path = child / "paper.json"
        if not child.is_dir() or not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        projects.append(meta)
    projects.sort(key=lambda meta: meta.get("title", "").lower())
    return projects


def write_index(projects: list[dict]) -> None:
    lines = [
        "# Paper projects",
        "",
        "Each row is one paper. Every project lives in `papers/`, and that folder is only for these projects.",
        "",
        "| Project | Status | Source |",
        "| --- | --- | --- |",
    ]
    if not projects:
        lines.append("| _None yet_ | | |")
    for meta in projects:
        title = meta["title"].replace("|", "\\|")
        slug = meta["slug"]
        url = (meta.get("sourceUrl") or "").strip()
        source = f"[link]({url})" if url else "Link not recorded yet"
        lines.append(
            f"| [{title}](papers/{slug}/README.md) | {meta['status']} | {source} |"
        )
    lines.append("")
    CATALOG.write_text("\n".join(lines), encoding="utf-8")

    folders = [{"name": "Paper readings", "path": "."}]
    for meta in projects:
        folders.append({"name": meta["title"], "path": f"papers/{meta['slug']}"})
    workspace = {
        "folders": folders,
        "settings": {"files.exclude": {"**/.git": True}},
    }
    WORKSPACE.write_text(json.dumps(workspace, indent=2) + "\n", encoding="utf-8")


def render(text: str, mapping: dict[str, str]) -> str:
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def cmd_new(args: argparse.Namespace) -> None:
    title = args.title.strip()
    if not title:
        sys.exit("A paper title is required.")
    slug = args.slug or slugify(title)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        sys.exit(
            f"Slug {slug!r} can only use lowercase letters, numbers, and hyphens."
        )
    dest = PAPERS / slug
    if dest.exists():
        sys.exit(f"papers/{slug} already exists.")
    if args.status not in STATUSES:
        sys.exit(f"Status must be one of: {', '.join(STATUSES)}.")

    url = args.url.strip()
    website = args.website.strip()
    origin = args.origin.strip()
    created = date.today().isoformat()
    mapping = {
        "TITLE": title,
        "SLUG": slug,
        "STATUS": args.status,
        "URL": url or "Not recorded yet",
        "WEBSITE": website or "Not recorded yet",
        "CREATED": created,
        "ORIGIN_BLOCK": f"{origin}\n\n" if origin else "",
    }

    PAPERS.mkdir(parents=True, exist_ok=True)
    dest.mkdir()
    for src in sorted(TEMPLATE.iterdir()):
        if not src.is_file() or src.name.startswith("."):
            continue
        (dest / src.name).write_text(
            render(src.read_text(encoding="utf-8"), mapping),
            encoding="utf-8",
        )
    meta = {
        "slug": slug,
        "title": title,
        "status": args.status,
        "sourceUrl": url,
        "website": website,
        "created": created,
        "origin": origin,
    }
    (dest / "paper.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    write_index(load_projects())
    print(f"Created papers/{slug}")


def cmd_list(_args: argparse.Namespace) -> None:
    projects = load_projects()
    if not projects:
        print("No paper projects yet.")
        return
    for meta in projects:
        url = meta.get("sourceUrl") or "no link yet"
        print(f"{meta['status']:8}  {meta['title']}  ({url})")


def cmd_index(_args: argparse.Namespace) -> None:
    projects = load_projects()
    write_index(projects)
    print(f"Updated the catalog for {len(projects)} paper project(s).")


def cmd_check(_args: argparse.Namespace) -> None:
    errors: list[str] = []
    if not PAPERS.is_dir():
        errors.append("papers/ is missing.")
    else:
        for child in sorted(PAPERS.iterdir()):
            if child.name == "README.md" and child.is_file():
                continue
            if not child.is_dir():
                errors.append(f"papers/{child.name} is not a paper project directory.")
                continue
            meta_path = child / "paper.json"
            if not meta_path.is_file():
                errors.append(f"papers/{child.name} has no paper.json.")
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("slug") != child.name:
                errors.append(
                    f"papers/{child.name} slug does not match its folder name."
                )
            for required in REQUIRED_FILES:
                if not (child / required).is_file():
                    errors.append(f"papers/{child.name} is missing {required}.")
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(f"{len(load_projects())} paper project(s) in papers/.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep each research paper in its own project under papers/."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    new = sub.add_parser("new", help="Create a project for one paper")
    new.add_argument("title")
    new.add_argument("--url", default="")
    new.add_argument("--website", default="")
    new.add_argument("--status", default="unread", choices=list(STATUSES))
    new.add_argument("--slug", default="")
    new.add_argument("--origin", default="")
    new.set_defaults(func=cmd_new)

    listed = sub.add_parser("list", help="List paper projects")
    listed.set_defaults(func=cmd_list)

    index = sub.add_parser("index", help="Rebuild PAPERS.md and the workspace file")
    index.set_defaults(func=cmd_index)

    check = sub.add_parser("check", help="Confirm papers/ contains only paper projects")
    check.set_defaults(func=cmd_check)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
