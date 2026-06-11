"""
Setup script for third-party tools used by RareSim.

Run once after cloning the repo:
    python scripts/setup/setup_third_party.py

Skips tools that are already set up.
"""

import subprocess
import sys
from pathlib import Path

from raresim.utils.paths import THIRD_PARTY_DIR

# List of third-party tools to set up
THIRD_PARTY_TOOLS = [
    {
        "name": "fast_hpo_cr",
        "repo": "https://github.com/tudorgroza/fast_hpo_cr.git",
        "target_dir": THIRD_PARTY_DIR / "fast_hpo_cr",
        "check_file": THIRD_PARTY_DIR / "fast_hpo_cr" / "HPOAnnotator.py",
    },
]


def _clone(repo: str, target: Path) -> bool:
    """Clone a git repo. Returns True on success"""
    print(f"Cloning {repo} into {target} ...")

    result = subprocess.run(
        ["git", "clone", repo, str(target)],
        check=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Failed to clone {repo}: {result.stderr}")
        return False

    print(f"Successfully cloned {repo}")
    return True


def _is_setup(tool: dict) -> bool:
    """Check if a tool is already cloned by looking for a specific file."""
    return tool["check_file"].exists()


# Main
def setup_all() -> None:
    """Set up all third-party tools."""

    THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Setting up third-party tools in {THIRD_PARTY_DIR} ...")

    all_ok = True
    for tool in THIRD_PARTY_TOOLS:
        name = tool["name"]
        print(f"[{name}]")

        if _is_setup(tool):
            print("Already set up, skipping.")
            continue

        ok = _clone(tool["repo"], tool["target_dir"])
        if ok:
            print("Done.")
        else:
            print(f"Failed to set up: \n git clone {tool['repo']} {tool['target_dir']}")
            all_ok = False

    if all_ok:
        print("All third-party tools ready.")
    else:
        print(
            "Some tools failed to set up. Please check the messages above and set up manually."
        )
        sys.exit(1)


if __name__ == "__main__":
    setup_all()
