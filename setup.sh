#!/usr/bin/env bash
#
# setup.sh — bootstrap RareSim from a fresh clone to a runnable state.
#
# Steps (all required — the package will not run without them):
#   1. Clone third-party tools    (raresim.build.setup_third_party)
#      fast_hpo_cr is a runtime dependency of the extraction methods.
#   2. Download ontology sources  (raresim.build.download_ontologies)
#   3. Build shared artifacts     (raresim.build.build_shared_artifacts)
#
# Usage:
#   ./setup.sh
#

set -euo pipefail

echo "=================================================="
echo "  RareSim — full setup from clone to runnable"
echo "=================================================="

echo ""
echo "[1/3] Setting up third-party tools (required)..."
python -m raresim.build.setup_third_party

echo ""
echo "[2/3] Downloading ontology sources..."
python -m raresim.build.load_ontologies_to_local

echo ""
echo "[3/3] Building shared artifacts..."
python -m raresim.build.build_shared_artifacts

echo ""
echo "=================================================="
echo "  Setup complete. Artifacts in outputs/artifacts/"
echo "  You can now run similarity pipelines."
echo "=================================================="