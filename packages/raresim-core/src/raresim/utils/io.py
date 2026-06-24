"""Shared I/O utilities for loading and saving data in the RareSim project."""

import json
from pathlib import Path
from raresim.types.result import MethodResults


def load_json(input_path: Path) -> dict:
    """Load a JSON file from a given path and return the data as a dictionary."""
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, output_path: Path) -> None:
    """Save a dictionary or list as a JSON file to a given path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_results(results: dict[str, MethodResults], path: Path) -> None:
    """
    Save similarity results to a JSON file, organized by method name as a whole.
    """
    save_json(
        {method.replace("/", "_"): mr.to_dict() for method, mr in results.items()},
        path,
    )


def save_individual_results(
    results: dict[str, MethodResults], output_dir: Path
) -> None:
    """Save for each method separately, with method name in the filename."""
    for method_name, method_results in results.items():
        top_k = method_results.config.top_k
        safe_name = method_name.replace("/", "_")
        save_json(method_results.to_dict(), output_dir / f"{safe_name}_top{top_k}.json")
