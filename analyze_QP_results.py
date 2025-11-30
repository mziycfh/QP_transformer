#!/usr/bin/env python3
"""
Results Analysis Script for QP Function Approximation Experiments

This script analyzes experimental results by reading result files and creating
comparison tables showing how R-squared varies with different layer (l) and
head (h) configurations for both SoftmaxTransformer and LinearTransformer models.

Usage:
    python analyze_results.py <n> <m>

Example:
    python analyze_results.py 5 3
    python analyze_results.py 10 6
"""

import os
import sys
import re
import pandas as pd
import glob
from pathlib import Path


def parse_result_file(file_path):
    """Parse a result file and extract key information"""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Extract model type
        model_type_match = re.search(r"Model Type: (\w+)", content)
        model_type = model_type_match.group(1) if model_type_match else "Unknown"

        # Extract configuration parameters
        n_match = re.search(r"Problem Dimension \(n\): (\d+)", content)
        m_match = re.search(r"Constraint Number \(m\): (\d+)", content)
        l_match = re.search(r"Transformer Layers: (\d+)", content)
        h_match = re.search(r"Attention Heads: (\d+)", content)

        n = int(n_match.group(1)) if n_match else None
        m = int(m_match.group(1)) if m_match else None
        l = int(l_match.group(1)) if l_match else None
        h = int(h_match.group(1)) if h_match else None

        # Extract R-squared statistics
        r_squared_match = re.search(r"R-squared: ([\d.]+)", content)

        r_squared = float(r_squared_match.group(1)) if r_squared_match else None

        return {
            "model_type": model_type,
            "n": n,
            "m": m,
            "layers": l,
            "heads": h,
            "r_squared": r_squared,
            "file_path": file_path,
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def find_result_files(base_dir, n, m):
    """Find result files in the new organized structure"""
    pattern = f"*_results.txt"
    # Search in the new organized structure: experiments/model/nX_mY/folder/file
    search_path = os.path.join(base_dir, "experiments", "*", f"n{n}_m{m}", "*", pattern)
    files = glob.glob(search_path)
    return files


def find_all_parameter_combinations(base_dir):
    """Find all parameter combinations in the new organized structure"""
    pattern = "*_results.txt"
    experiments_dir = os.path.join(base_dir, "experiments")
    all_files = glob.glob(os.path.join(experiments_dir, "*", "*", "*", pattern))

    combinations = set()
    for file_path in all_files:
        # Extract n and m from path: experiments/model/nX_mY/folder/file
        path_parts = file_path.split(os.sep)
        if len(path_parts) >= 4:
            dimension_part = path_parts[-3]  # nX_mY
            match = re.search(r"n(\d+)_m(\d+)", dimension_part)
            if match:
                n = int(match.group(1))
                m = int(match.group(2))
                combinations.add((n, m))

    return sorted(list(combinations))


def create_comparison_table(results, model_type, n, m, metric="r_squared"):
    """Create a comparison table for a specific model type and metric"""
    # Filter results for this model type
    model_results = [r for r in results if r and r["model_type"] == model_type]

    if not model_results:
        print(f"No results found for {model_type} with n={n}, m={m}")
        return None

    # Create DataFrame
    df = pd.DataFrame(model_results)

    # Fail fast on duplicates for same (layers, heads)
    duplicate_mask = df.duplicated(subset=["layers", "heads"], keep=False)
    if duplicate_mask.any():
        dup_df = df.loc[duplicate_mask, ["layers", "heads", metric, "file_path"]]
        # Group duplicates to show all conflicting files
        conflict_lines = []
        for (l, h), group in dup_df.groupby(["layers", "heads"]):
            files = "\n        - ".join(group["file_path"].tolist())
            values = ", ".join(
                [f"{v:.6f}" for v in group[metric].tolist() if v is not None]
            )
            conflict_lines.append(
                f"(layers={l}, heads={h}) has {len(group)} experiments with R-squared values [{values}]:\n        - {files}"
            )
        conflict_msg = (
            f"Duplicate experiments detected for model {model_type} (n={n}, m={m}) with the same (layers, heads).\n"
            f"Please remove duplicates or keep only one result per configuration.\n\n"
            + "\n\n".join(conflict_lines)
        )
        raise RuntimeError(conflict_msg)

    # Pivot table: layers as rows, heads as columns, specified metric as values
    pivot_table = df.pivot_table(
        values=metric,
        index="layers",
        columns="heads",
        aggfunc="first",
    )

    # Sort by layers and heads
    pivot_table = pivot_table.sort_index()
    pivot_table = pivot_table.sort_index(axis=1)

    return pivot_table


def save_tables_to_txt(tables_dict, n, m, output_dir):
    """Save all tables to a single TXT file"""
    # Create rsquared_results_summary directory if it doesn't exist
    summary_dir = os.path.join(output_dir, "rsquared_results_summary")
    os.makedirs(summary_dir, exist_ok=True)

    filename = f"rsquared_comparison_n{n}_m{m}.txt"
    filepath = os.path.join(summary_dir, filename)

    with open(filepath, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"R-squared Comparison Results (n={n}, m={m})\n")
        f.write("=" * 80 + "\n\n")

        for model_type, table in tables_dict.items():
            f.write(f"{model_type} Results:\n")
            f.write("=" * 50 + "\n")

            if table is not None:
                f.write(f"\nR-SQUARED Results:\n")
                f.write("-" * 30 + "\n")

                # Find best configuration
                best_value = table.max().max()
                best_configs = []
                for layer in table.index:
                    for head in table.columns:
                        if table.loc[layer, head] == best_value:
                            best_configs.append((layer, head))

                f.write(f"Best R-squared: {best_value:.4f}\n")
                f.write(f"Best configurations (layers, heads): {best_configs}\n\n")

                # Write table
                f.write(f"R-squared Table (rows=layers, columns=heads):\n")
                f.write(table.to_string())
                f.write("\n\n")
            else:
                f.write(f"R-squared: No results found\n\n")

            f.write("\n" + "=" * 50 + "\n\n")

    print(f"Saved comparison results to: {filepath}")
    return filepath


def print_table_summary(table, model_type, n, m):
    """Print a summary of the table"""
    print(f"\n{'='*60}")
    print(f"{model_type} Results Summary (n={n}, m={m})")
    print(f"{'='*60}")
    print(f"Table shape: {table.shape}")
    print(f"Layer range: {table.index.min()} - {table.index.max()}")
    print(f"Head range: {table.columns.min()} - {table.columns.max()}")
    print(f"R-squared range: {table.min().min():.4f} - {table.max().max():.4f}")

    # Find best configuration
    max_r_squared = table.max().max()
    best_configs = []
    for layer in table.index:
        for head in table.columns:
            if table.loc[layer, head] == max_r_squared:
                best_configs.append((layer, head))

    print(f"Best R-squared: {max_r_squared:.4f}")
    print(f"Best configurations: {best_configs}")

    print(f"\nTable preview:")
    print(table.head())


def analyze_single_combination(n, m, script_dir):
    """Analyze results for a single n, m combination"""
    print(f"\n{'='*60}")
    print(f"Analyzing results for n={n}, m={m}")
    print(f"{'='*60}")

    # Find all result files
    result_files = find_result_files(script_dir, n, m)

    if not result_files:
        print(f"No result files found for n={n}, m={m}")
        return None

    print(f"Found {len(result_files)} result files")

    # Parse all result files
    results = []
    for file_path in result_files:
        result = parse_result_file(file_path)
        if result:
            results.append(result)

    if not results:
        print("No valid results could be parsed")
        return None

    print(f"Successfully parsed {len(results)} result files")

    # Create comparison tables for each model type
    model_types = ["SoftmaxTransformer", "LinearTransformer"]
    tables_dict = {}

    for model_type in model_types:
        print(f"\nProcessing {model_type}...")
        table = create_comparison_table(results, model_type, n, m, "r_squared")
        tables_dict[model_type] = table

        if table is not None:
            print(f"    Found {table.shape[0]}x{table.shape[1]} configurations")
        else:
            print(f"    No results found for {model_type}")

    # Save all tables to a single TXT file
    txt_file = save_tables_to_txt(tables_dict, n, m, script_dir)

    # Print overall comparison
    print(f"\nOverall Comparison:")
    print("-" * 50)
    for model_type in model_types:
        print(f"\n{model_type}:")
        table = tables_dict[model_type]
        if table is not None:
            best_value = table.max().max()
            best_configs = []
            for layer in table.index:
                for head in table.columns:
                    if table.loc[layer, head] == best_value:
                        best_configs.append((layer, head))
            print(f"  Best R-squared: {best_value:.4f} (configs: {best_configs})")
        else:
            print(f"  No results")

    return txt_file


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check command line arguments
    if len(sys.argv) == 3:
        # Specific n, m provided
        try:
            n = int(sys.argv[1])
            m = int(sys.argv[2])
            combinations = [(n, m)]
        except ValueError:
            print("Error: n and m must be integers")
            sys.exit(1)
    elif len(sys.argv) == 1:
        # No arguments provided, find all combinations
        print("No parameters provided. Searching for all available combinations...")
        combinations = find_all_parameter_combinations(script_dir)

        if not combinations:
            print("No result files found. Make sure you have run experiments.")
            sys.exit(1)

        print(f"Found {len(combinations)} parameter combinations: {combinations}")
    else:
        print("Usage:")
        print(
            "  python analyze_results.py                    # Analyze all available combinations"
        )
        print(
            "  python analyze_results.py <n> <m>            # Analyze specific combination"
        )
        print("Example: python analyze_results.py 5 3")
        sys.exit(1)

    print(f"Searching in directory: {script_dir}")

    # Analyze each combination
    saved_files = []
    for n, m in combinations:
        txt_file = analyze_single_combination(n, m, script_dir)
        if txt_file:
            saved_files.append(txt_file)

    # Final summary
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    print(f"Analyzed {len(combinations)} parameter combinations")
    print(f"Saved {len(saved_files)} result files to rsquared_results_summary/ folder:")
    for file in saved_files:
        print(f"  - {os.path.basename(file)}")

    print(f"\nYou can now:")
    print(f"1. View all detailed results in the rsquared_results_summary/ folder")
    print(f"2. Compare SoftmaxTransformer vs LinearTransformer performance")
    print(f"3. Analyze R-squared trends across different configurations")
    print(f"4. Identify optimal layer and head configurations")


if __name__ == "__main__":
    main()
