def create_latex_table(final_results, output_dir):
    """Create LaTeX table from benchmark results"""

    # Extract and organize data
    table_data = []

    for result in final_results["results"]:
        if result.get("results") is None:
            continue

        config = result["config"]

        # Extract main metrics from each task
        row = {
            "bits": config["bits"],
            "group_size": config["group_size"],
            "model_path": config["quantized_path"],
        }

        # Add metrics from each task
        for task_name, task_results in result["results"].items():
            for metric_name, value in task_results.items():
                if isinstance(value, (int, float)):
                    # Use common metric names for tasks
                    if "acc" in metric_name.lower() or "accuracy" in metric_name.lower():
                        row[task_name] = value
                    elif metric_name in ["exact_match", "em"]:
                        row[task_name] = value
                    elif "norm" in metric_name.lower() and task_name == "gsm8k":
                        row[task_name] = value

        table_data.append(row)

    if not table_data:
        print("No data available for LaTeX table generation")
        return

    # Sort by bits and group_size
    table_data.sort(key=lambda x: (x["bits"], x["group_size"]))

    # Extract model name from summary
    model_name = final_results["summary"]["base_model"].split("/")[-1]
    lora_rank = final_results["summary"]["lora_config"]["r"]

    # Define task columns (adjust based on your actual tasks)
    task_columns = []
    sample_row = table_data[0]

    # Common evaluation tasks mapping
    task_mapping = {
        "arc_challenge": "ARC-c",
        "arc_easy": "ARC-e",
        "boolq": "BoolQ",
        "hellaswag": "HellaSwag",
        "openbookqa": "OpenBookQA",
        "piqa": "PIQA",
        "winogrande": "Winogrande",
        "gsm8k": "GSM8K",
        "mmlu": "MMLU",
    }

    for task in sample_row.keys():
        if task not in ["bits", "group_size", "model_path"]:
            latex_name = task_mapping.get(task, task.upper())
            task_columns.append((task, latex_name))

    # Generate LaTeX table
    latex_content = []

    # Table header
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\tiny")
    latex_content.append("\\setlength{\\tabcolsep}{4pt} % Adjust column spacing for readability")
    latex_content.append(
        f"\\caption{{Performance evaluation of Daniel initialization with rank $r={lora_rank}$ by varying quantization levels of the residual matrix ($\\boldsymbol{{W}}_r$). Results show the trade-off between compression and performance.}}"
    )
    latex_content.append(f"\\label{{tab:daniel_rank_{lora_rank}_quant_tradeoff}}")
    latex_content.append("\\hspace*{-1cm}")

    # Column specification
    num_cols = 3 + len(task_columns)  # Model, Rank, Bits + task columns
    col_spec = "l" + "c" * (num_cols - 1)
    latex_content.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("\\toprule")

    # Header row
    header = ["\\textbf{Model}", "\\textbf{Rank ($r$)}", "\\textbf{Bits ($\\boldsymbol{W}_r$)}"]
    header.extend([f"\\textbf{{{name}}}" for _, name in task_columns])
    latex_content.append(" & ".join(header) + " \\\\")
    latex_content.append("\\midrule")

    # Data rows
    latex_content.append("")
    latex_content.append(f"% {model_name} Block")

    num_rows = len(table_data)
    latex_content.append(f"\\multirow{{{num_rows}}}{{*}}{{\\shortstack{{{model_name} \\\\ {lora_rank}}}}}")

    for i, row in enumerate(table_data):
        row_data = []

        if i == 0:
            row_data.append("")  # Already handled by multirow
        else:
            row_data.append("")

        row_data.append(f"{lora_rank}")  # Rank

        # Bits formatting
        if row["bits"] == 16:
            bits_str = "16 (FP16)"
        else:
            bits_str = f"{row['bits']}-bit"
        row_data.append(bits_str)

        # Task results
        for task, _ in task_columns:
            if task in row:
                value = row[task]
                if isinstance(value, float):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                row_data.append(formatted_value)
            else:
                row_data.append("--")

        latex_content.append(" & ".join(row_data) + " \\\\")

        # Add cmidrule between different bit configurations if needed
        if i < len(table_data) - 1 and row["group_size"] != table_data[i + 1]["group_size"]:
            latex_content.append("\\cmidrule{2-" + str(num_cols) + "}")

    # Table footer
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")

    # Write to file
    latex_file = os.path.join(output_dir, "benchmark_results_table.tex")
    with open(latex_file, "w") as f:
        f.write("\n".join(latex_content))

    print(f"📄 LaTeX table saved to: {latex_file}")

    # Also print to console for easy copying
    print(f"\n{'=' * 80}")
    print("📋 LATEX TABLE (ready to copy):")
    print(f"{'=' * 80}")
    print("\n".join(latex_content))
    print(f"{'=' * 80}")


def create_multi_rank_latex_table(results_files, output_dir):
    """Create comprehensive LaTeX table from multiple rank results"""

    all_data = []
    models_ranks = {}

    # Load all result files
    for results_file in results_files:
        with open(results_file, "r") as f:
            data = json.load(f)

        model_name = data["summary"]["base_model"].split("/")[-1]
        rank = data["summary"]["lora_config"]["r"]

        if model_name not in models_ranks:
            models_ranks[model_name] = []
        models_ranks[model_name].append(rank)

        # Process results
        for result in data["results"]:
            if result.get("results") is None:
                continue

            config = result["config"]

            row = {
                "model": model_name,
                "rank": rank,
                "bits": config["bits"],
                "group_size": config["group_size"],
            }

            # Extract task results
            for task_name, task_results in result["results"].items():
                for metric_name, value in task_results.items():
                    if isinstance(value, (int, float)) and any(
                        x in metric_name.lower() for x in ["acc", "exact_match", "em", "norm"]
                    ):
                        row[task_name] = value
                        break

            all_data.append(row)

    # Sort data
    all_data.sort(key=lambda x: (x["model"], x["rank"], x["bits"], x["group_size"]))

    # Generate LaTeX
    latex_content = []

    # Table setup
    latex_content.extend(
        [
            "\\begin{table}[htbp]",
            "\\tiny",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\caption{Comprehensive performance evaluation of Daniel initialization by varying adaptation rank ($r$) and quantization level of the residual matrix ($\\boldsymbol{W}_r$). The experiment maps the trade-off between information preserved in adapters and compression of the residual.}",
            "\\label{tab:daniel_comprehensive_rank_quant_tradeoff}",
            "\\hspace*{-1cm}",
        ]
    )

    # Extract task columns
    task_mapping = {
        "arc_challenge": "ARC-c",
        "arc_easy": "ARC-e",
        "boolq": "BoolQ",
        "hellaswag": "HellaSwag",
        "openbookqa": "OpenBookQA",
        "piqa": "PIQA",
        "winogrande": "Winogrande",
        "gsm8k": "GSM8K",
        "mmlu": "MMLU",
    }

    sample_row = all_data[0] if all_data else {}
    task_columns = []
    for task in sample_row.keys():
        if task not in ["model", "rank", "bits", "group_size"]:
            latex_name = task_mapping.get(task, task.upper())
            task_columns.append((task, latex_name))

    # Table header
    num_cols = 3 + len(task_columns)
    col_spec = "l" + "c" * (num_cols - 1)
    latex_content.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("\\toprule")

    header = ["\\textbf{Model}", "\\textbf{Rank ($r$)}", "\\textbf{Bits ($\\boldsymbol{W}_r$)}"]
    header.extend([f"\\textbf{{{name}}}" for _, name in task_columns])
    latex_content.append(" & ".join(header) + " \\\\")
    latex_content.append("\\midrule")
    latex_content.append("")

    # Generate data rows grouped by model
    current_model = None
    current_rank = None

    for i, row in enumerate(all_data):
        # Model block header
        if row["model"] != current_model:
            if current_model is not None:
                latex_content.append("\\midrule")
                latex_content.append("")

            current_model = row["model"]
            current_rank = None

            # Count rows for this model
            model_rows = len([r for r in all_data if r["model"] == current_model])
            ranks_for_model = sorted(set(r["rank"] for r in all_data if r["model"] == current_model))

            latex_content.append(f"% {current_model} Block")
            latex_content.append(f"\\multirow{{{model_rows}}}{{*}}{{\\shortstack{{{current_model}}}}}")

        # Rank subgroup
        if row["rank"] != current_rank:
            if current_rank is not None:
                latex_content.append("\\cmidrule{2-" + str(num_cols) + "}")

            current_rank = row["rank"]
            rank_rows = len([r for r in all_data if r["model"] == current_model and r["rank"] == current_rank])

            latex_content.append(f"& \\multirow{{{rank_rows}}}{{*}}{{{current_rank}}}")

        # Data row
        row_data = ["", ""]  # Model and rank handled by multirow

        # Bits
        bits_str = "16 (FP16)" if row["bits"] == 16 else f"{row['bits']}-bit"
        row_data.append(bits_str)

        # Task results
        for task, _ in task_columns:
            if task in row:
                value = row[task]
                formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                row_data.append(formatted_value)
            else:
                row_data.append("--")

        latex_content.append(" & ".join(row_data) + " \\\\")

    # Table footer
    latex_content.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    # Save and print
    latex_file = os.path.join(output_dir, "comprehensive_benchmark_table.tex")
    with open(latex_file, "w") as f:
        f.write("\n".join(latex_content))

    print(f"📄 Comprehensive LaTeX table saved to: {latex_file}")
    print(f"\n{'=' * 80}")
    print("📋 COMPREHENSIVE LATEX TABLE (ready to copy):")
    print(f"{'=' * 80}")
    print("\n".join(latex_content))
    print(f"{'=' * 80}")
