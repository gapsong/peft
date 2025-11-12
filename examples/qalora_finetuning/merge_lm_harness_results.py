#!/usr/bin/env python
import os
import re
import sys
import json
import argparse
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# ==============================================================================
# Hilfsfunktionen: Namensbereinigung und Parsing
# ==============================================================================
def _clean_method_label(raw: str) -> str:
    """
    Vereinheitlicht die Darstellungsnamen für Methoden.
    """
    mapping = {
        "GPTQ LORA": "GPTQ QLORA",
        "POST QUANTIZATION": "GPTQ POST QUANTIZATION",
        "BASELINE": "INITIAL QUANTIZATION",
        "REFERENCE": "REFERENCE",
        "QALORA": "QALORA",
    }
    return mapping.get(raw, raw)


def _parse_model_name_generic(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Parst Modell-/Ordnernamen der Form:
      - mode_<method>_bits_<b>_rank_<r>_group_<g>[_training_skip_...]
      - initial_quantization_<b>bit
      - SmoLM2-1.7B_Instruct
    und liefert ein Dict mit method/bits/rank/group/training.
    """
    m = re.search(r'mode_(\w+)_bits_(\d+)_rank_(\d+)_group_(\d+)', model_name)
    if m:
        method, bits, rank, group = m.groups()
        training_status = "Untrained" if "training_skip_True" in model_name else "Trained"
        method_label = _clean_method_label(method.replace('_', ' ').upper())
        return {
            "method": method_label,
            "bits": int(bits),
            "rank": int(rank),
            "group": int(group),
            "training": training_status,
        }

    m2 = re.search(r'^initial_quantization_(\d+)bit$', model_name, flags=re.IGNORECASE)
    if m2:
        bits = int(m2.group(1))
        return {
            "method": _clean_method_label("BASELINE"),
            "bits": bits,
            "rank": 0,  # Rank 0 = kein LoRA
            "group": None,
            "training": "Baseline",
        }

    if model_name == "SmoLM2-1.7B_Instruct":
        return {
            "method": _clean_method_label("REFERENCE"),
            "bits": 0,
            "rank": 0,
            "group": None,
            "training": "Reference",
        }

    return None


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


def _pick_metric(metrics: Dict[str, Any], bench_name: str) -> Optional[Tuple[str, float]]:
    """
    Wählt eine geeignete Metrik pro Benchmark:
      - für wikitext: bits_per_byte,none > word_perplexity,none > byte_perplexity,none
      - sonst: acc_norm,none > acc,none > exact_match,strict-match > exact_match,flexible-extract
      - fallback: erster numerischer Wert
    """
    if not isinstance(metrics, dict):
        return None

    numeric = {k: v for k, v in metrics.items() if _is_number(v)}
    if not numeric:
        return None

    bn = (bench_name or "").lower()
    if "wikitext" in bn:
        for k in ["bits_per_byte,none", "word_perplexity,none", "byte_perplexity,none"]:
            if k in numeric and _is_number(numeric[k]):
                return k, float(numeric[k])

    for k in ["acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,flexible-extract"]:
        if k in numeric and _is_number(numeric[k]):
            return k, float(numeric[k])

    k0 = next(iter(numeric))
    return k0, float(numeric[k0])


# ==============================================================================
# TEIL 1: FUNKTION ZUM ZUSAMMENFÜHREN DER ERGEBNISSE
# ==============================================================================
def merge_lm_harness_results_recursive(base_paths: List[str], output_file: str) -> None:
    """
    Durchsucht rekursiv alle Unterverzeichnisse der angegebenen Basis-Pfade nach
    'eval_results.json', extrahiert den 'results'-Block und führt alle Ergebnisse
    in einer einzigen Ausgabedatei zusammen. Der Name des Eltern-Ordners dient
    als Schlüssel (z. B. mode_qalora_bits_2_rank_128_group_32).
    """
    if isinstance(base_paths, (str, os.PathLike)):
        base_paths = [base_paths]  # type: ignore[arg-type]

    merged_data: Dict[str, Dict[str, Any]] = {}

    for base in base_paths:
        expanded = os.path.expanduser(str(base))
        if not os.path.isdir(expanded):
            print(f"Warnung: Überspringe ungültigen Pfad: {expanded}")
            continue

        print(f"Durchsuche rekursiv: {expanded}")
        for dirpath, _, filenames in os.walk(expanded):
            if "eval_results.json" not in filenames:
                continue

            file_path = os.path.join(dirpath, "eval_results.json")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
            except Exception as e:
                print(f"- Fehler beim Lesen {file_path}: {e}")
                continue

            # Häufige Struktur: {"results": {...}} – sonst ganzen Inhalt nehmen
            results = content.get("results", content)
            if not isinstance(results, dict):
                print(f"- Überspringe (keine dict-Results): {file_path}")
                continue

            model_key = os.path.basename(dirpath)
            if not model_key:
                model_key = os.path.basename(os.path.dirname(dirpath))

            # Merge (Benchmarks zusammenführen)
            tgt = merged_data.setdefault(model_key, {})
            for bench, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                tgt.setdefault(bench, {})
                # nicht-destruktiv vereinigen/aktualisieren
                tgt[bench].update(metrics)

    if not merged_data:
        print("Keine Daten gefunden, Ausgabedatei wird nicht erstellt.")
        return

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        print(f"OK: Zusammengeführt -> {output_file} (Modelle: {len(merged_data)})")
    except IOError as e:
        print(f"FEHLER beim Schreiben: {e}")


# ==============================================================================
# TEIL 2: FUNKTION ZUR VISUALISIERUNG
# ==============================================================================
def plot_lm_harness_results(json_path: str, output_pdf_path: str) -> None:
    print(f"\nLese zusammengeführte lm-harness Daten von: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Fehler beim Lesen der Datei: {e}")
        sys.exit(1)

    # Flatten -> eine Zeile pro (model, benchmark)
    rows: List[Dict[str, Any]] = []
    for model_name, benchmarks in (data or {}).items():
        meta = _parse_model_name_generic(model_name)
        if not meta or not isinstance(benchmarks, dict):
            continue

        for bench, metrics in (benchmarks or {}).items():
            sel = _pick_metric(metrics, bench)
            if not sel:
                continue
            metric, value = sel
            rows.append(
                {
                    "model": model_name,
                    "benchmark": bench,
                    "metric": metric,
                    "value": float(value),
                    **meta,
                }
            )

    if not rows:
        print("Keine gültigen Daten zum Plotten gefunden.")
        return

    df = pd.DataFrame(rows)

    # Rank als geordnete Kategorie
    rank_order = sorted(df["rank"].unique().tolist())
    df["rank"] = pd.Categorical(df["rank"], categories=rank_order, ordered=True)
    df = df.sort_values(by=["training", "bits", "method", "rank"])

    # Farben pro Methode, Marker pro Bit
    method_order = sorted(df["method"].unique().tolist())
    palette_colors = sns.color_palette("tab10", n_colors=len(method_order))
    method_palette = {m: c for m, c in zip(method_order, palette_colors)}

    bits_all = sorted(df["bits"].unique().tolist())
    marker_map = {2: "o", 3: "X"}  # 2bit = Kreis, 3bit = X
    fallback_markers = ["s", "D", "^", "v", "P", "*", "h", "8", "<", ">"]
    fm_idx = 0
    for b in bits_all:
        if b not in marker_map:
            marker_map[b] = fallback_markers[fm_idx % len(fallback_markers)]
            fm_idx += 1

    print("\nVorschau Daten:")
    print(df.head())

    sns.set_theme(style="whitegrid")
    with PdfPages(output_pdf_path) as pdf:
        for bench in df["benchmark"].unique():
            df_bm = df[df["benchmark"] == bench].dropna(subset=["value"])
            if df_bm.empty:
                continue

            metric_name = df_bm["metric"].iloc[0]
            lower_is_better = any(x in metric_name for x in ["perplexity", "bits_per_byte"])
            higher_is_better = not lower_is_better

            # Baseline-/Referenz-Werte (keine Ranks)
            base_map = (
                df_bm[df_bm["method"] == "INITIAL QUANTIZATION"]
                .groupby("bits")["value"]
                .first()
                .to_dict()
            )
            ref_val = (
                df_bm[df_bm["method"] == "REFERENCE"]["value"].dropna().iloc[0]
                if (df_bm["method"] == "REFERENCE").any()
                else None
            )

            # Nur trainierte LoRA-Kurven zeichnen
            lo_methods = [m for m in method_order if m not in ["INITIAL QUANTIZATION", "REFERENCE"]]
            df_lines = df_bm[(df_bm["method"].isin(lo_methods)) & (df_bm["training"] == "Trained")]

            if df_lines.empty and not base_map and ref_val is None:
                print(f"  - Überspringe '{bench}' (keine Daten).")
                continue

            bits_order_lora = sorted(df_lines["bits"].unique().tolist()) if not df_lines.empty else sorted(base_map.keys())

            # Marker-Liste passend zur style_order
            markers_list = [marker_map[b] for b in bits_order_lora] if bits_order_lora else True

            g = sns.relplot(
                data=df_lines,
                kind="line",
                x="rank",
                y="value",
                hue="method",
                hue_order=lo_methods,
                palette={m: method_palette[m] for m in lo_methods} if lo_methods else None,
                style="bits",
                style_order=bits_order_lora if bits_order_lora else None,
                markers=markers_list,
                dashes=False,
                markersize=8,
                height=6,
                aspect=2.6,
            )

            ax = g.ax

            # Baselines/Referenz als horizontale Linien einzeichnen
            bits_line_styles = {2: "--", 3: ":"}  # pro Bit ein Linienstil
            for b in bits_order_lora:
                if b not in bits_line_styles:
                    bits_line_styles[b] = (0, (5, 2))

            for b, y in base_map.items():
                ax.axhline(
                    y=y,
                    color="black",
                    linestyle=bits_line_styles.get(b, "--"),
                    linewidth=1.5,
                    alpha=0.9,
                )
            if ref_val is not None:
                ax.axhline(y=ref_val, color="grey", linestyle="-.", linewidth=1.3, alpha=0.9)

            # Titel/Labels
            # Reserve space on the right for legends and center the title
            g.fig.subplots_adjust(left=0.07, right=0.70, top=0.90, bottom=0.12)
            g.fig.suptitle(f"{bench}", x=0.5, ha="center", fontsize=16, weight="bold")
            ax.set_xlabel("LoRA Rank", fontsize=12)
            ax.set_ylabel(
                f"Score ({'höher ist besser' if higher_is_better else 'niedriger ist besser'})",
                fontsize=12,
            )

            # Legenden aufbauen
            if g._legend is not None:
                g._legend.remove()

            method_handles = [
                Line2D(
                    [0],
                    [0],
                    color=method_palette[m],
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    markersize=7,
                    label=m,
                )
                for m in lo_methods
            ]
            bit_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker=marker_map[b],
                    linestyle="None",
                    markersize=8,
                    label=f"{b}-bit (LoRA)",
                )
                for b in bits_order_lora
            ]
            base_handles: List[Line2D] = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=bits_line_styles[b],
                    linewidth=1.5,
                    label=f"Baseline {b}-bit",
                )
                for b in sorted(base_map.keys())
            ]
            if ref_val is not None:
                base_handles.append(Line2D([0], [0], color="grey", linestyle="-.", linewidth=1.3, label="Reference"))

            # Figure-level legends placed within the reserved right margin (no clipping)
            legend_kw = dict(frameon=False, borderaxespad=0.0, handlelength=2.0,
                             handletextpad=0.6, labelspacing=0.4, fontsize=9, title_fontsize=9)
            # X positions in figure coords inside the reserved area
            x_leg = 0.74
            if method_handles:
                g.fig.legend(handles=method_handles,
                             labels=[h.get_label() for h in method_handles],
                             title="Method (color)", loc="upper left",
                             bbox_to_anchor=(x_leg, 0.92), **legend_kw)
            if bit_handles:
                g.fig.legend(handles=bit_handles,
                             labels=[h.get_label() for h in bit_handles],
                             title="Bits (marker)", loc="upper left",
                             bbox_to_anchor=(x_leg, 0.62), **legend_kw)
            if base_handles:
                g.fig.legend(handles=base_handles,
                             labels=[h.get_label() for h in base_handles],
                             title="Baseline (lines)", loc="upper left",
                             bbox_to_anchor=(x_leg, 0.32), **legend_kw)

            # Save full figure (avoid tight cropping to keep legends)
            pdf.savefig(g.fig)
            plt.close(g.fig)
            print(f"  - Plot '{bench}' hinzugefügt")

    print(f"Fertig: PDF gespeichert unter {output_pdf_path}")


# ==============================================================================
# TEIL 2.5: EXPORT ALS "WIDE"-CSV (bits,method,r16,r128,r256) PRO BENCHMARK
# ==============================================================================
def export_wide_csv(
    json_path: str,
    csv_path: str,
    benches: Optional[List[str]] = None,
    percent: bool = False,
) -> None:
    """
    Erzeugt eine CSV mit Spalten: bits, method, r16, r128, r256

    - Pro angefragtem Benchmark (Standard: alle -> je Bench eine Datei)
    - Baselines/Reference ohne Rank werden auf alle rXX-Spalten repliziert
    - Metrikauswahl identisch zu Plot:
        * wikitext -> bits_per_byte,none (sonst word_perplexity,none)
        * sonst -> acc_norm,none > acc,none > exact_match,strict-match > exact_match,flexible-extract > erster numerischer
    - percent=True: Werte werden mit 100 multipliziert
    """
    print(f"Lese JSON für CSV-Export: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten
    rows: List[Dict[str, Any]] = []
    for model_name, benchmarks in (data or {}).items():
        meta = _parse_model_name_generic(model_name)
        if not meta or not isinstance(benchmarks, dict):
            continue
        for bench, metrics in benchmarks.items():
            sel = _pick_metric(metrics, bench)
            if not sel:
                continue
            metric, value = sel
            rows.append(
                {
                    "model": model_name,
                    "benchmark": bench,
                    "metric": metric,
                    "value": float(value),
                    **meta,
                }
            )

    if not rows:
        print("Keine Daten gefunden.")
        return

    df = pd.DataFrame(rows)
    if percent:
        df["value"] = df["value"] * 100.0

    bench_list = benches or sorted(df["benchmark"].unique().tolist())

    # Ranks, die wir als Spalten wollen
    rank_cols = [16, 128, 256]
    rank_name_map = {16: "r16", 128: "r128", 256: "r256"}

    for bench in bench_list:
        dfb = df[df["benchmark"] == bench].copy()
        if dfb.empty:
            print(f"- {bench}: keine Daten.")
            continue

        # Nur trainierte Kurven + Baselines/Reference
        df_lines = dfb[(dfb["training"] == "Trained")]
        df_base = dfb[dfb["rank"] == 0]  # INITIAL QUANTIZATION / REFERENCE

        # Pivot für Ranks
        wide = (
            df_lines[df_lines["rank"].isin(rank_cols)]
            .pivot_table(index=["bits", "method"], columns="rank", values="value", aggfunc="mean")
            .copy()
        )
        for r in rank_cols:
            if r not in wide.columns:
                wide[r] = np.nan
        wide = wide[rank_cols].reset_index()
        wide.rename(columns=rank_name_map, inplace=True)

        # Baselines als eigene Zeilen, Wert auf alle rXX kopieren
        base_rows: List[Dict[str, Any]] = []
        for _, r in df_base.iterrows():
            val = float(r["value"])
            base_rows.append({"bits": int(r["bits"]), "method": r["method"], "r16": val, "r128": val, "r256": val})

        if base_rows:
            wide = pd.concat([wide, pd.DataFrame(base_rows)], ignore_index=True)

        # sortieren
        wide = wide.sort_values(by=["bits", "method"]).reset_index(drop=True)

        # Ausgabe-Dateiname (bei mehreren Benches -> je Bench eine Datei)
        base, ext = os.path.splitext(csv_path)
        out_file = f"{base}_{bench}.csv" if len(bench_list) > 1 else csv_path

        wide.to_csv(out_file, index=False)
        print(f"- CSV exportiert: {out_file}  [{bench}]")


# ==============================================================================
# TEIL 3: CLI
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Führt lm-harness Ergebnisse zusammen, visualisiert und/oder exportiert sie."
    )
    parser.add_argument(
        "--merge",
        type=str,
        nargs="+",
        metavar="BASE_PATH",
        help="Ein oder mehrere Ordner, die rekursiv nach eval_results.json durchsucht werden.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="merged_lm_harness_results.json",
        help="Ausgabedatei für den Merge-Prozess.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        metavar="JSON_FILE",
        help="Pfad zur zusammengeführten JSON-Datei zum Plotten.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="lm_harness_plots.pdf",
        help="PDF-Ausgabedatei für die Plots.",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        metavar="CSV_FILE",
        help="Exportiert eine Wide-CSV (bits,method,r16,r128,r256). Bei mehreren Benchmarks wird je Benchmark eine Datei erzeugt.",
    )
    parser.add_argument(
        "--csv-from",
        type=str,
        metavar="JSON_FILE",
        help="Quelle für den CSV-Export (Standard: --plot oder -o, falls vorhanden).",
    )
    parser.add_argument(
        "--bench",
        type=str,
        nargs="+",
        help="Liste der Benchmarks für den CSV-Export (Standard: alle).",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="CSV-Werte mit 100 multiplizieren (z. B. für Prozent).",
    )

    args = parser.parse_args()

    if args.merge:
        merge_lm_harness_results_recursive(args.merge, args.output)

    if args.plot:
        plot_lm_harness_results(args.plot, args.plot_output)

    if args.export_csv:
        json_src = args.csv_from or args.plot or args.output
        if not json_src:
            print("Fehlender JSON-Pfad für --export-csv (verwende --csv-from oder --plot oder -o).")
        else:
            export_wide_csv(json_src, args.export_csv, benches=args.bench, percent=args.percent)

    if not args.merge and not args.plot and not args.export_csv:
        print("Keine Aktion angegeben. Bitte --merge, --plot und/oder --export-csv verwenden.")


if __name__ == "__main__":
    main()