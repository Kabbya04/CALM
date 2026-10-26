"""One-shot dump of CALM/results for report.md. Not a paper runner."""

import json
import os
from collections import defaultdict
from statistics import mean, stdev

RES = r"E:\Academics\CALM\CALM\results"


def load_all():
    rows = []
    for name in sorted(os.listdir(RES)):
        if not name.endswith(".json") or name == "queue_status.json":
            continue
        path = os.path.join(RES, name)
        with open(path, encoding="utf-8") as handle:
            rec = json.load(handle)
        rec["_file"] = name
        rows.append(rec)
    return rows


def fmt(values, places=2):
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.{places}f}"
    return f"{mean(values):.{places}f}±{stdev(values):.{places}f}"


def seeds_line(values, places=2):
    if not values:
        return "—"
    parts = [f"{v:.{places}f}" for v in values]
    return " / ".join(parts)


def main():
    rows = load_all()
    print(f"JSON_COUNT {len(rows)}")
    print("\n=== FILES ===")
    for rec in rows:
        print(rec["_file"])

    groups = defaultdict(list)
    for rec in rows:
        groups[rec["_file"].rsplit("_seed", 1)[0]].append(rec)

    print("\n=== GROUPED ===")
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda r: r.get("seed", 0))
        print(f"\n## {key}  n={len(items)}")
        for rec in items:
            keep = {
                k: rec[k]
                for k in rec
                if k
                in {
                    "variant",
                    "dataset",
                    "protocol",
                    "seed",
                    "avg_acc",
                    "bwt",
                    "mean_accuracy",
                    "ci95",
                    "accuracy",
                    "labels_used",
                    "labels_per_query",
                    "queries",
                    "rho",
                    "error_rate",
                    "k_shot",
                    "age_mode",
                    "flat_memory",
                    "use_disagreement",
                }
            }
            print(f"  seed={rec.get('seed')} {keep}")
            if "memory" in rec:
                print(f"    memory={rec['memory']}")
            if "curve" in rec:
                for point in rec["curve"]:
                    print(
                        f"    rho={point['rho']}: cov={point['coverage']:.4f} "
                        f"def={point['deferral_rate']:.4f} "
                        f"dep_acc={point['deployed_accuracy']:.4f} "
                        f"dep_err={point['deployed_error']:.4f} "
                        f"model={point['model_accuracy']:.4f} "
                        f"fpr={point['fpr']:.4f}"
                    )

        numeric_keys = [
            "avg_acc",
            "bwt",
            "mean_accuracy",
            "ci95",
            "accuracy",
            "labels_used",
            "labels_per_query",
        ]
        print("  MEAN±STD:")
        for field in numeric_keys:
            values = [float(r[field]) for r in items if field in r]
            if values:
                print(f"    {field}: {fmt(values)}   seeds={seeds_line(values)}")

        if items and "curve" in items[0]:
            print("  CURVE MEAN±STD:")
            rhos = [p["rho"] for p in items[0]["curve"]]
            for rho in rhos:
                for field in (
                    "coverage",
                    "deferral_rate",
                    "deployed_accuracy",
                    "deployed_error",
                    "model_accuracy",
                    "fpr",
                ):
                    values = []
                    for rec in items:
                        for point in rec["curve"]:
                            if abs(point["rho"] - rho) < 1e-9:
                                values.append(float(point[field]))
                    print(f"    rho={rho} {field}: {fmt(values, 4)}  seeds={seeds_line(values, 4)}")

        if items and "memory" in items[0]:
            print("  MEMORY MEAN:")
            for field in ("stored_bytes", "examples", "episode"):
                values = [float(r["memory"][field]) for r in items if field in r.get("memory", {})]
                if values:
                    print(f"    {field}: {fmt(values, 0)}")
            for tier in ("recent", "medium", "old"):
                values = [
                    float(r["memory"]["tier_counts"][tier])
                    for r in items
                    if "tier_counts" in r.get("memory", {})
                ]
                if values:
                    print(f"    tier.{tier}: {fmt(values, 1)}  seeds={seeds_line(values, 0)}")


if __name__ == "__main__":
    main()
