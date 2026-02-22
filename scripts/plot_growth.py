import os
import csv
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="绘制动态生长对照图")
    parser.add_argument("--csv", default="eval/summary_ppl.csv", type=str, help="CSV 路径")
    parser.add_argument("--x", default="weight", type=str, help="x 轴列名")
    parser.add_argument("--y", default="ppl", type=str, help="y 轴列名")
    parser.add_argument("--out", default="eval/plot_ppl.png", type=str, help="输出图片")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    x_vals = []
    y_vals = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.x not in row or args.y not in row:
                continue
            x_vals.append(str(row[args.x]))
            try:
                y_vals.append(float(row[args.y]))
            except Exception:
                y_vals.append(float("nan"))

    if not x_vals:
        raise ValueError("No valid rows found in CSV")

    plt.figure(figsize=(8, 4))
    plt.bar(x_vals, y_vals)
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"[OK] Saved plot to {args.out}")


if __name__ == "__main__":
    main()
