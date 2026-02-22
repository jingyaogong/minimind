import os
import json
import argparse
import csv


def collect_json_files(path):
    files = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".json"):
                files.append(os.path.join(path, name))
    elif os.path.isfile(path):
        files.append(path)
    return files


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="汇总 PPL 评测结果为 CSV")
    parser.add_argument("--inputs", required=True, type=str, help="JSON 文件或目录（eval_ppl 输出）")
    parser.add_argument("--out", default="eval/summary_ppl.csv", type=str, help="输出 CSV 路径")
    parser.add_argument("--group_by", default="", type=str, help="按某字段分组（可选）")
    parser.add_argument("--out_grouped", default="", type=str, help="分组汇总 CSV 输出路径（可选）")
    args = parser.parse_args()

    files = collect_json_files(args.inputs)
    if not files:
        raise ValueError(f"No json files found in {args.inputs}")

    rows = []
    for path in files:
        data = load_json(path)
        data["file"] = os.path.basename(path)
        rows.append(data)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # 写 CSV
    keys = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[OK] Saved {len(rows)} rows to {args.out}")

    if args.group_by:
        grouped = {}
        for r in rows:
            key = r.get(args.group_by)
            if key is None:
                continue
            grouped.setdefault(key, []).append(r)

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        out_grouped = args.out_grouped or os.path.splitext(args.out)[0] + f"_grouped_{args.group_by}.csv"
        # 收集数值列
        numeric_cols = set()
        for r in rows:
            for k, v in r.items():
                if isinstance(v, (int, float)) or to_float(v) is not None:
                    numeric_cols.add(k)

        base_cols = sorted(numeric_cols)
        std_cols = [f"{c}_std" for c in base_cols]
        fieldnames = [args.group_by, "count"] + base_cols + std_cols
        with open(out_grouped, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key, rs in grouped.items():
                row = {args.group_by: key, "count": len(rs)}
                for col in numeric_cols:
                    vals = []
                    for r in rs:
                        v = r.get(col)
                        v = to_float(v)
                        if v is not None:
                            vals.append(v)
                    if vals:
                        mean = sum(vals) / len(vals)
                        var = sum((x - mean) ** 2 for x in vals) / len(vals)
                        row[col] = round(mean, 6)
                        row[f"{col}_std"] = round(var ** 0.5, 6)
                writer.writerow(row)
        print(f"[OK] Saved grouped summary to {out_grouped}")


if __name__ == "__main__":
    main()
