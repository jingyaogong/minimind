import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentic.data_analysis_env import AgenticToolEnv, get_agentic_tools


CHANNELS = ["抖音", "微信", "百度", "小红书", "自然流量"]
CATEGORIES = ["美妆", "数码", "服饰", "食品", "家居"]
PROVINCES = ["广东", "浙江", "北京", "上海", "四川", "湖北"]

METRIC_DOCS = [
    {
        "id": "metric_refund_amount",
        "title": "退款金额",
        "text": "退款金额使用 orders 表的 refund_amount 字段求和；按渠道、品类、省份或时间过滤后再聚合。",
    },
    {
        "id": "metric_net_revenue",
        "title": "净收入",
        "text": "净收入 = GMV - 退款金额。GMV 使用 orders.gmv，退款金额使用 orders.refund_amount。",
    },
    {
        "id": "metric_refund_rate",
        "title": "退款率",
        "text": "退款率 = 退款金额 / GMV。按品类或渠道对 refund_amount 与 gmv 分别求和后再相除。",
    },
    {
        "id": "metric_roas",
        "title": "ROAS",
        "text": "ROAS = GMV / 广告花费。GMV 来自 orders 表，广告花费来自 marketing 表的 spend 字段。",
    },
    {
        "id": "metric_new_users",
        "title": "新增用户",
        "text": "新增用户使用 users 表的 new_users 字段求和；活跃用户使用 active_users 字段求和。",
    },
    {
        "id": "schema_orders",
        "title": "orders 表字段",
        "text": "orders 包含 month、quarter、channel、category、province、gmv、refund_amount、orders_count。",
    },
    {
        "id": "schema_marketing",
        "title": "marketing 表字段",
        "text": "marketing 包含 month、quarter、channel、spend、impressions、clicks。",
    },
    {
        "id": "schema_users",
        "title": "users 表字段",
        "text": "users 包含 month、quarter、channel、province、active_users、new_users。",
    },
]


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty csv rows: {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def relpath(path):
    cwd = os.getcwd()
    try:
        return os.path.relpath(path, cwd)
    except ValueError:
        return path


def quarter_of(month):
    return f"Q{(month - 1) // 3 + 1}"


def generate_tables(output_dir, seed):
    rng = random.Random(seed)
    table_dir = os.path.join(output_dir, "tables")
    mkdir(table_dir)

    orders = []
    for month in range(1, 13):
        season = 1.0 + (0.18 if month in {6, 11, 12} else 0.0) + (0.08 if month in {1, 2} else 0.0)
        for channel_idx, channel in enumerate(CHANNELS):
            for category_idx, category in enumerate(CATEGORIES):
                for province_idx, province in enumerate(PROVINCES):
                    base = 42000 + channel_idx * 4600 + category_idx * 3100 + province_idx * 1800
                    gmv = round((base + rng.uniform(-5200, 6200)) * season, 2)
                    refund_rate = 0.025 + 0.006 * category_idx + 0.003 * channel_idx + rng.uniform(0.0, 0.018)
                    refund_amount = round(gmv * refund_rate, 2)
                    orders_count = max(1, int(gmv / (90 + category_idx * 12 + rng.uniform(-8, 8))))
                    orders.append(
                        {
                            "month": month,
                            "quarter": quarter_of(month),
                            "channel": channel,
                            "category": category,
                            "province": province,
                            "gmv": gmv,
                            "refund_amount": refund_amount,
                            "orders_count": orders_count,
                        }
                    )

    marketing = []
    for month in range(1, 13):
        for channel_idx, channel in enumerate(CHANNELS):
            spend = round(18000 + channel_idx * 3500 + month * 420 + rng.uniform(-1800, 2200), 2)
            clicks = max(1, int(spend / (2.2 + channel_idx * 0.25 + rng.uniform(-0.2, 0.25))))
            impressions = clicks * rng.randint(18, 36)
            marketing.append(
                {
                    "month": month,
                    "quarter": quarter_of(month),
                    "channel": channel,
                    "spend": spend,
                    "impressions": impressions,
                    "clicks": clicks,
                }
            )

    users = []
    for month in range(1, 13):
        for channel_idx, channel in enumerate(CHANNELS):
            for province_idx, province in enumerate(PROVINCES):
                active_users = max(100, int(5200 + month * 110 + channel_idx * 520 + province_idx * 260 + rng.uniform(-420, 520)))
                new_users = max(20, int(active_users * (0.08 + channel_idx * 0.008 + rng.uniform(0.0, 0.04))))
                users.append(
                    {
                        "month": month,
                        "quarter": quarter_of(month),
                        "channel": channel,
                        "province": province,
                        "active_users": active_users,
                        "new_users": new_users,
                    }
                )

    paths = {
        "orders": os.path.join(table_dir, "orders.csv"),
        "marketing": os.path.join(table_dir, "marketing.csv"),
        "users": os.path.join(table_dir, "users.csv"),
    }
    write_csv(paths["orders"], orders)
    write_csv(paths["marketing"], marketing)
    write_csv(paths["users"], users)
    return {"orders": orders, "marketing": marketing, "users": users, "paths": paths}


def table_specs(paths):
    return [
        {
            "name": "orders",
            "path": relpath(paths["orders"]),
            "description": "订单聚合表，包含 GMV、退款金额和订单数。",
        },
        {
            "name": "marketing",
            "path": relpath(paths["marketing"]),
            "description": "投放聚合表，包含广告花费、曝光和点击。",
        },
        {
            "name": "users",
            "path": relpath(paths["users"]),
            "description": "用户活跃聚合表，包含活跃用户和新增用户。",
        },
    ]


def sum_field(rows, field, **filters):
    total = 0.0
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            total += float(row[field])
    return round(total, 2)


def top_group(rows, group_key, value_field, filters):
    agg = defaultdict(float)
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            agg[row[group_key]] += float(row[value_field])
    key, value = max(agg.items(), key=lambda item: item[1])
    return key, round(value, 2)


def refund_rate_by_category(rows, month):
    agg = defaultdict(lambda: {"gmv": 0.0, "refund": 0.0})
    for row in rows:
        if row["month"] == month:
            agg[row["category"]]["gmv"] += float(row["gmv"])
            agg[row["category"]]["refund"] += float(row["refund_amount"])
    rates = {
        category: values["refund"] / values["gmv"]
        for category, values in agg.items()
        if values["gmv"] > 0
    }
    category, rate = max(rates.items(), key=lambda item: item[1])
    return category, round(rate, 4)


def monthly_gmv_by_channel(orders, month, channel):
    return sum(float(row["gmv"]) for row in orders if row["month"] == month and row["channel"] == channel)


def marketing_spend(marketing, month, channel):
    return sum(float(row["spend"]) for row in marketing if row["month"] == month and row["channel"] == channel)


def build_tool_call(name, arguments):
    return {"name": name, "arguments": arguments}


def make_sample(sample_id, question, answer, checks, tables, expected_tools, actions, task_type, max_turns=4):
    sample = {
        "id": sample_id,
        "question": question,
        "tables": tables,
        "documents": METRIC_DOCS,
        "tools": ["retriever", "python_executor", "calculator"],
        "expected_tools": expected_tools,
        "answer": answer,
        "checks": checks,
        "max_turns": max_turns,
        "metadata": {"task_type": task_type},
    }
    env = AgenticToolEnv(sample)
    trajectory = []
    for action in actions:
        trajectory.append({"role": "assistant", "tool_call": action})
        result = env.execute(action["name"], action["arguments"])
        trajectory.append({"role": "tool", "name": action["name"], "content": result})
    trajectory.append({"role": "assistant", "content": answer})
    sample["expert_trajectory"] = trajectory
    sample["tool_schemas"] = get_agentic_tools(sample["tools"])
    return sample


def task_top_refund_channel(sample_id, tables, data, rng):
    quarter = rng.choice(["Q1", "Q2", "Q3", "Q4"])
    channel, value = top_group(data["orders"], "channel", "refund_amount", {"quarter": quarter})
    question = f"2024年{quarter}退款金额最高的渠道是什么？请给出退款金额。"
    code = (
        f'orders = pd.read_csv("{tables[0]["path"]}")\n'
        f'df = orders[orders["quarter"] == "{quarter}"]\n'
        'agg = df.groupby("channel")["refund_amount"].sum().sort_values(ascending=False)\n'
        'result = {"channel": agg.index[0], "refund_amount": round(float(agg.iloc[0]), 2)}'
    )
    answer = f"2024年{quarter}退款金额最高的渠道是{channel}，退款金额为{value:.2f}元。"
    return make_sample(
        sample_id,
        question,
        answer,
        {"contains": [channel], "number": value, "tolerance": 0.02},
        tables,
        ["retriever", "python_executor"],
        [
            build_tool_call("retriever", {"query": "退款金额 指标口径 orders refund_amount", "top_k": 2}),
            build_tool_call("python_executor", {"code": code}),
        ],
        "top_refund_channel",
    )


def task_net_revenue(sample_id, tables, data, rng):
    month = rng.randint(1, 12)
    channel = rng.choice(CHANNELS)
    gmv = sum_field(data["orders"], "gmv", month=month, channel=channel)
    refund = sum_field(data["orders"], "refund_amount", month=month, channel=channel)
    value = round(gmv - refund, 2)
    question = f"2024年{month}月{channel}渠道的净收入是多少？"
    code = (
        f'orders = pd.read_csv("{tables[0]["path"]}")\n'
        f'df = orders[(orders["month"] == {month}) & (orders["channel"] == "{channel}")]\n'
        'result = {"net_revenue": round(float(df["gmv"].sum() - df["refund_amount"].sum()), 2)}'
    )
    answer = f"2024年{month}月{channel}渠道的净收入为{value:.2f}元。"
    return make_sample(
        sample_id,
        question,
        answer,
        {"contains": [channel], "number": value, "tolerance": 0.02},
        tables,
        ["retriever", "python_executor"],
        [
            build_tool_call("retriever", {"query": "净收入 GMV 退款金额 计算口径", "top_k": 2}),
            build_tool_call("python_executor", {"code": code}),
        ],
        "net_revenue",
    )


def task_refund_rate_category(sample_id, tables, data, rng):
    month = rng.randint(1, 12)
    category, rate = refund_rate_by_category(data["orders"], month)
    percent = round(rate * 100, 2)
    question = f"2024年{month}月退款率最高的品类是什么？退款率是多少？"
    code = (
        f'orders = pd.read_csv("{tables[0]["path"]}")\n'
        f'df = orders[orders["month"] == {month}]\n'
        'agg = df.groupby("category")[["refund_amount", "gmv"]].sum()\n'
        'agg["refund_rate"] = agg["refund_amount"] / agg["gmv"]\n'
        'top = agg.sort_values("refund_rate", ascending=False).iloc[0]\n'
        'result = {"category": agg.sort_values("refund_rate", ascending=False).index[0], "refund_rate": round(float(top["refund_rate"] * 100), 2)}'
    )
    answer = f"2024年{month}月退款率最高的品类是{category}，退款率为{percent:.2f}%。"
    return make_sample(
        sample_id,
        question,
        answer,
        {"contains": [category], "number": percent, "tolerance": 0.02},
        tables,
        ["retriever", "python_executor"],
        [
            build_tool_call("retriever", {"query": "退款率 指标口径 refund_amount gmv", "top_k": 2}),
            build_tool_call("python_executor", {"code": code}),
        ],
        "refund_rate_category",
    )


def task_roas(sample_id, tables, data, rng):
    month = rng.randint(1, 12)
    channel = rng.choice([c for c in CHANNELS if c != "自然流量"])
    gmv = monthly_gmv_by_channel(data["orders"], month, channel)
    spend = marketing_spend(data["marketing"], month, channel)
    value = round(gmv / spend, 2)
    question = f"2024年{month}月{channel}渠道的ROAS是多少？"
    code = (
        f'orders = pd.read_csv("{tables[0]["path"]}")\n'
        f'marketing = pd.read_csv("{tables[1]["path"]}")\n'
        f'gmv = orders[(orders["month"] == {month}) & (orders["channel"] == "{channel}")]["gmv"].sum()\n'
        f'spend = marketing[(marketing["month"] == {month}) & (marketing["channel"] == "{channel}")]["spend"].sum()\n'
        'result = {"roas": round(float(gmv / spend), 2)}'
    )
    answer = f"2024年{month}月{channel}渠道的ROAS为{value:.2f}。"
    return make_sample(
        sample_id,
        question,
        answer,
        {"contains": [channel], "number": value, "tolerance": 0.02},
        tables,
        ["retriever", "python_executor"],
        [
            build_tool_call("retriever", {"query": "ROAS GMV 广告花费 指标口径", "top_k": 2}),
            build_tool_call("python_executor", {"code": code}),
        ],
        "roas",
    )


def task_new_users(sample_id, tables, data, rng):
    month = rng.randint(1, 12)
    channel = rng.choice(CHANNELS)
    province, value = top_group(data["users"], "province", "new_users", {"month": month, "channel": channel})
    question = f"2024年{month}月{channel}渠道新增用户最多的省份是哪个？新增用户数是多少？"
    code = (
        f'users = pd.read_csv("{tables[2]["path"]}")\n'
        f'df = users[(users["month"] == {month}) & (users["channel"] == "{channel}")]\n'
        'agg = df.groupby("province")["new_users"].sum().sort_values(ascending=False)\n'
        'result = {"province": agg.index[0], "new_users": int(agg.iloc[0])}'
    )
    answer = f"2024年{month}月{channel}渠道新增用户最多的省份是{province}，新增用户数为{int(value)}。"
    return make_sample(
        sample_id,
        question,
        answer,
        {"contains": [province], "number": int(value), "tolerance": 0.01},
        tables,
        ["retriever", "python_executor"],
        [
            build_tool_call("retriever", {"query": "新增用户 users new_users 字段口径", "top_k": 2}),
            build_tool_call("python_executor", {"code": code}),
        ],
        "new_users",
    )


def task_incremental_gmv(sample_id, tables, data, rng):
    month = rng.randint(1, 12)
    channel = rng.choice([c for c in CHANNELS if c != "自然流量"])
    pct = rng.choice([10, 12, 15, 20])
    gmv = monthly_gmv_by_channel(data["orders"], month, channel)
    spend = marketing_spend(data["marketing"], month, channel)
    roas = gmv / spend
    incremental = round(spend * pct / 100 * roas, 2)
    question = f"如果2024年{month}月{channel}渠道广告花费增加{pct}%，且ROAS保持不变，预计GMV增加多少？"
    code = (
        f'orders = pd.read_csv("{tables[0]["path"]}")\n'
        f'marketing = pd.read_csv("{tables[1]["path"]}")\n'
        f'gmv = orders[(orders["month"] == {month}) & (orders["channel"] == "{channel}")]["gmv"].sum()\n'
        f'spend = marketing[(marketing["month"] == {month}) & (marketing["channel"] == "{channel}")]["spend"].sum()\n'
        'result = {"spend": round(float(spend), 2), "roas": round(float(gmv / spend), 4)}'
    )
    expression = f"{spend:.2f} * {pct} / 100 * {roas:.6f}"
    answer = f"预计GMV增加{incremental:.2f}元。"
    return make_sample(
        sample_id,
        question,
        answer,
        {"contains": ["GMV"], "number": incremental, "tolerance": 0.05},
        tables,
        ["retriever", "python_executor", "calculator"],
        [
            build_tool_call("retriever", {"query": "ROAS GMV 广告花费 预算增加 计算口径", "top_k": 2}),
            build_tool_call("python_executor", {"code": code}),
            build_tool_call("calculator", {"expression": expression}),
        ],
        "incremental_gmv",
        max_turns=5,
    )


TASK_BUILDERS = [
    task_top_refund_channel,
    task_net_revenue,
    task_refund_rate_category,
    task_roas,
    task_new_users,
    task_incremental_gmv,
]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_split(name, size, tables, data, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(size):
        task_fn = TASK_BUILDERS[i % len(TASK_BUILDERS)]
        sample_id = f"agentic_{name}_{i:06d}"
        rows.append(task_fn(sample_id, tables, data, rng))
    rng.shuffle(rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build Agentic DataAnalysis tasks")
    parser.add_argument("--output_dir", default="dataset/agentic_data")
    parser.add_argument("--train_size", type=int, default=30000)
    parser.add_argument("--dev_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    mkdir(args.output_dir)
    data = generate_tables(args.output_dir, args.seed)
    tables = table_specs(data["paths"])
    splits = {
        "train": build_split("train", args.train_size, tables, data, args.seed + 1),
        "dev": build_split("dev", args.dev_size, tables, data, args.seed + 2),
        "test": build_split("test", args.test_size, tables, data, args.seed + 3),
    }
    for split, rows in splits.items():
        write_jsonl(os.path.join(args.output_dir, f"{split}.jsonl"), rows)
    manifest = {
        "task": "AgenticDataAnalysis",
        "seed": args.seed,
        "splits": {split: len(rows) for split, rows in splits.items()},
        "tables": tables,
        "tools": [tool["function"]["name"] for tool in get_agentic_tools()],
        "task_types": [fn.__name__.replace("task_", "") for fn in TASK_BUILDERS],
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
