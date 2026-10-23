import os
import re
import json
import time
import argparse
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from modelscope import snapshot_download

# ==================== 1. 3GPP 协议核心规范与术语词表 ====================
SPEC_REGEX = r'\bTS\s*38\.\d{3}\b|\bTR\s*38\.\d{3}\b|\bTS\s*23\.\d{3}\b'
CORE_RAN_TERMS = [
    # 物理层与信道
    'PDCCH', 'PUSCH', 'PUCCH', 'PRACH', 'DMRS', 'CSI-RS', 'SSB', 'PBCH', 'SRS', 'BWP', 'PT-RS',
    # 控制信令与单元
    'DCI', 'UCI', 'MAC CE', 'RRC', 'RLC', 'PDCP', 'SDAP', 'NAS', 'SIB1', 'SIB',
    # 移动性与度量
    'LTM', 'RSRP', 'RSRQ', 'SINR', 'TA', 'TCI', 'QCL', 'C-RNTI', 'PCI', 'RLF', 'HO',
    # 前沿特性
    'NES', 'RedCap', 'eRedCap', 'XR', 'SBFD', 'CLI', 'Ambient IoT', 'NTN', 'MUSIM', 'SDT'
]

# 6 类核心任务的标准结构要素检测模式
TASK_STRUCTURE_PATTERNS = {
    'proposal_understanding': [
        (r'解决.*问题|问题根因', '解决什么问题'),
        (r'核心方案|总体设计', '核心方案'),
        (r'Trigger|触发条件|触发机制', 'Trigger'),
        (r'Configuration|配置', 'Configuration'),
        (r'UE\s*行为|终端行为', 'UE 行为'),
        (r'Network\s*行为|网络行为|基站行为', 'Network 行为'),
        (r'新增.*修改.*信息|信元|协议修改', '新增或修改的信息'),
        (r'协议流程|信令流程|交互流程', '协议流程'),
        (r'优点|优势|收益', '优点'),
        (r'潜在限制|局限|约束', '潜在限制')
    ],
    'proposal_comparison': [
        (r'核心思路|方案思路', '各方案核心思路'),
        (r'共同点|相同点', '共同点'),
        (r'差异点|不同点', '差异点'),
        (r'冲突点|分歧点', '冲突点'),
        (r'优缺点|各自优缺点', '各自优缺点'),
        (r'已经覆盖|已覆盖', '哪些问题已经覆盖'),
        (r'仍未解决|未决问题|遗留', '哪些问题仍未解决')
    ],
    'gap_to_solution': [
        (r'问题根因|根因分析', '问题根因'),
        (r'总体设计|架构设计', '总体设计'),
        (r'新增[/、]修改.*机制|协议机制', '新增/修改的协议机制'),
        (r'UE\s*行为|终端行为', 'UE 行为'),
        (r'Network\s*行为|网络行为|基站行为', 'Network 行为'),
        (r'Trigger|触发条件', 'Trigger'),
        (r'Configuration|配置信元', 'Configuration'),
        (r'IE|Timer|Procedure|信元|定时器', 'IE / Timer / Procedure'),
        (r'协议流程|交互时序', '协议流程'),
        (r'解决.*Gap|解决.*问题', '解决了哪些 Gap'),
        (r'Trade-off|权衡|代价', 'Trade-off'),
        (r'潜在风险|边界风险', '潜在风险')
    ],
    'gap_analysis': [
        (r'根因|背景|协议限制', '根因与背景'),
        (r'缺陷|瓶颈|Gap', '瓶颈与缺陷'),
        (r'开销|时延|复杂度', '开销与时延影响'),
        (r'兼容|互操作|向后兼容', '兼容性与互操作分析')
    ],
    'candidate_critic': [
        (r'解决.*目标\s*Gap|解决.*问题', '是否真正解决目标Gap'),
        (r'新问题|副作用', '是否引入新问题/副作用'),
        (r'冗余设计|复用', '冗余设计与复用已有机制'),
        (r'信令开销|开销评估', '信令开销评估'),
        (r'复杂度|实现复杂度', '实现复杂度分析'),
        (r'向后兼容|兼容性风险', '向后兼容性风险'),
        (r'极端场景|Corner\s*Case', '极端场景与边界'),
        (r'替代方案|更简洁', '是否存在更简洁替代方案'),
        (r'评审结论|修改建议', '评审结论与修改建议')
    ],
    'remaining_gap': [
        (r'已解决|基线成果|当前阶段', '当前已解决的核心部分'),
        (r'未覆盖|场景盲区', '仍未覆盖的典型场景'),
        (r'极端边界|Corner\s*Case', '极端边界与 Corner Cases'),
        (r'开销|代价|代价分析', '开销存储与时延代价'),
        (r'演进建议|未来工作|后续路线', '演进建议与未来工作方向')
    ]
}

# ==================== 2. 评测打分函数 ====================
def compute_structure_score(task_type, text):
    patterns = TASK_STRUCTURE_PATTERNS.get(task_type, [])
    if not patterns:
        return 100.0, []
    hits = []
    for pat, name in patterns:
        if re.search(pat, text, re.I):
            hits.append(name)
    score = (len(hits) / len(patterns)) * 100.0
    return score, hits

def compute_spec_compliance(text):
    specs = set(re.findall(SPEC_REGEX, text, re.I))
    terms = set()
    for t in CORE_RAN_TERMS:
        if re.search(r'\b' + re.escape(t) + r'\b', text, re.I):
            terms.add(t.upper())
    total_unique_hits = len(specs) + len(terms)
    # 达到 8 个不同标准规范与关键术语即得满分 100
    density_score = min(100.0, total_unique_hits * 12.5)
    return density_score, list(specs), list(terms)

def compute_char_lcs(s1, s2):
    """计算两段文本的最长公共子序列（用于评估专家语义相似度），优化空间复杂度"""
    s1_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', s1)
    s2_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', s2)
    if not s1_clean or not s2_clean:
        return 0.0
    
    # 截取前 800 字符进行快速精确动态规划
    s1_sub = s1_clean[:800]
    s2_sub = s2_clean[:800]
    
    dp = [0] * (len(s2_sub) + 1)
    for c1 in s1_sub:
        new_dp = [0] * (len(s2_sub) + 1)
        for j, c2 in enumerate(s2_sub):
            if c1 == c2:
                new_dp[j + 1] = dp[j] + 1
            else:
                new_dp[j + 1] = max(new_dp[j], dp[j + 1])
        dp = new_dp
    
    lcs_len = dp[-1]
    f1 = (2.0 * lcs_len) / (len(s1_sub) + len(s2_sub)) * 100.0
    return f1

# ==================== 3. 主评测执行流程 ====================
def run_benchmark(model, tokenizer, test_samples, device, desc="模型", batch_size=5, max_new_tokens=1024):
    print(f"\n=======================================================", flush=True)
    print(f"  开始 3GPP-Agent 50 题全量盲测基准评测 (并行提速版, Batch={batch_size}): {desc}", flush=True)
    print(f"=======================================================", flush=True)

    results = []
    task_scores = defaultdict(lambda: {'struct': [], 'spec': [], 'semantic': [], 'composite': []})
    
    total = len(test_samples)
    start_time = time.time()

    # 设置左侧 padding 用于 decoder-only 批量生成
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for b_start in range(0, total, batch_size):
        b_samples = test_samples[b_start : b_start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                s['messages'][:-1],
                tokenize=False,
                add_generation_prompt=True
            )
            for s in b_samples
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        for i, sample in enumerate(b_samples):
            sid = sample['sample_id']
            task_type = sample['task_type']
            topic = sample['topic']
            ref_assistant = sample['messages'][-1]['content']

            gen_ids = outputs[i][input_len:]
            pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # 计算三维指标
            struct_score, struct_hits = compute_structure_score(task_type, pred_text)
            spec_score, spec_hits, term_hits = compute_spec_compliance(pred_text)
            semantic_f1 = compute_char_lcs(pred_text, ref_assistant)

            # 综合得分加权 (40% 结构遵从 + 35% 协议规范 + 25% 专家语义对齐)
            composite = 0.40 * struct_score + 0.35 * spec_score + 0.25 * semantic_f1

            task_scores[task_type]['struct'].append(struct_score)
            task_scores[task_type]['spec'].append(spec_score)
            task_scores[task_type]['semantic'].append(semantic_f1)
            task_scores[task_type]['composite'].append(composite)

            results.append({
                'sample_id': sid,
                'task_type': task_type,
                'topic': topic,
                'struct_score': round(struct_score, 2),
                'spec_score': round(spec_score, 2),
                'semantic_f1': round(semantic_f1, 2),
                'composite_score': round(composite, 2),
                'struct_hits': struct_hits,
                'spec_hits': spec_hits,
                'term_hits': term_hits,
                'pred_len': len(pred_text),
                'generated_response': pred_text
            })

        processed = min(b_start + batch_size, total)
        elapsed = time.time() - start_time
        last_sid = b_samples[-1]['sample_id']
        last_tt = b_samples[-1]['task_type']
        last_comp = results[-1]['composite_score']
        print(f"进度: [{processed:2d}/{total:2d}] | 耗时: {elapsed:.1f}s ({elapsed/processed:.1f}s/题) | 最新: {last_sid} ({last_tt}) -> 综合得分: {last_comp:.1f}", flush=True)

    # 4. 汇总统计报告
    print("\n" + "="*70, flush=True)
    print(f"  3GPP-Agent 50 题盲测基准评测报告 - {desc}", flush=True)
    print("="*70, flush=True)
    print(f"{'任务类型 (Task Type)':<25} | {'样本数':<5} | {'结构遵从':<8} | {'规范术语':<8} | {'语义对齐':<8} | {'综合得分':<8}")
    print("-" * 70)

    all_struct = []
    all_spec = []
    all_semantic = []
    all_composite = []

    summary_by_task = {}
    for tt, scores in task_scores.items():
        avg_st = sum(scores['struct']) / len(scores['struct'])
        avg_sp = sum(scores['spec']) / len(scores['spec'])
        avg_se = sum(scores['semantic']) / len(scores['semantic'])
        avg_cp = sum(scores['composite']) / len(scores['composite'])
        
        all_struct.extend(scores['struct'])
        all_spec.extend(scores['spec'])
        all_semantic.extend(scores['semantic'])
        all_composite.extend(scores['composite'])

        summary_by_task[tt] = {
            'count': len(scores['struct']),
            'struct_avg': round(avg_st, 2),
            'spec_avg': round(avg_sp, 2),
            'semantic_avg': round(avg_se, 2),
            'composite_avg': round(avg_cp, 2)
        }
        print(f"{tt:<25} | {len(scores['struct']):<5} | {avg_st:6.1f}% | {avg_sp:6.1f}% | {avg_se:6.1f}% | {avg_cp:6.1f}%")

    print("-" * 70)
    overall_st = sum(all_struct) / len(all_struct)
    overall_sp = sum(all_spec) / len(all_spec)
    overall_se = sum(all_semantic) / len(all_semantic)
    overall_cp = sum(all_composite) / len(all_composite)
    print(f"{'全量总评 (OVERALL)':<25} | {total:<5} | {overall_st:6.1f}% | {overall_sp:6.1f}% | {overall_se:6.1f}% | {overall_cp:6.1f}%")
    print("="*70 + "\n", flush=True)

    report_payload = {
        'model_description': desc,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': total,
        'overall_summary': {
            'structure_completeness_avg': round(overall_st, 2),
            'spec_compliance_avg': round(overall_sp, 2),
            'semantic_alignment_avg': round(overall_se, 2),
            'composite_score_avg': round(overall_cp, 2)
        },
        'task_summary': summary_by_task,
        'details': results
    }
    return report_payload

def main():
    parser = argparse.ArgumentParser(description="3GPP-Agent 50-sample Benchmark Evaluation")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model identifier")
    parser.add_argument("--lora_dir", type=str, default=None, help="Path to LoRA weights directory")
    parser.add_argument("--test_file", type=str, default="dataset/ran2_benchmark_test_50.jsonl", help="Path to 50 test jsonl")
    parser.add_argument("--output_json", type=str, default="output/benchmark_baseline_base_3b.json", help="Path to output json")
    parser.add_argument("--desc", type=str, default="Qwen2.5-3B-Instruct (Base 未微调)", help="Model description")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples (for dry run)")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for parallel generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens to generate")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Device] 评测设备: {device} ({torch.cuda.get_device_name(0) if 'cuda' in device else 'CPU'})", flush=True)

    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"未找到测试集: {args.test_file}")

    with open(args.test_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        samples = samples[:args.limit]
    print(f"[Dataset] 已加载测试集: {len(samples)} 道题目 ({args.test_file})", flush=True)

    print(f"[Model] 正在加载模型权重: {args.model_id} ...", flush=True)
    local_model_dir = r"C:\Users\ZoOoM\.cache\modelscope\hub\models\Qwen\Qwen2___5-3B-Instruct"
    if os.path.exists(local_model_dir):
        print(f"[Model] 使用本地权重: {local_model_dir}", flush=True)
        model_dir = local_model_dir
    else:
        model_dir = snapshot_download(args.model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit 量化加载以保护 8GB 显存并加速推断
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )

    if args.lora_dir:
        print(f"[LoRA] 正在挂载 LoRA 权重: {args.lora_dir} ...", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_dir)

    model.eval()

    # 运行 Benchmark
    report = run_benchmark(model, tokenizer, samples, device, desc=args.desc, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[Saved] 评测完整 JSON 数据已保存至: {args.output_json}", flush=True)

if __name__ == "__main__":
    main()
