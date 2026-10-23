# 3GPP 提案阅读助手

读取本地 TDoc 元数据和 DOCX 原文，再调用 Qwen2.5-3B-Instruct 生成带来源编号的中文回答。MiniMind 的模型实现和训练代码是另一条独立路径。

## 启动

在项目目录下运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_web.py
```

默认选中“只查原文，不调用模型”。点击“比较两篇 XR 提案”，再点击检索按钮，可查看文件、段落位置和原文片段；取消“只查原文”后才加载模型生成回答。

现有虚拟环境已含所需依赖。新环境参考 `requirements-3gpp.txt`，按硬件安装对应的 PyTorch CUDA 版本。检索不需要下载模型。

默认模型位置是当前用户目录下的 `.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct`。也可以在启动前指定：

```powershell
$env:THREEGPP_MODEL_DIR = 'D:\models\Qwen2.5-3B-Instruct'
$env:THREEGPP_LORA_DIR = 'D:\models\ran2_adapter'
```

仅从本地加载模型，不自动下载。默认禁用现有 LoRA，网页可勾选以比较。本轮少量实测中，现有适配器出现了引用缺失和无依据的流程描述，不据此判断总体优劣。其训练数据质量和专业效果仍需独立核实。

## 已改进的行为

- 按编号检索时保留所有请求的提案，最多 6 篇。比较两篇时不再只保留第一篇。
- 无编号时按标题、提交单位和会议元数据进行 BM25 词项排序，并支持少量明确列出的中英文查询别名。这不是向量检索，也不是整个原文库的全文检索。
- 在候选 DOCX 正文中选择相关片段，每篇最多两个片段，每片段最多 900 字符。每个片段有来源编号、文件路径和 XML 段落范围。段落编号含空段和表格内段落，不是 Word 页码。
- 提取 Word 可见文本，规范化空白。未覆盖图片、公式对象、页眉页脚、脚注和外部附件。所选片段不一定足以概括整篇提案。
- 不再把训练集的助手回答当作参考原文。元数据、旧版 `.doc` 或读取失败都不算正文证据。
- 任何明确询问的提案缺少可读原文时，展示缺失情况并停止生成内容分析；已找到的其他片段仍可查看。补充材料会标为未经核实，目前不能替代缺失的原文。
- 要求区分提案主张、推断与资料不足，要求标注 `[S1]` 等来源。程序检测不存在的来源编号和遗漏引用，未判断引用是否真正支持结论。
- 去掉“协议合规率”和旧综合分的准确率暗示。网页示例改为实际存在且能读取的 XR、NTN 文档，不预先编造提案内容。
- 首次需要生成时才加载模型；支持 CUDA 4-bit、CPU 全精度回退、串行生成、异常传递和生成超时。CPU 路径未在本轮执行，其内存和耗时可能很大。

生成超时从完成模型加载后开始计算，默认 120 秒。程序能请求停止并结束等待，但不能强制中断卡住的 CUDA 操作；后台未结束时继续拒绝新请求。输入预算最多 4096 token，超过时提示缩短材料，不会静默截去第二篇提案；最多生成 2048 token。

## 代码入口

| 文件 | 作用 |
| --- | --- |
| `agent_core.py` | 保持已有导入名称的入口 |
| `rag_evidence.py` | 编号识别、BM25、DOCX 提取、来源记录 |
| `agent_runtime.py` | 模型加载、流式生成、证据不足处理、引用格式检查 |
| `app_web.py` | Streamlit 页面及 Markdown/JSON 导出 |
| `server_demo.py` | 本地 FastAPI 接口 |
| `tests/test_3gpp_agent.py` | 行为回归及生成异常测试 |

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe eval_agent_retrieval.py
.\.venv\Scripts\python.exe scripts/check_agent_interfaces.py
.\.venv\Scripts\python.exe scripts/check_3gpp_generation.py --compare --max-tokens 384
```

检索检查在 8 个固定本地案例上比较本轮备份的旧检索器与当前代码，检查原文覆盖及资料不足时是否停止生成。它不是独立抽样的质量评测，不能用通过比例宣传专业准确率。报告保存至 `output/agent_retrieval_regression.json`。

真实生成检查保存至 `output/agent_generation_smoke.json`，保留问题、来源、输出、引用检查和耗时。生成成功和引用格式通过都不代表事实正确，需要逐条对照原文才能比较原模型和 LoRA 的实际质量。

网页的首段文本时间从收到问题开始，包含检索和可能发生的模型加载，不是纯模型首 token 延迟。冷启动和热启动耗时不能直接混作性能提升。

## HTTP 接口

```powershell
.\.venv\Scripts\python.exe server_demo.py
```

- `GET /health`：服务状态及模型是否已加载。
- `POST /v1/proposals/analyze`：SSE 返回提案检索、来源、生成内容、完成或错误事件。请求示例：`{"question":"对比 R2-2408133 和 R2-2408134","retrieval_only":true}`。
- `POST /v1/chat/completions`：基础聊天兼容接口，支持 `stream=true/false`。此接口不检索原文；文档分析使用上一个接口。未实现完整 OpenAI 功能，如工具调用、usage、鉴权、finish_reason 的精细区分。

默认只监听 `127.0.0.1:8000`。

## 评测边界

后续应固定一组具有完整原文的问题，由懂协议的人标注所需段落和可验证结论。在相同来源及生成预算下，比较“原模型＋检索”和“LoRA＋检索”的事实错误数、原文支持情况、资料不足识别及耗时。现有字符相似度、术语数量和结构词得分不承担这项判断。

本轮修改前的三个入口文件保存在 `output/agent_upgrade_backup_20261021/`。本轮未修改已有模型权重、训练数据或 MiniMind 代码。
