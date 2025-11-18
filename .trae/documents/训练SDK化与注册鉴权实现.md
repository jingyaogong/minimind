# 总体架构
- 服务端统一暴露训练服务与进程管理接口；引入客户端注册与鉴权，训练仅对已注册客户端开放。
- 训练能力保持原子化：按 `train_type`（pretrain/sft/lora/dpo/ppo/grpo/spo）分派脚本与参数，保留现有脚本接口。
- 提供 Python SDK，封装注册、发起训练、查询进程与日志等操作，客户端只需少量模型参数即可启动训练。
- Web 端保留现有操作界面，新增注册流程与鉴权头注入。

# 服务端改造（Flask）
- 新增注册接口：`POST /api/register`，入参：`name`、`email`；生成并返回 `client_id` 与 `api_key`，并在服务端保存（`trainer_web/clients.json`）。
- 新增鉴权中间件：校验请求头 `Authorization: Bearer <api_key>`；解析出 `client_id` 并附加到请求上下文。
- 训练相关接口全部鉴权并按 `client_id` 隔离：
  - 保留并保护现有接口：`POST /train`（trainer_web/train_web_ui.py:247-266）、`GET /processes`（trainer_web/train_web_ui.py:267-286）、`GET /logs/<process_id>`（trainer_web/train_web_ui.py:288-374）、`POST /stop/<process_id>`（trainer_web/train_web_ui.py:466-485）、`POST /delete/<process_id>`（trainer_web/train_web_ui.py:487-520）、`GET /logfiles`、`GET /logfile-content/<filename>`、`DELETE /delete-logfile/<filename>`。
  - `training_processes` 结构增加 `client_id` 字段；`training_processes.json` 持久化同步该字段（trainer_web/train_web_ui.py:540-607）。
  - 在读取日志、停止/删除进程时先校验归属（仅允许操作本客户端的进程与日志）。
- 抽象原子训练分发器：新增 `dispatcher` 模块，统一 `train_type → 脚本路径 + 参数构造`，内部直接复用现有命令拼装逻辑（trainer_web/train_web_ui.py:59-175）。脚本与参数名均保持不变。
- 安全与配置：
  - `api_key` 服务端按 `sha256` 存储，避免明文；可选 `SERVER_SECRET` 用于签发/验证简易令牌。
  - 新增布尔配置 `ALLOW_ANONYMOUS=false`，默认禁止匿名训练。

# 客户端 SDK（Python）
- 包名：`minimind_sdk`；类：`MinimindClient(base_url, api_key=None)`。
- 方法：
  - `register(name, email) -> {client_id, api_key}`
  - `start_training(train_type, **params) -> process_id`（保留与 `/train` 相同字段约定）
  - `get_processes() -> list`、`get_logs(process_id) -> str`
  - `stop(process_id)`、`delete(process_id)`、`get_logfiles()`、`get_logfile_content(filename)`、`delete_logfile(filename)`
- 简化封装：按训练类型提供高层方法：`train_sft(...)`、`train_dpo(...)`、`train_ppo(...)` 等，只暴露核心参数，其余按当前脚本默认值填充（例如 SFT 参考 trainer/train_full_sft.py:83-105）。
- 错误处理：统一抛出带有服务端错误信息的异常；返回类型标准化。

# Web 端适配
- 新增“注册”入口：表单提交到 `POST /api/register`，成功后将 `api_key` 保存到 `localStorage`。
- `apiClient.js` 在所有请求中自动注入 `Authorization` 头（trainer_web/static/js/services/apiClient.js:1-73）。
- 进程与日志列表仅展示当前客户端数据；保留现有 Tab 与交互（trainer_web/static/js/app.js:1-24、train/form.js:1-127）。

# 接口保留与兼容
- 现有 REST 路径与请求体字段保持不变；仅增加注册接口与鉴权要求。
- 训练脚本 CLI 接口完全保留（`argparse` 参数不改动），通过分发器集中调用。
- 对历史脚本/工具的兼容：如果需要，可保留一个仅本机可用的“匿名模式”开关用于调试（默认关闭）。

# 数据与安全
- 客户注册数据文件：`trainer_web/clients.json`；字段：`client_id`、`name`、`email`、`api_key_hash`、`created_at`。
- 进程记录增加 `client_id`，日志操作前进行归属校验；多租户隔离。
- 不在日志中打印敏感信息；不落盘明文 `api_key`。

# 交付与变更点
- 服务端：`train_web_ui.py` 引入鉴权与 `client_id` 归属；新增 `auth.py` 与 `dispatcher.py`。
- SDK：`minimind_sdk` 源码与示例；示例脚本：注册→发起 SFT 训练→轮询进程→查看日志→停止。
- Web：新增注册 UI、`apiClient.js` 注入鉴权、列表隔离逻辑。

# 验收与测试
- 未注册请求调用 `/train` 返回 `401`；注册后可正常发起训练。
- 用 SDK 完成一次 SFT 训练，能看到进程、日志、停止与删除操作均正常并隔离。
- 多客户端并发训练与日志隔离验证；重启服务后通过 `training_processes.json` 成功恢复状态。
