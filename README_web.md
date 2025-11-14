## 总览
- 目标：将当前单文件 `static/js/script.js` 的所有逻辑拆分为职责清晰的 ES Modules，并在不改变既有后端接口的前提下，优化交互与视觉表现，使界面更专业、易用。
- 范围：仅前端（HTML/CSS/JS），保持与后端 `train_web_ui.py` 的 REST 接口兼容；可选提供后端小幅辅助（如 SSE）但不作为本次必须项。

## 现状拆解
- 导航与标签：`openTab`、页面切换与进程/日志刷新（script.js:2–29, 23–29）。
- 训练类型联动与默认值：`train_type` 变更、分区显隐与各类型默认填充（script.js:32–154）。
- 轮询与请求：带超时/重试的 `fetchWithTimeoutAndRetry`（script.js:200–248），进程状态轮询 `startProcessPolling`/`checkProcessStatusChanges`（script.js:182–199, 251–279）。
- 进程列表与项更新：`loadProcesses`/`updateProcessItem`/`addProcessItemToGroup`（script.js:467–667, 357–465）。
- SwanLab：链接检测/打开（script.js:282–355）。
- 日志查看：进程日志自动刷新与容器滚动（script.js:669–788），日志文件列表/查看/删除（script.js:1126–1462）。
- 全局通知与确认：`showNotification`/`showConfirmDialog`（script.js:790–885）。
- 训练表单提交：数据整形、模式分支、启动训练后切到“训练进程”（script.js:1052–1124）。

## 拆分方案（ES Modules）
- 采用原生 ES Modules，无需打包器；在 `index.html` 将单一脚本替换为入口 `app.js`（type="module"），其余模块按需 import。
- 目录结构（`static/js/`）：
  - `app.js`：应用入口，初始化标签、轮询、首屏加载。
  - `services/apiClient.js`：`fetchWithTimeoutAndRetry` 与接口封装（`getProcesses`/`getLogs`/`startTrain`/`stop`/`deleteProcess`/`getLogFiles`/`getLogFileContent`/`deleteLogFile`）。
  - `ui/tabs.js`：`openTab` 与标签选中态管理、切换后触发对应加载。
  - `ui/notify.js`：`showNotification`（success/error/info）。
  - `ui/dialog.js`：`showConfirmDialog`/`closeDialog`。
  - `train/form.js`：训练类型显隐与默认值、`submit` 整形与发起训练；合并现有 `index.html` 内联 GPU 选择器联动。
  - `processes/list.js`：`loadProcesses`、分组折叠/展开、`updateProcessItem`、状态 chip 与操作按钮逻辑（含 SwanLab）。
  - `processes/logs.js`：进程日志区域展开/关闭、自动刷新定时器管理（`logTimers`）、`refreshLog`/`loadLogContent`。
  - `logfiles/list.js`：日志文件分组、查看全文、删除项与分组高度更新。
  - `utils/dom.js`：通用 dom/help 方法（`qs`/`qsa`/`el`/`setVisible`/`setText` 等）。
- 现有函数到模块的映射：
  - `openTab` → ui/tabs.js（script.js:2–29）。
  - 类型联动与默认值 → train/form.js（script.js:32–154, 1052–1124）。
  - `fetchWithTimeoutAndRetry` → services/apiClient.js（script.js:200–248）。
  - 轮询相关 → processes/list.js（script.js:182–199, 251–279）。
  - 进程项增/改/组折叠 → processes/list.js（script.js:357–667, 467–602）。
  - SwanLab → processes/list.js + services（script.js:282–355）。
  - 进程日志 → processes/logs.js（script.js:669–788）。
  - 通知/确认 → ui/notify.js, ui/dialog.js（script.js:790–885）。
  - 进程停止/删除 → processes/list.js + services（script.js:887–1050）。
  - 日志文件列表/操作 → logfiles/list.js（script.js:1126–1462）。

## 加载与初始化
- 在 `app.js`：
  - 绑定 tab 切换，首屏激活“开始训练”。
  - 初始化训练表单默认值与 GPU 选择器显隐。
  - 启动进程状态轮询，只在“训练进程”栏激活；离开时清理定时器与日志定时器。
  - 首次进入“训练进程”时触发 `loadProcesses`，进入“日志文件”触发 `loadLogFiles`。

## 交互优化
- 表单
  - 必填项校验与错误提示；失焦即时校验；数值范围与格式（学习率、小数、正整数）。
  - 将“强化学习参数”块 Default 折叠，仅在选择 DPO/PPO 展开；提供字段级 tooltip（问号提示）。
  - 训练命令预览：在提交前显示将要执行的命令（只读文本），便于核对。
  - 记忆最近一次配置（localStorage），一键恢复。
- 进程列表
  - 顶部筛选：按训练类型/状态过滤；关键字搜索（按时间/类型）。
  - 操作区对齐与悬停提示；SwanLab 按钮仅在链接可用时亮显，否则显示“生成中”。
  - 进程项采用更明显的层次：标题（时间）、副标题（类型），右侧状态 chip；按钮组固定顺序。
  - 自适应展开动画与内容高度回流优化，减少抖动。
- 日志查看
  - 自动滚动可开关（“跟随最新”开/关）；复制按钮；加载失败重试入口。
  - 大文件懒加载：初次只读尾部 N 行，支持“加载更多历史”。

## 视觉美化
- 主题变量：在 `style.css` 顶部引入 CSS 变量（背景、主色、强调色、阴影、圆角），统一渐变与阴影。
- 表单栅格：更稳定的两列到单列响应式断点；标签与控件对齐；hint 文案灰度提升。
- 状态 chip：统一尺寸/色阶，添加图标（运行/停止/错误/完成）。
- 按钮规范：主次区分、禁用态与加载态；滚动条与日志容器对比度优化。
- Header 与 Tabs：粘性头部、Tab 选中态对比更明显；移动端竖排布局间距优化。

## 兼容性与无后端改动原则
- 保持与现有接口一致：`/train`、`/processes`、`/logs/:id`、`/stop/:id`、`/delete/:id`、`/logfiles`、`/logfile-content/:filename`、`/delete-logfile/:filename`（见 train_web_ui.py）。
- 轮询频率保持 5s，可配置；进程日志刷新 1s，仅在“运行中”且展开时。

## 可选增强（后端协作，非必须）
- SSE/WS 实时日志：新增 `/logs/stream/:id`，前端以 EventSource/WS 订阅；减少轮询负载与延迟。
- 进程持久化与重启恢复优化：后端提供简易查询缓存 API，前端减少初始闪烁。

## 交付内容
- 新的 JS 模块文件（上述结构），`index.html` 改为 `type="module"` 入口加载；移除旧 `script.js` 引用。
- 更新后的 `style.css`（保留既有风格，加入变量与状态样式优化）。
- 无后端接口变更；提供 README 片段说明模块职责与使用（如需要）。

## 验收与回滚
- 功能对比清单：
  - 单卡/多卡 Pretrain/SFT/LoRA/DPO/PPO 启动
  - 进程列表与状态变更、SwanLab 链接打开
  - 进程日志展开/自动刷新、日志文件列表/查看/删除
- 若出现问题，可通过 `index.html` 切回旧 `script.js` 引用完成快速回滚。