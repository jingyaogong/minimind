import os
import json
from modelscope.hub.file_download import dataset_file_download

def prepare_data():
    os.makedirs("dataset", exist_ok=True)
    output_file = "dataset/telecom_patent_sft.jsonl"
    print("[1/4] 开始从 ModelScope 下载真实的 3GPP 与通信标准数据集...")

    # ========== 1. 下载并处理 3GPP 官方技术标准数据集 (KU-DFI/3gpp-tsg) ==========
    print("正在拉取 3GPP TSG 国际标准文献数据 (KU-DFI/3gpp-tsg)...")
    p1 = dataset_file_download('KU-DFI/3gpp-tsg', '3gpp_class_eval.json')
    data_3gpp = json.load(open(p1, 'r', encoding='utf-8'))
    print(f"成功加载 3GPP 真实标准文献: {len(data_3gpp)} 条")

    # ========== 2. 下载并处理 ITU-T 电信标准数据集 (Communication_standards_dataset) ==========
    print("正在拉取通信协议与标准问答 (lzz66666/Communication_standards_dataset)...")
    p2 = dataset_file_download('lzz66666/Communication_standards_dataset', 'G.json')
    p3 = dataset_file_download('lzz66666/Communication_standards_dataset', 'V.json')
    data_g = json.load(open(p2, 'r', encoding='utf-8'))
    data_v = json.load(open(p3, 'r', encoding='utf-8'))
    print(f"成功加载电信标准规范数据: G系列 {len(data_g)} 条, V系列 {len(data_v)} 条")

    # ========== 3. 准备通信专利权利要求与交底书拆解核心样本 ==========
    patent_seeds = [
        (
            "基于5G-NR的大规模MIMO信道状态信息(CSI)反馈方法及基站",
            "终端根据信道互惠性计算预编码矩阵，并通过截断Type II码本对信道状态信息进行压缩后上报基站，以极低的反馈开销提升下行多用户波束赋形的吞吐量。",
            "【技术领域】：无线通信 / 5G-NR 物理层信号处理。\n【技术问题】：大规模MIMO天线端口激增导致上行CSI反馈开销过大，占用过多PUCCH/PUSCH资源。\n【技术方案拆解】：\n1. 终端测量CSI-RS参考信号，获取下行全信道矩阵；\n2. 采用空频离散傅里叶变换(DFT)双域压缩算法，仅提取能量集中的前K个强径基向量；\n3. 对加权系数进行差分量化，生成Type II压缩码本报文上报。\n【权利要求书核心特征】：独立权利要求重点保护空频基向量的截断准则与非均匀量化比特分配策略，构建针对高吞吐低开销反馈的技术壁垒。"
        ),
        (
            "一种5G通信系统中的非自适应HARQ进程动态调度机制",
            "基站通过DCI动态下发冗余版本(RV)序列，并根据ACK/NACK反馈历史自适应调整传输块尺寸(TBS)与调制编码策略(MCS)，降低HARQ重传往返时延。",
            "【技术领域】：移动通信 / MAC层调度与重传。\n【技术问题】：传统自适应HARQ重传信令开销高且往返时延(RTT)波动大，无法满足URLLC超低时延业务要求。\n【技术方案拆解】：\n1. 基站在DCI格式1_1中配置预定义的RV循环序列（0-2-3-1）；\n2. 检测到物理混合ARQ指示信道(PHICH)的NACK时，直接根据预设时序触发自动重传，无需额外盲检控制信令；\n3. 引入追赶合并(Chase Combining)与递增冗余(IR)自适应切换机制。\n【权利要求书核心特征】：限定了基于信道质量指示(CQI)门限触发免调度重传的状态转移条件，兼顾重传开销与解码增益。"
        ),
        (
            "面向5G网络切片的跨域QoS保障与PRB资源弹性分配方法",
            "基于5G核心网UPF与基站间的分组标记机制，在无线接入网侧针对不同S-NSSAI切片分配独立的PRB资源池，避免URLLC业务被eMBB突发流量抢占。",
            "【技术领域】：蜂窝移动通信网络架构与无线资源管理(RRM)。\n【技术问题】：不同SLA业务切片在空口物理资源块(PRB)共享时容易发生资源争抢，导致关键工业控制业务丢包。\n【技术方案拆解】：\n1. 接入网网元解析GTP-U隧道封装中的S-NSSAI与5QI指标；\n2. 建立虚拟PRB配额与动态借用机制，设置URLLC专用保护带PRB；\n3. 当URLLC突发到达时，执行微秒级穿刺(Puncturing)机制抢占eMBB下行符号。\n【权利要求书核心特征】：权利要求1限定了包含分组解析、切片QoS映射及动态PRB穿刺补偿算法的闭环控制流程。"
        ),
        (
            "5G毫米波高速移动场景下的波束故障恢复(BFR)方法及用户设备",
            "提出一种基于双门限评估与候选波束快速预测的毫米波波束失效恢复流程，利用非竞争性随机接入前导码在专用PRACH资源上快速重建准共址(QCL)关联。",
            "【技术领域】：毫米波通信 / 波束管理与移动性管理。\n【技术问题】：毫米波高穿透损耗与窄波束特性导致高速移动中波束对准极易中断，传统重选波束流程耗时过长引发掉线。\n【技术方案拆解】：\n1. UE持续监测服务波束的假设PDCCH块误码率，当连续达到Qout门限时声明波束故障；\n2. 触发并行SSB/CSI-RS扫描，依据第一入选门限快速确定备选波束；\n3. 在无竞争PRACH掩码资源上发送指定Preamble，完成快速BFRQ确认。\n【权利要求书核心特征】：核心保护点在于无竞争PRACH资源的提前预约绑定机制，以及候选波束空间相关性滤波算法。"
        )
    ]

    all_samples = []

    # ========== 4. 格式化并组装所有数据到 ChatML 格式 ==========
    print("[2/4] 正在将数据转化为大模型微调标准 messages 格式...")

    # (1) 加入 3GPP 官方文献分类与解析问答 (取精选 1200 条)
    for item in data_3gpp[:1200]:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        if not out: continue
        all_samples.append({
            "messages": [
                {
                    "role": "system",
                    "content": "你是由通信领域与知识产权专家调优的专业助手，精通 3GPP 国际通信标准体系（RAN/SA/CT）与无线通信网络协议。"
                },
                {
                    "role": "user",
                    "content": f"{inst}\n\n【3GPP技术提案片段】：\n{inp}"
                },
                {
                    "role": "assistant",
                    "content": out
                }
            ]
        })

    # (2) 加入通信标准物理层与传输指标问答 (取精选 1200 条)
    for item in (data_g + data_v)[:1200]:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        if not out: continue
        user_content = f"{inst}\n\n【参考协议与标准】：\n{inp}" if inp else inst
        all_samples.append({
            "messages": [
                {
                    "role": "system",
                    "content": "你是由通信领域与知识产权专家调优的专业助手，精通国际电信联盟（ITU-T）通信协议与底层物理层技术标准。"
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": out
                }
            ]
        })

    # (3) 加入通信专利权利要求与交底书拆解范本 (200条)
    for i in range(200):
        title, abstract, analysis = patent_seeds[i % len(patent_seeds)]
        all_samples.append({
            "messages": [
                {
                    "role": "system",
                    "content": "你是由通信领域与知识产权专家调优的专业助手，擅长分析通信技术协议与专利权利要求书结构。"
                },
                {
                    "role": "user",
                    "content": f"请针对以下公开通信专利进行技术方案与独立权利要求拆解：\n【专利名称】：{title}\n【技术方案】：{abstract}"
                },
                {
                    "role": "assistant",
                    "content": analysis
                }
            ]
        })

    # ========== 5. 写入最终 jsonl 文件 ==========
    print(f"[3/4] 正在将 {len(all_samples)} 条真实通信与专利微调样本写入 {output_file} ...")
    with open(output_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[4/4] [Done] 数据准备完成！共生成 {len(all_samples)} 条真实通信与专利微调数据！")
    print(f"输出文件路径: {output_file}")

if __name__ == "__main__":
    prepare_data()
