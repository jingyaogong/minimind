// 标签切换功能
function openTab(evt, tabName) {
    var i, tabContent, tabLinks;
    tabContent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabContent.length; i++) {
        tabContent[i].classList.add("hidden");
    }
    tabLinks = document.getElementsByClassName("tab");
    for (i = 0; i < tabLinks.length; i++) {
        tabLinks[i].classList.remove("active");
    }
    document.getElementById(tabName).classList.remove("hidden");
    evt.currentTarget.classList.add("active");
    
    // 如果切换到进程页面，刷新进程列表
    if (tabName === 'processes') {
        loadProcesses();
    } else if (tabName === 'logfiles') {
        loadLogFiles();
    }
}

// 根据训练类型显示/隐藏特定参数
document.getElementById('train_type').addEventListener('change', function() {
    const trainType = this.value;
    const pretrainSftFields = document.querySelectorAll('.pretrain-sft');
    const loraFields = document.querySelectorAll('.lora');
    
    pretrainSftFields.forEach(field => {
        field.style.display = (trainType === 'pretrain' || trainType === 'sft') ? 'block' : 'none';
    });
    
    loraFields.forEach(field => {
        field.style.display = trainType === 'lora' ? 'block' : 'none';
    });
    
    // 设置默认值
    if (trainType === 'pretrain') {
        document.getElementById('save_dir').value = '../out';
        document.getElementById('save_weight').value = 'pretrain';
        document.getElementById('epochs').value = '1';
        document.getElementById('batch_size').value = '32';
        document.getElementById('learning_rate').value = '5e-4';
        document.getElementById('data_path').value = '../dataset/pretrain_hq.jsonl';
        document.getElementById('from_weight').value = 'none';
        document.getElementById('log_interval').value = '100';
        document.getElementById('save_interval').value = '100';
    } else if (trainType === 'sft') {
        document.getElementById('save_dir').value = '../out';
        document.getElementById('save_weight').value = 'full_sft';
        document.getElementById('epochs').value = '2';
        document.getElementById('batch_size').value = '16';
        document.getElementById('learning_rate').value = '5e-7';
        document.getElementById('data_path').value = '../dataset/sft_mini_512.jsonl';
        document.getElementById('from_weight').value = 'pretrain';
        document.getElementById('log_interval').value = '100';
        document.getElementById('save_interval').value = '100';
    } else if (trainType === 'lora') {
        document.getElementById('save_dir').value = '../out/lora';
        document.getElementById('lora_name').value = 'lora_identity';
        document.getElementById('epochs').value = '50';
        document.getElementById('batch_size').value = '32';
        document.getElementById('learning_rate').value = '1e-4';
        document.getElementById('data_path').value = '../dataset/lora_identity.jsonl';
        document.getElementById('from_weight').value = 'full_sft';
        document.getElementById('log_interval').value = '10';
        document.getElementById('save_interval').value = '1';
    }
});

// 初始触发一次change事件以设置默认值
document.getElementById('train_type').dispatchEvent(new Event('change'));

// 加载进程列表
function loadProcesses() {
    fetch('/processes')
        .then(response => response.json())
        .then(data => {
            const processList = document.getElementById('process-list');
            processList.innerHTML = '';
            
            if (data.length === 0) {
                processList.innerHTML = '<p>暂无训练进程</p>';
                return;
            }
            
            data.forEach(process => {
                const processItem = document.createElement('div');
                processItem.className = 'process-item';
                
                let statusClass = '';
                // 根据后端返回的status字段设置状态类
                if (process.status === '运行中') {
                    statusClass = 'status-running';
                } else if (process.status === '手动停止') {
                    statusClass = 'status-manual-stop';
                } else if (process.status === '出错') {
                    statusClass = 'status-error';
                } else {
                    statusClass = 'status-completed';
                }
                
                processItem.innerHTML = `
                    <div class="process-info">
                        <div>
                            <strong>${process.train_type}</strong> - ${process.start_time}
                        </div>
                        <div>
                            <span class="process-status ${statusClass}">${process.status}</span>
                        </div>
                    </div>
                    <div>
                        <button class="btn-logs" onclick="showLogs('${process.id}')">查看日志</button>
                        <button class="btn-logs" onclick="refreshLog('${process.id}')">刷新日志</button>
                        ${process.running ? `<button class="btn-stop" onclick="stopProcess('${process.id}')">停止训练</button>` : ''}
                    </div>
                    <div id="logs-${process.id}" class="logs-container hidden"></div>
                `;
                
                processList.appendChild(processItem);
            });
        });
}

// 显示日志
function showLogs(processId) {
    const logsContainer = document.getElementById(`logs-${processId}`);
    logsContainer.classList.toggle('hidden');
    
    if (!logsContainer.classList.contains('hidden')) {
        loadLogContent(processId, logsContainer);
    }
}

// 刷新日志
function refreshLog(processId) {
    const logsContainer = document.getElementById(`logs-${processId}`);
    if (!logsContainer.classList.contains('hidden')) {
        loadLogContent(processId, logsContainer);
    }
}

// 加载日志内容
function loadLogContent(processId, logsContainer) {
    fetch(`/logs/${processId}`)
        .then(response => response.text())
        .then(logs => {
            logsContainer.textContent = logs;
            logsContainer.scrollTop = logsContainer.scrollHeight;
        });
}

// 停止进程
function stopProcess(processId) {
    if (confirm('确定要停止这个训练进程吗？')) {
        fetch(`/stop/${processId}`, {
            method: 'POST'
        })
        .then(() => {
            loadProcesses();
        });
    }
}

// 表单提交处理
document.getElementById('train-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());
    
    fetch('/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            alert('训练已开始！');
            openTab(event, 'processes');
        } else {
            alert('训练启动失败：' + result.error);
        }
    });
});

// 加载日志文件列表
function loadLogFiles() {
    fetch('/logfiles')
        .then(response => response.json())
        .then(data => {
            const logfilesList = document.getElementById('logfiles-list');
            logfilesList.innerHTML = '';
            
            if (data.length === 0) {
                logfilesList.innerHTML = '<p>暂无日志文件</p>';
                return;
            }
            
            // 按日期倒序排序
            data.sort((a, b) => new Date(b.modified_time) - new Date(a.modified_time));
            
            data.forEach(logfile => {
                const fileItem = document.createElement('div');
                fileItem.className = 'process-item';
                
                // 从文件名提取训练类型
                let trainType = '未知';
                if (logfile.filename.includes('train_pretrain_')) {
                    trainType = 'pretrain';
                } else if (logfile.filename.includes('train_sft_')) {
                    trainType = 'sft';
                } else if (logfile.filename.includes('train_lora_')) {
                    trainType = 'lora';
                }
                
                fileItem.innerHTML = `
                    <div class="process-info">
                        <div>
                            <strong>${trainType}</strong> - ${logfile.filename}
                        </div>
                        <div>
                            <span class="process-status status-completed">已保存</span>
                            <span style="margin-left: 10px; color: #999; font-size: 0.9em;">${logfile.modified_time}</span>
                        </div>
                    </div>
                    <div>
                        <button class="btn-logs" onclick="viewLogFile('${logfile.filename}', this)">查看日志</button>
                    </div>
                    <div id="log-content-${logfile.filename.replace(/\./g, '-')}" class="logs-container hidden"></div>
                `;
                
                logfilesList.appendChild(fileItem);
            });
        });
}

// 查看日志文件内容
function viewLogFile(filename, button) {
    const safeFilename = filename.replace(/\./g, '-');
    const logContainer = button.closest('.process-item').querySelector(`#log-content-${safeFilename}`);
    logContainer.classList.toggle('hidden');
    
    if (!logContainer.classList.contains('hidden') && logContainer.textContent === '') {
        logContainer.textContent = '加载日志中...';
        
        fetch(`/logfile-content/${encodeURIComponent(filename)}`)
            .then(response => response.text())
            .then(logs => {
                logContainer.textContent = logs;
                logContainer.scrollTop = 0;
            });
    }
}