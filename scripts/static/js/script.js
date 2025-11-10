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
    
    // 当切换到其他标签页时，清除所有日志自动更新定时器
    if (tabName !== 'processes') {
        for (let processId in logTimers) {
            clearInterval(logTimers[processId]);
            delete logTimers[processId];
        }
    }
    
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
    const dpoFields = document.querySelectorAll('.dpo');
    const dpoParamCard = document.querySelector('.parameter-card.dpo');
    
    pretrainSftFields.forEach(field => {
        field.style.display = (trainType === 'pretrain' || trainType === 'sft') ? 'block' : 'none';
    });
    
    loraFields.forEach(field => {
        field.style.display = trainType === 'lora' ? 'block' : 'none';
    });
    
    dpoFields.forEach(field => {
        field.style.display = trainType === 'dpo' ? 'block' : 'none';
    });
    
    if (dpoParamCard) {
        dpoParamCard.style.display = trainType === 'dpo' ? 'block' : 'none';
    }
    
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
        // 模型结构参数默认值
        document.getElementById('hidden_size').value = '512';
        document.getElementById('num_hidden_layers').value = '8';
        document.getElementById('max_seq_len').value = '512';
        document.getElementById('use_moe').value = '0';
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
        // 模型结构参数默认值
        document.getElementById('hidden_size').value = '512';
        document.getElementById('num_hidden_layers').value = '8';
        document.getElementById('max_seq_len').value = '512';
        document.getElementById('use_moe').value = '0';
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
        // 模型结构参数默认值
        document.getElementById('hidden_size').value = '512';
        document.getElementById('num_hidden_layers').value = '8';
        document.getElementById('max_seq_len').value = '512';
        document.getElementById('use_moe').value = '0';
    } else if (trainType === 'dpo') {
        document.getElementById('save_dir').value = '../out';
        document.getElementById('save_weight').value = 'dpo';
        document.getElementById('epochs').value = '1';
        document.getElementById('batch_size').value = '4';
        document.getElementById('learning_rate').value = '4e-8';
        document.getElementById('data_path').value = '../dataset/dpo.jsonl';
        document.getElementById('from_weight').value = 'full_sft';
        document.getElementById('log_interval').value = '100';
        document.getElementById('save_interval').value = '100';
        document.getElementById('beta').value = '0.1';
        // 模型结构参数默认值
        document.getElementById('hidden_size').value = '512';
        document.getElementById('num_hidden_layers').value = '8';
        document.getElementById('max_seq_len').value = '1024';
        document.getElementById('use_moe').value = '0';
    }
});

// 初始触发一次change事件以设置默认值
document.getElementById('train_type').dispatchEvent(new Event('change'));

// 页面加载时启动进程状态轮询
window.addEventListener('load', () => {
    startProcessPolling();
    
    // 加载初始进程列表
    loadProcesses();
});

// 在进程标签页切换时也触发一次检查
const tabLinks = document.querySelectorAll('.tab');
tabLinks.forEach(tab => {
    tab.addEventListener('click', () => {
        // 给标签页切换一些时间完成
        setTimeout(() => {
            const processesTab = document.querySelector('.tab.active');
            if (processesTab && processesTab.textContent.includes('进程')) {
                checkProcessStatusChanges();
            }
        }, 100);
    });
});

// 添加定时轮询机制，每5秒检查一次进程状态变化
let processPollingTimer;

// 开始进程状态轮询
function startProcessPolling() {
    // 清除可能存在的旧定时器
    if (processPollingTimer) {
        clearInterval(processPollingTimer);
    }
    // 每5秒检查一次进程状态
    processPollingTimer = setInterval(() => {
        // 只有在进程标签页打开时才检查
        const processesTab = document.querySelector('.tab.active');
        if (processesTab && processesTab.textContent.includes('进程')) {
            checkProcessStatusChanges();
        }
    }, 5000);
}

// 检查进程状态变化（特别是错误状态）
function checkProcessStatusChanges() {
    fetch('/processes')
        .then(response => response.json())
        .then(data => {
            // 遍历每个进程，检查状态变化
            data.forEach(process => {
                const processItem = document.querySelector(`[data-process-id="${process.id}"]`);
                if (processItem) {
                    const currentStatus = processItem.dataset.processStatus;
                    const newStatus = process.status;
                    
                    // 如果状态发生变化，更新进程项
                    if (currentStatus !== newStatus) {
                        updateProcessItem(processItem, process);
                        
                        // 如果状态变为出错，显示通知
                        if (newStatus === '出错') {
                            showNotification(`进程 ${process.train_type} 已出错`, 'error');
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('检查进程状态时出错:', error);
        });
}

// 更新单个进程项
function updateProcessItem(processItem, process) {
    // 更新数据属性
    processItem.dataset.processStatus = process.status;
    
    // 更新状态类和文本
    const statusElement = processItem.querySelector('.process-status');
    if (statusElement) {
        // 移除所有状态类
        statusElement.classList.remove('status-running', 'status-manual-stop', 'status-error', 'status-completed');
        
        // 添加新的状态类
        let statusClass = '';
        if (process.status === '运行中') {
            statusClass = 'status-running';
        } else if (process.status === '手动停止') {
            statusClass = 'status-manual-stop';
        } else if (process.status === '出错') {
            statusClass = 'status-error';
        } else {
            statusClass = 'status-completed';
        }
        
        statusElement.classList.add(statusClass);
        statusElement.textContent = process.status;
    }
    
    // 更新停止按钮
    const stopButton = processItem.querySelector('.btn-stop');
    if (stopButton) {
        if (!process.running) {
            stopButton.remove();
        }
    } else if (process.running) {
        // 如果按钮不存在但进程仍在运行，添加停止按钮
        const buttonContainer = processItem.querySelector('div:last-child');
        if (buttonContainer) {
            const newStopButton = document.createElement('button');
            newStopButton.className = 'btn-stop';
            newStopButton.onclick = () => stopProcess(process.id);
            newStopButton.textContent = '停止训练';
            buttonContainer.appendChild(newStopButton);
        }
    }
    
    // 处理删除按钮
    const deleteButton = processItem.querySelector('.btn-delete');
    if (!process.running) {
        // 非运行中状态应该有删除按钮
        if (!deleteButton) {
            const buttonContainer = processItem.querySelector('div:last-child');
            if (buttonContainer) {
                const newDeleteButton = document.createElement('button');
                newDeleteButton.className = 'btn-delete';
                newDeleteButton.onclick = () => deleteProcess(process.id);
                newDeleteButton.textContent = '删除';
                buttonContainer.appendChild(newDeleteButton);
            }
        }
    } else if (deleteButton) {
        // 运行中状态不应该有删除按钮
        deleteButton.remove();
    }
    
    // 如果进程不再运行中，清除日志定时器
    if (!process.running && logTimers[process.id]) {
        clearInterval(logTimers[process.id]);
        delete logTimers[process.id];
    }
}

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
            
            // 按启动时间倒序排序，最新启动的进程显示在前面
            data.sort((a, b) => new Date(b.start_time) - new Date(a.start_time));
            
            // 按训练类型分组
            const groupedProcesses = {};
            data.forEach(process => {
                if (!groupedProcesses[process.train_type]) {
                    groupedProcesses[process.train_type] = [];
                }
                groupedProcesses[process.train_type].push(process);
            });
            
            // 定义训练类型的显示顺序（可以根据需要调整）
            const trainTypeOrder = ['pretrain', 'sft', 'lora', 'dpo'];
            
            // 创建类型分组并添加收起/展开功能的辅助函数
            function createTypeGroupWithToggle(trainType, processes) {
                // 创建类型分组容器
                const typeGroup = document.createElement('div');
                typeGroup.className = 'process-type-group';
                
                // 创建标题容器（包含标题文本和切换按钮）
                const titleContainer = document.createElement('div');
                titleContainer.className = 'process-type-header';
                titleContainer.dataset.expanded = 'true'; // 默认展开状态
                
                // 添加类型标题
                const typeTitle = document.createElement('h3');
                typeTitle.className = 'process-type-title';
                typeTitle.textContent = getTrainTypeDisplayName(trainType);
                
                // 添加切换按钮
                const toggleButton = document.createElement('button');
                toggleButton.className = 'toggle-btn';
                toggleButton.innerHTML = '▼'; // 向下箭头表示展开
                toggleButton.onclick = function(e) {
                    e.stopPropagation(); // 防止触发标题容器的点击事件
                    toggleGroup(titleContainer);
                };
                
                // 将标题和按钮添加到容器
                titleContainer.appendChild(typeTitle);
                titleContainer.appendChild(toggleButton);
                
                // 添加点击标题也可以切换展开/收起
                titleContainer.onclick = function() {
                    toggleGroup(titleContainer);
                };
                
                // 创建内容容器，用于容纳进程项
                const contentContainer = document.createElement('div');
                contentContainer.className = 'process-type-content';
                
                // 添加该类型的所有进程
                processes.forEach(process => {
                    addProcessItemToGroup(contentContainer, process);
                });
                
                // 将标题容器和内容容器添加到类型分组
                typeGroup.appendChild(titleContainer);
                typeGroup.appendChild(contentContainer);
                
                return typeGroup;
            }
            
            // 切换分组展开/收起状态的函数
            function toggleGroup(headerElement) {
                const isExpanded = headerElement.dataset.expanded === 'true';
                const contentContainer = headerElement.nextElementSibling;
                const toggleButton = headerElement.querySelector('.toggle-btn');
                
                if (isExpanded) {
                    // 收起分组
                    headerElement.dataset.expanded = 'false';
                    contentContainer.style.maxHeight = '0';
                    contentContainer.style.overflow = 'hidden';
                    toggleButton.innerHTML = '▶'; // 向右箭头表示收起
                } else {
                    // 展开分组前先重置样式
                    contentContainer.style.overflow = 'hidden'; // 确保计算高度准确
                    contentContainer.style.maxHeight = 'none'; // 临时设置为none以获取真实高度
                    
                    // 获取真实高度
                    const actualHeight = contentContainer.scrollHeight;
                    
                    // 然后先设置为0，准备动画
                    contentContainer.style.maxHeight = '0';
                    
                    // 强制重排
                    contentContainer.offsetHeight;
                    
                    // 展开分组
                    headerElement.dataset.expanded = 'true';
                    contentContainer.style.maxHeight = actualHeight + 'px';
                    
                    setTimeout(() => {
                        // 动画完成后，设置为真实高度并允许溢出
                        contentContainer.style.maxHeight = 'none';
                        contentContainer.style.overflow = 'visible';
                    }, 300); // 动画完成后显示溢出内容
                    
                    toggleButton.innerHTML = '▼'; // 向下箭头表示展开
                }
            }
            
            // 首先显示有明确顺序的训练类型
            trainTypeOrder.forEach(trainType => {
                if (groupedProcesses[trainType]) {
                    const typeGroup = createTypeGroupWithToggle(trainType, groupedProcesses[trainType]);
                    processList.appendChild(typeGroup);
                    
                    // 从分组中移除已处理的类型
                    delete groupedProcesses[trainType];
                }
            });
            
            // 显示剩余的训练类型（不在预定义顺序中的）
            Object.keys(groupedProcesses).forEach(trainType => {
                const typeGroup = createTypeGroupWithToggle(trainType, groupedProcesses[trainType]);
                processList.appendChild(typeGroup);
            });
        });
}

// 获取训练类型的显示名称
function getTrainTypeDisplayName(trainType) {
    const typeNames = {
        'pretrain': '预训练 (Pretrain)',
        'sft': '全参数监督微调 (SFT - Full)',
        'lora': 'LoRA监督微调 (SFT - Lora)',
        'dpo': '直接偏好优化 (RL - DPO)'
    };
    return typeNames[trainType] || trainType;
}

// 添加进程项到分组
function addProcessItemToGroup(parentElement, process) {
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
    
    // 设置进程数据属性，用于后续检查状态
    processItem.dataset.processId = process.id;
    processItem.dataset.processStatus = process.status;
    
    // 检查是否显示删除按钮（对于非运行中的进程）
    const showDeleteButton = !process.running;
    
    processItem.innerHTML = `
        <div class="process-info">
            <div>
                <strong>${process.start_time}</strong>
            </div>
            <div>
                <span class="process-status ${statusClass}">${process.status}</span>
            </div>
        </div>
        <div>
            <button class="btn-logs" onclick="showLogs('${process.id}')">查看日志</button>
            <button class="btn-logs" onclick="refreshLog('${process.id}')">刷新日志</button>
            ${process.running ? `<button class="btn-stop" onclick="stopProcess('${process.id}')">停止训练</button>` : ''}
            ${showDeleteButton ? `<button class="btn-delete" onclick="deleteProcess('${process.id}')">删除</button>` : ''}
        </div>
        <div id="logs-${process.id}" class="logs-container hidden"></div>
    `;
    
    parentElement.appendChild(processItem);
}

// 存储训练日志定时器的ID
let logTimers = {};

// 显示日志
function showLogs(processId) {
    const logsContainer = document.getElementById(`logs-${processId}`);
    logsContainer.classList.toggle('hidden');
    
    if (!logsContainer.classList.contains('hidden')) {
        // 加载日志内容
        loadLogContent(processId, logsContainer);
        
        // 清除可能存在的旧定时器
        if (logTimers[processId]) {
            clearInterval(logTimers[processId]);
        }
        
        // 查找进程元素，获取其状态
        const processItem = document.querySelector(`[data-process-id="${processId}"]`);
        const isRunning = processItem && processItem.dataset.processStatus === '运行中';
        
        // 只有运行中的进程才设置自动刷新定时器
        if (isRunning) {
            logTimers[processId] = setInterval(() => {
                // 检查日志容器是否可见
                if (!logsContainer.classList.contains('hidden')) {
                    // 再次检查进程状态，确保仍然是运行中
                    const currentProcessItem = document.querySelector(`[data-process-id="${processId}"]`);
                    const stillRunning = currentProcessItem && currentProcessItem.dataset.processStatus === '运行中';
                    
                    if (stillRunning) {
                        loadLogContent(processId, logsContainer);
                    } else {
                        // 如果进程不再运行中，清除定时器
                        clearInterval(logTimers[processId]);
                        delete logTimers[processId];
                    }
                }
            }, 1000);
        }
    } else {
        // 隐藏日志时清除定时器
        if (logTimers[processId]) {
            clearInterval(logTimers[processId]);
            delete logTimers[processId];
        }
    }
}

// 刷新日志
function refreshLog(processId) {
    const logsContainer = document.getElementById(`logs-${processId}`);
    if (!logsContainer.classList.contains('hidden')) {
        loadLogContent(processId, logsContainer);
        
        // 查找进程元素，获取其状态
        const processItem = document.querySelector(`[data-process-id="${processId}"]`);
        const isRunning = processItem && processItem.dataset.processStatus === '运行中';
        
        // 重置定时器，确保从现在开始每1秒更新一次
        if (logTimers[processId]) {
            clearInterval(logTimers[processId]);
        }
        
        // 只有运行中的进程才设置定时器
        if (isRunning) {
            logTimers[processId] = setInterval(() => {
                if (!logsContainer.classList.contains('hidden')) {
                    // 再次检查进程状态
                    const currentProcessItem = document.querySelector(`[data-process-id="${processId}"]`);
                    const stillRunning = currentProcessItem && currentProcessItem.dataset.processStatus === '运行中';
                    
                    if (stillRunning) {
                        loadLogContent(processId, logsContainer);
                    } else {
                        // 如果进程不再运行中，清除定时器
                        clearInterval(logTimers[processId]);
                        delete logTimers[processId];
                    }
                }
            }, 1000);
        }
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

// 显示右上角消息弹窗
function showNotification(message, type = 'success') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // 添加到页面
    document.body.appendChild(notification);
    
    // 显示动画
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // 自动关闭
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// 自定义确认对话框
function showConfirmDialog(message, onConfirm, onCancel = null) {
    // 检查是否已存在对话框，如果有则移除
    const existingDialog = document.querySelector('.custom-dialog');
    if (existingDialog) {
        document.body.removeChild(existingDialog);
    }
    
    // 创建对话框遮罩
    const dialogOverlay = document.createElement('div');
    dialogOverlay.className = 'dialog-overlay';
    
    // 创建对话框容器
    const dialogContainer = document.createElement('div');
    dialogContainer.className = 'custom-dialog';
    
    // 设置对话框内容
    dialogContainer.innerHTML = `
        <div class="dialog-content">
            <div class="dialog-message">${message}</div>
            <div class="dialog-actions">
                <button class="dialog-button dialog-cancel">取消</button>
                <button class="dialog-button dialog-confirm">确认</button>
            </div>
        </div>
    `;
    
    // 添加到页面
    dialogOverlay.appendChild(dialogContainer);
    document.body.appendChild(dialogOverlay);
    
    // 添加动画类
    setTimeout(() => {
        dialogOverlay.classList.add('show');
        dialogContainer.classList.add('show');
    }, 10);
    
    // 绑定确认按钮事件
    const confirmBtn = dialogContainer.querySelector('.dialog-confirm');
    confirmBtn.addEventListener('click', () => {
        if (onConfirm) onConfirm();
        closeDialog(dialogOverlay);
    });
    
    // 绑定取消按钮事件
    const cancelBtn = dialogContainer.querySelector('.dialog-cancel');
    cancelBtn.addEventListener('click', () => {
        if (onCancel) onCancel();
        closeDialog(dialogOverlay);
    });
    
    // 点击遮罩层也可以取消
    dialogOverlay.addEventListener('click', (e) => {
        if (e.target === dialogOverlay) {
            if (onCancel) onCancel();
            closeDialog(dialogOverlay);
        }
    });
}

// 关闭对话框函数
function closeDialog(dialogOverlay) {
    dialogOverlay.classList.remove('show');
    const dialogContainer = dialogOverlay.querySelector('.custom-dialog');
    dialogContainer.classList.remove('show');
    
    setTimeout(() => {
        if (dialogOverlay.parentNode) {
            document.body.removeChild(dialogOverlay);
        }
    }, 300);
}

// 删除进程
function deleteProcess(processId) {
    showConfirmDialog(
        '确定要删除这个训练进程吗？此操作不可恢复。',
        () => {
            // 确认删除
            fetch(`/delete/${processId}`, {
                method: 'POST'
            })
            .then(response => {
                // 检查响应状态
                if (!response.ok) {
                    throw new Error('删除请求失败');
                }
                return response.json().catch(() => ({})); // 即使没有JSON响应也继续
            })
            .then(() => {
                // 从UI中移除进程项
                const processItem = document.querySelector(`[data-process-id="${processId}"]`);
                if (processItem) {
                    // 添加移除动画
                    processItem.style.transition = 'opacity 0.3s, transform 0.3s';
                    processItem.style.opacity = '0';
                    processItem.style.transform = 'translateX(-20px)';
                    
                    // 动画完成后移除
                    setTimeout(() => {
                        // 获取内容容器和类型分组
                        const contentContainer = processItem.closest('.process-type-content');
                        const typeGroup = contentContainer ? contentContainer.closest('.process-type-group') : null;
                        
                        if (processItem.parentNode) {
                            processItem.parentNode.removeChild(processItem);
                        }
                        
                        // 检查内容容器中是否还有进程项
                        if (contentContainer) {
                            const remainingProcessesInGroup = contentContainer.querySelectorAll('.process-item');
                            
                            // 如果内容容器中没有进程项了，移除整个分组
                            if (remainingProcessesInGroup.length === 0 && typeGroup) {
                                setTimeout(() => {
                                    // 添加分组移除动画
                                    typeGroup.style.transition = 'opacity 0.3s, transform 0.3s';
                                    typeGroup.style.opacity = '0';
                                    typeGroup.style.transform = 'translateY(-10px)';
                                    
                                    setTimeout(() => {
                                        if (typeGroup.parentNode) {
                                            typeGroup.parentNode.removeChild(typeGroup);
                                        }
                                        
                                        // 检查是否还有进程项
                                        const remainingProcesses = document.querySelectorAll('.process-item');
                                        if (remainingProcesses.length === 0) {
                                            const processList = document.getElementById('process-list');
                                            processList.innerHTML = '<p>暂无训练进程</p>';
                                        }
                                    }, 300);
                                }, 100);
                            } else {
                                // 重新计算内容容器的高度（如果分组是展开的）
                                const headerElement = contentContainer.previousElementSibling;
                                if (headerElement && headerElement.dataset.expanded === 'true') {
                                    contentContainer.style.maxHeight = contentContainer.scrollHeight + 'px';
                                }
                                
                                // 检查是否还有进程项
                                const remainingProcesses = document.querySelectorAll('.process-item');
                                if (remainingProcesses.length === 0) {
                                    const processList = document.getElementById('process-list');
                                    processList.innerHTML = '<p>暂无训练进程</p>';
                                }
                            }
                        } else {
                            // 检查是否还有进程项
                            const remainingProcesses = document.querySelectorAll('.process-item');
                            if (remainingProcesses.length === 0) {
                                const processList = document.getElementById('process-list');
                                processList.innerHTML = '<p>暂无训练进程</p>';
                            }
                        }
                    }, 300);
                }
                
                // 清除日志定时器
                if (logTimers[processId]) {
                    clearInterval(logTimers[processId]);
                    delete logTimers[processId];
                }
                
                showNotification('训练进程已删除', 'success');
            })
            .catch(error => {
                console.error('删除进程时出错:', error);
                showNotification('删除进程失败，请刷新页面重试', 'error');
            });
        }
    );
}

// 停止进程
function stopProcess(processId) {
    showConfirmDialog(
        '确定要停止这个训练进程吗？',
        () => {
            // 确认停止
            fetch(`/stop/${processId}`, {
                method: 'POST'
            })
            .then(() => {
                // 立即给用户反馈，设置为手动停止状态
                const processItem = document.querySelector(`[data-process-id="${processId}"]`);
                if (processItem) {
                    // 先设置为手动停止状态（即时反馈）
                    processItem.dataset.processStatus = '手动停止';
                    const statusElement = processItem.querySelector('.process-status');
                    if (statusElement) {
                        statusElement.classList.remove('status-running', 'status-error', 'status-completed');
                        statusElement.classList.add('status-manual-stop');
                        statusElement.textContent = '手动停止';
                    }
                    
                    // 移除停止按钮
                    const stopButton = processItem.querySelector('.btn-stop');
                    if (stopButton) {
                        stopButton.remove();
                    }
                    
                    // 清除日志定时器
                    if (logTimers[processId]) {
                        clearInterval(logTimers[processId]);
                        delete logTimers[processId];
                    }
                }
                
                // 显示通知
                showNotification('训练进程已停止', 'info');
                
                // 为了确保状态准确，仍然获取完整列表并更新单个进程
                fetch('/processes')
                    .then(response => response.json())
                    .then(data => {
                        // 从完整列表中找到特定进程
                        const updatedProcess = data.find(p => p.id === processId);
                        if (updatedProcess && processItem) {
                            // 使用updateProcessItem函数更新进程（如果需要的话）
                            updateProcessItem(processItem, updatedProcess);
                        }
                    })
                    .catch(error => {
                        console.error('获取进程列表更新单个进程失败:', error);
                    });
            })
            .catch(() => {
                showNotification('停止进程失败', 'error');
            });
        },
        () => {
            // 取消停止，可选的回调
            showNotification('已取消停止操作', 'info');
        }
    );
}

// 表单提交处理
document.getElementById('train-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());
    
    // 立即显示加载通知
    showNotification('正在启动训练...', 'info');
    
    // 延迟1秒后发送请求，确保两个通知之间有间隔
    setTimeout(() => {
        fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('网络响应异常');
                }
                return response.json();
            })
            .then(result => {
                if (result.success) {
                    showNotification('训练已开始！', 'success');
                    
                    // 延迟1秒后切换到进程标签页，确保进程信息已更新
                    setTimeout(() => {
                        // 获取进程标签按钮并触发点击
                        const processTab = document.querySelector('.tab[onclick*="processes"]');
                        if (processTab) {
                            processTab.click();
                        }
                    }, 1000);
                } else {
                    // 显示错误通知
                    showNotification('训练启动失败：' + result.error, 'error');
                }
            })
            .catch(error => {
                // 显示等待通知
                showNotification('启动训练中，请耐心等待...', 'info');
            });
        }, 1000); // 1秒延迟确保两个通知之间有间隔
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
            
            // 按训练类型分组
            const groupedLogFiles = {};
            data.forEach(logfile => {
                // 从文件名提取训练类型
                let trainType = '自定义训练';
                if (logfile.filename.includes('train_pretrain_')) {
                    trainType = 'pretrain';
                } else if (logfile.filename.includes('train_sft_')) {
                    trainType = 'sft';
                } else if (logfile.filename.includes('train_lora_')) {
                    trainType = 'lora';
                } else if (logfile.filename.includes('train_dpo_')) {
                    trainType = 'dpo';
                } else if (logfile.filename.includes('train_ppo_')) {
                    trainType = 'ppo';
                } else if (logfile.filename.includes('train_grpo_')) {
                    trainType = 'grpo';
                } else if (logfile.filename.includes('train_spo_')) {
                    trainType = 'spo';
                }
                
                // 为日志文件添加训练类型信息
                logfile.train_type = trainType;
                
                if (!groupedLogFiles[trainType]) {
                    groupedLogFiles[trainType] = [];
                }
                groupedLogFiles[trainType].push(logfile);
            });
            
            // 定义训练类型的显示顺序（与进程显示顺序保持一致）
            const trainTypeOrder = ['pretrain', 'sft', 'lora', 'dpo', 'ppo', 'grpo', 'spo', '未知'];
            
            // 创建类型分组并添加收起/展开功能的辅助函数
            function createTypeGroupWithToggle(trainType, logfiles) {
                // 创建类型分组容器
                const typeGroup = document.createElement('div');
                typeGroup.className = 'process-type-group';
                
                // 创建标题容器（包含标题文本和切换按钮）
                const titleContainer = document.createElement('div');
                titleContainer.className = 'process-type-header';
                titleContainer.dataset.expanded = 'true'; // 默认展开状态
                
                // 添加类型标题
                const typeTitle = document.createElement('h3');
                typeTitle.className = 'process-type-title';
                typeTitle.textContent = getTrainTypeDisplayName(trainType);
                
                // 添加切换按钮
                const toggleButton = document.createElement('button');
                toggleButton.className = 'toggle-btn';
                toggleButton.innerHTML = '▼'; // 向下箭头表示展开
                toggleButton.onclick = function(e) {
                    e.stopPropagation(); // 防止触发标题容器的点击事件
                    toggleGroup(titleContainer);
                };
                
                // 将标题和按钮添加到容器
                titleContainer.appendChild(typeTitle);
                titleContainer.appendChild(toggleButton);
                
                // 添加点击标题也可以切换展开/收起
                titleContainer.onclick = function() {
                    toggleGroup(titleContainer);
                };
                
                // 创建内容容器，用于容纳日志文件项
                const contentContainer = document.createElement('div');
                contentContainer.className = 'process-type-content';
                
                // 添加该类型的所有日志文件
                logfiles.forEach(logfile => {
                    addLogFileItemToGroup(contentContainer, logfile);
                });
                
                // 将标题容器和内容容器添加到类型分组
                typeGroup.appendChild(titleContainer);
                typeGroup.appendChild(contentContainer);
                
                return typeGroup;
            }
            
            // 切换分组展开/收起状态的函数
            function toggleGroup(headerElement) {
                const isExpanded = headerElement.dataset.expanded === 'true';
                const contentContainer = headerElement.nextElementSibling;
                const toggleButton = headerElement.querySelector('.toggle-btn');
                
                if (isExpanded) {
                    // 收起分组
                    headerElement.dataset.expanded = 'false';
                    contentContainer.style.maxHeight = '0';
                    contentContainer.style.overflow = 'hidden';
                    toggleButton.innerHTML = '▶'; // 向右箭头表示收起
                } else {
                    // 展开分组前先重置样式
                    contentContainer.style.overflow = 'hidden'; // 确保计算高度准确
                    contentContainer.style.maxHeight = 'none'; // 临时设置为none以获取真实高度
                    
                    // 获取真实高度
                    const actualHeight = contentContainer.scrollHeight;
                    
                    // 然后先设置为0，准备动画
                    contentContainer.style.maxHeight = '0';
                    
                    // 强制重排
                    contentContainer.offsetHeight;
                    
                    // 展开分组
                    headerElement.dataset.expanded = 'true';
                    contentContainer.style.maxHeight = actualHeight + 'px';
                    
                    setTimeout(() => {
                        // 动画完成后，设置为真实高度并允许溢出
                        contentContainer.style.maxHeight = 'none';
                        contentContainer.style.overflow = 'visible';
                    }, 300); // 动画完成后显示溢出内容
                    
                    toggleButton.innerHTML = '▼'; // 向下箭头表示展开
                }
            }
            
            // 首先显示有明确顺序的训练类型
            trainTypeOrder.forEach(trainType => {
                if (groupedLogFiles[trainType]) {
                    const typeGroup = createTypeGroupWithToggle(trainType, groupedLogFiles[trainType]);
                    logfilesList.appendChild(typeGroup);
                    
                    // 从分组中移除已处理的类型
                    delete groupedLogFiles[trainType];
                }
            });
            
            // 显示剩余的训练类型（不在预定义顺序中的）
            Object.keys(groupedLogFiles).forEach(trainType => {
                const typeGroup = createTypeGroupWithToggle(trainType, groupedLogFiles[trainType]);
                logfilesList.appendChild(typeGroup);
            });
        });
}

// 添加日志文件项到分组
function addLogFileItemToGroup(parentElement, logfile) {
    const fileItem = document.createElement('div');
    fileItem.className = 'process-item';
    
    fileItem.innerHTML = `
        <div class="process-info">
            <div>
                <strong>${logfile.filename}</strong>
            </div>
            <div>
                <span class="process-status status-completed">已保存</span>
                <span style="margin-left: 10px; color: #999; font-size: 0.9em;">${logfile.modified_time}</span>
            </div>
        </div>
        <div>
            <button class="btn-logs" onclick="viewLogFile('${logfile.filename}', this)">查看日志</button>
            <button class="btn-delete" onclick="deleteLogFile('${logfile.filename}', this)">删除</button>
        </div>
        <div id="log-content-${logfile.filename.replace(/\./g, '-')}" class="logs-container hidden"></div>
    `;
    
    parentElement.appendChild(fileItem);
}

// 删除日志文件
function deleteLogFile(filename, button) {
    // 显示确认对话框
    showConfirmDialog(
        `确定要删除日志文件 "${filename}" 吗？此操作无法恢复。`,
        function() {
            // 用户确认删除
            const processItem = button.closest('.process-item');
            const contentContainer = processItem.closest('.process-type-content');
            const typeGroup = contentContainer.closest('.process-type-group');
            
            // 显示加载状态
            const originalText = button.textContent;
            button.textContent = '删除中...';
            button.disabled = true;
            
            // 发送删除请求到服务器
            fetch(`/delete-logfile/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('删除失败');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    // 从UI中移除日志项
                    processItem.remove();
                    
                    // 检查是否需要移除整个分组
                    if (contentContainer.children.length === 0) {
                        typeGroup.remove();
                    } else {
                        // 更新内容容器的高度
                        const headerElement = contentContainer.previousElementSibling;
                        if (headerElement && headerElement.dataset.expanded === 'true') {
                            // 临时设置为none以获取真实高度
                            contentContainer.style.maxHeight = 'none';
                            const actualHeight = contentContainer.scrollHeight;
                            contentContainer.style.maxHeight = actualHeight + 'px';
                        }
                    }
                    
                    // 显示成功通知
                    showNotification(`日志文件 "${filename}" 已成功删除`);
                } else {
                    throw new Error(data.message || '删除失败');
                }
            })
            .catch(error => {
                console.error('删除日志失败:', error);
                showNotification(`删除失败: ${error.message}`, 'error');
                // 恢复按钮状态
                button.textContent = originalText;
                button.disabled = false;
            });
        },
        function() {
            // 用户取消删除，不执行任何操作
        }
    );
}

// 查看日志文件内容
function viewLogFile(filename, button) {
    // 生成安全的文件名，用于元素ID
    const safeFilename = filename.replace(/[^a-zA-Z0-9_.-]/g, '_');
    const idSafeFilename = safeFilename.replace(/\./g, '-');
    
    // 查找对应的日志容器
    const processItem = button.closest('.process-item');
    const logContainer = processItem.querySelector(`#log-content-${idSafeFilename}`);
    
    // 获取内容容器和头部元素
    const contentContainer = processItem.closest('.process-type-content');
    const headerElement = contentContainer ? contentContainer.previousElementSibling : null;
    
    // 确保日志所在的分组是展开状态
    if (contentContainer && headerElement && headerElement.dataset.expanded !== 'true') {
        // 如果分组是折叠的，先展开分组
        toggleGroup(headerElement);
    }
    
    // 切换日志容器的显示状态
    if (logContainer.classList.contains('hidden')) {
        logContainer.classList.remove('hidden');
        
        // 如果是首次加载，获取日志内容
        if (logContainer.textContent.trim() === '') {
            logContainer.textContent = '加载中...';
            
            fetch(`/logfile-content/${encodeURIComponent(filename)}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('获取日志失败');
                    }
                    return response.text();
                })
                .then(logs => {
                    // 保留空白格式，使用textContent而不是innerHTML
                    logContainer.textContent = logs;
                    logContainer.scrollTop = 0;
                    
                    // 更新内容容器的高度
                    updateContentContainerHeight();
                })
                .catch(error => {
                    console.error('获取日志失败:', error);
                    logContainer.innerHTML = `<p class="error">获取日志失败: ${error.message}</p>`;
                    
                    // 更新内容容器的高度
                    updateContentContainerHeight();
                });
        } else {
            // 如果不是首次加载，直接更新内容容器的高度
            updateContentContainerHeight();
        }
    } else {
        logContainer.classList.add('hidden');
        
        // 更新内容容器的高度
        updateContentContainerHeight();
    }
    
    // 辅助函数：更新内容容器的高度
    function updateContentContainerHeight() {
        if (contentContainer && headerElement && headerElement.dataset.expanded === 'true') {
            // 保存当前的maxHeight设置
            const currentMaxHeight = contentContainer.style.maxHeight;
            
            // 暂时设置为none以获取真实高度
            contentContainer.style.maxHeight = 'none';
            
            // 获取真实高度
            const actualHeight = contentContainer.scrollHeight;
            
            // 检查是否需要重新设置高度
            if (currentMaxHeight === 'none' || parseInt(currentMaxHeight) !== actualHeight) {
                contentContainer.style.maxHeight = actualHeight + 'px';
                
                // 短暂延迟后再设置为none，确保高度变化正确应用
                setTimeout(() => {
                    if (headerElement.dataset.expanded === 'true') {
                        contentContainer.style.maxHeight = 'none';
                    }
                }, 300);
            } else {
                // 如果高度没有变化，恢复之前的设置
                contentContainer.style.maxHeight = currentMaxHeight;
            }
        }
    }
}