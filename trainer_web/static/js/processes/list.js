import { getProcesses, stopProcess as apiStop, deleteProcess as apiDelete } from '../services/apiClient.js';
import { showNotification } from '../ui/notify.js';
import { showConfirmDialog } from '../ui/dialog.js';
import { el, clearChildren } from '../utils/dom.js';
import { showLogs, refreshLog, clearLogTimerFor } from './logs.js';

let processPollingTimer = null;

export function startProcessPolling() {
  if (processPollingTimer) clearInterval(processPollingTimer);
  processPollingTimer = setInterval(() => {
    const tab = document.querySelector('.tab.active');
    if (tab && tab.textContent.includes('进程')) checkProcessStatusChanges();
  }, 5000);
}

export function stopProcessPolling() {
  if (processPollingTimer) {
    clearInterval(processPollingTimer);
    processPollingTimer = null;
  }
}

export function checkProcessStatusChanges() {
  return getProcesses()
    .then((data) => {
      data.forEach((p) => {
        const item = document.querySelector(`[data-process-id="${p.id}"]`);
        if (!item) return;
        const cur = item.dataset.processStatus;
        const next = p.status;
        if (cur !== next) {
          updateProcessItem(item, p);
          if (next === '出错') showNotification(`进程 ${p.train_type} 已出错`, 'error');
        }
      });
    })
    .catch(() => {
      showNotification('连接服务器失败，请刷新页面重试', 'error');
    });
}

export function loadProcesses() {
  return getProcesses().then((data) => {
    const list = document.getElementById('process-list');
    clearChildren(list);
    if (data.length === 0) {
      list.innerHTML = '<p>暂无训练进程</p>';
      return;
    }
    data.sort((a, b) => new Date(b.start_time) - new Date(a.start_time));
    const groups = {};
    data.forEach((p) => {
      if (!groups[p.train_type]) groups[p.train_type] = [];
      groups[p.train_type].push(p);
    });
    const order = ['pretrain', 'sft', 'lora', 'dpo'];
    const types = [...order.filter((t) => groups[t]), ...Object.keys(groups).filter((t) => !order.includes(t))];
    types.forEach((t) => {
      const g = createTypeGroupWithToggle(t, groups[t]);
      list.appendChild(g);
    });
  });
}

function createTypeGroupWithToggle(trainType, processes) {
  const group = el('div', { class: 'process-type-group' });
  const header = el('div', { class: 'process-type-header' });
  header.dataset.expanded = 'true';
  const title = el('h3', { class: 'process-type-title', text: getTrainTypeDisplayName(trainType) });
  const toggle = el('button', { class: 'toggle-btn' });
  toggle.innerHTML = '▼';
  toggle.onclick = (e) => {
    e.stopPropagation();
    toggleGroup(header);
  };
  header.appendChild(title);
  header.appendChild(toggle);
  header.onclick = () => toggleGroup(header);
  const content = el('div', { class: 'process-type-content' });
  processes.forEach((p) => addProcessItemToGroup(content, p));
  group.appendChild(header);
  group.appendChild(content);
  return group;
}

function toggleGroup(header) {
  const expanded = header.dataset.expanded === 'true';
  const content = header.nextElementSibling;
  const toggle = header.querySelector('.toggle-btn');
  if (expanded) {
    header.dataset.expanded = 'false';
    content.style.maxHeight = '0';
    content.style.overflow = 'hidden';
    toggle.innerHTML = '▶';
  } else {
    content.style.overflow = 'hidden';
    content.style.maxHeight = 'none';
    const h = content.scrollHeight;
    content.style.maxHeight = '0';
    content.offsetHeight;
    header.dataset.expanded = 'true';
    content.style.maxHeight = h + 'px';
    setTimeout(() => {
      content.style.maxHeight = 'none';
      content.style.overflow = 'visible';
    }, 300);
    toggle.innerHTML = '▼';
  }
}

function getTrainTypeDisplayName(trainType) {
  const names = {
    pretrain: '预训练 (Pretrain)',
    sft: '全参数监督微调 (SFT - Full)',
    lora: 'LoRA监督微调 (SFT - Lora)',
    dpo: '直接偏好优化 (RL - DPO)',
    ppo: 'PPO',
    grpo: 'GRPO',
    spo: 'SPO',
  };
  return names[trainType] || trainType;
}

export function addProcessItemToGroup(parent, process) {
  const item = el('div', { class: 'process-item' });
  let statusClass = 'status-completed';
  if (process.status === '运行中') statusClass = 'status-running';
  else if (process.status === '手动停止') statusClass = 'status-manual-stop';
  else if (process.status === '出错') statusClass = 'status-error';
  item.dataset.processId = process.id;
  item.dataset.processStatus = process.status;
  item.dataset.trainMonitor = process.train_monitor || 'none';
  item.dataset.swanlabUrl = process.swanlab_url || '';
  const showDelete = !process.running;
  const showSwanlab = process.train_monitor !== 'none';
  const swanBtn = showSwanlab ? `<button class="btn-swanlab" data-swan="${process.id}">SwanLab</button>` : '';
  item.innerHTML = `
    <div class="process-info">
      <div><strong>${process.start_time}</strong></div>
      <div><span class="process-status ${statusClass}">${process.status}</span></div>
    </div>
    <div>
      <button class="btn-logs" data-show="${process.id}">查看日志</button>
      <button class="btn-logs" data-refresh="${process.id}">刷新日志</button>
      ${swanBtn}
      ${process.running ? `<button class="btn-stop" data-stop="${process.id}">停止训练</button>` : ''}
      ${showDelete ? `<button class="btn-delete" data-del="${process.id}">删除</button>` : ''}
    </div>
    <div id="logs-${process.id}" class="logs-container hidden"></div>
  `;
  parent.appendChild(item);
  bindItemButtons(item, process);
}

function bindItemButtons(item, process) {
  const showBtn = item.querySelector('[data-show]');
  if (showBtn) showBtn.addEventListener('click', () => showLogs(process.id));
  const refreshBtn = item.querySelector('[data-refresh]');
  if (refreshBtn) refreshBtn.addEventListener('click', () => refreshLog(process.id));
  const swanBtn = item.querySelector('[data-swan]');
  if (swanBtn) swanBtn.addEventListener('click', () => checkAndOpenSwanlab(process.id));
  const stopBtn = item.querySelector('[data-stop]');
  if (stopBtn) stopBtn.addEventListener('click', () => stopProcess(process.id));
  const delBtn = item.querySelector('[data-del]');
  if (delBtn) delBtn.addEventListener('click', () => deleteProcess(process.id));
}

export function updateProcessItem(item, process) {
  item.dataset.processStatus = process.status;
  item.dataset.trainMonitor = process.train_monitor || 'none';
  if (process.swanlab_url) item.dataset.swanlabUrl = process.swanlab_url;
  const statusEl = item.querySelector('.process-status');
  if (statusEl) {
    statusEl.classList.remove('status-running', 'status-manual-stop', 'status-error', 'status-completed');
    let cls = 'status-completed';
    if (process.status === '运行中') cls = 'status-running';
    else if (process.status === '手动停止') cls = 'status-manual-stop';
    else if (process.status === '出错') cls = 'status-error';
    statusEl.classList.add(cls);
    statusEl.textContent = process.status;
  }
  const btnContainer = item.querySelector('div:nth-child(2)');
  const existingSwan = item.querySelector('.btn-swanlab');
  const showSwan = process.train_monitor !== 'none';
  if (showSwan && !existingSwan && btnContainer) {
    const b = el('button', { class: 'btn-swanlab' });
    b.textContent = 'SwanLab';
    b.onclick = () => checkAndOpenSwanlab(process.id);
    const stop = btnContainer.querySelector('.btn-stop');
    if (stop) btnContainer.insertBefore(b, stop);
    else btnContainer.appendChild(b);
  } else if (!showSwan && existingSwan) existingSwan.remove();
  const stopBtn = item.querySelector('.btn-stop');
  if (stopBtn) {
    if (!process.running) stopBtn.remove();
  } else if (process.running && btnContainer) {
    const n = el('button', { class: 'btn-stop' });
    n.textContent = '停止训练';
    n.onclick = () => stopProcess(process.id);
    btnContainer.appendChild(n);
  }
  const delBtn = item.querySelector('.btn-delete');
  if (!process.running) {
    if (!delBtn) {
      const c = item.querySelector('div:last-child');
      if (c) {
        const d = el('button', { class: 'btn-delete' });
        d.textContent = '删除';
        d.onclick = () => deleteProcess(process.id);
        c.appendChild(d);
      }
    }
  } else if (delBtn) delBtn.remove();
  if (!process.running) clearLogTimerFor(process.id);
}

export function deleteProcess(processId) {
  showConfirmDialog('确定要删除这个训练进程吗？此操作不可恢复。', () => {
    apiDelete(processId)
      .then(() => {
        const item = document.querySelector(`[data-process-id="${processId}"]`);
        if (item && item.parentNode) {
          item.style.transition = 'opacity 0.3s, transform 0.3s';
          item.style.opacity = '0';
          item.style.transform = 'translateX(-20px)';
          setTimeout(() => {
            const content = item.closest('.process-type-content');
            const group = content ? content.closest('.process-type-group') : null;
            item.parentNode.removeChild(item);
            if (content) {
              const remain = content.querySelectorAll('.process-item');
              if (remain.length === 0 && group) {
                setTimeout(() => {
                  group.style.transition = 'opacity 0.3s, transform 0.3s';
                  group.style.opacity = '0';
                  group.style.transform = 'translateY(-10px)';
                  setTimeout(() => {
                    if (group.parentNode) group.parentNode.removeChild(group);
                    const left = document.querySelectorAll('.process-item');
                    if (left.length === 0) {
                      const list = document.getElementById('process-list');
                      list.innerHTML = '<p>暂无训练进程</p>';
                    }
                  }, 300);
                }, 100);
              } else {
                const header = content.previousElementSibling;
                if (header && header.dataset.expanded === 'true') content.style.maxHeight = content.scrollHeight + 'px';
                const left = document.querySelectorAll('.process-item');
                if (left.length === 0) {
                  const list = document.getElementById('process-list');
                  list.innerHTML = '<p>暂无训练进程</p>';
                }
              }
            }
          }, 300);
        }
        clearLogTimerFor(processId);
        showNotification('训练进程已删除', 'success');
      })
      .catch(() => {
        showNotification('删除进程失败，请刷新页面重试', 'error');
      });
  });
}

export function stopProcess(processId) {
  showConfirmDialog('确定要停止这个训练进程吗？', () => {
    apiStop(processId)
      .then(() => {
        const item = document.querySelector(`[data-process-id="${processId}"]`);
        if (item) {
          item.dataset.processStatus = '手动停止';
          const statusEl = item.querySelector('.process-status');
          if (statusEl) {
            statusEl.classList.remove('status-running', 'status-error', 'status-completed');
            statusEl.classList.add('status-manual-stop');
            statusEl.textContent = '手动停止';
          }
          const stopBtn = item.querySelector('.btn-stop');
          if (stopBtn) stopBtn.remove();
          clearLogTimerFor(processId);
        }
        showNotification('训练进程已停止', 'info');
        getProcesses()
          .then((data) => {
            const updated = data.find((p) => p.id === processId);
            if (updated && item) updateProcessItem(item, updated);
          })
          .catch(() => {});
      })
      .catch(() => {
        showNotification('停止进程失败', 'error');
      });
  }, () => {
    showNotification('已取消停止操作', 'info');
  });
}

export function checkAndOpenSwanlab(processId) {
  const item = document.querySelector(`[data-process-id="${processId}"]`);
  const monitor = item ? item.dataset.trainMonitor : 'none';
  if (monitor === 'none') {
    showNotification('此训练未启用监控功能', 'info');
    return;
  }
  let url = item ? item.dataset.swanlabUrl : '';
  if (!url || url.trim() === '') {
    getProcesses()
      .then((data) => {
        const p = data.find((x) => x.id === processId);
        if (p && p.swanlab_url) {
          url = p.swanlab_url;
          if (item) item.dataset.swanlabUrl = url;
          openSwanlab(url);
        } else {
          showNotification('SwanLab链接尚未生成，请稍后再试', 'info');
        }
      })
      .catch(() => {
        showNotification('获取SwanLab链接失败，请稍后再试', 'error');
      });
  } else openSwanlab(url);
}

function openSwanlab(url) {
  if (!isValidUrl(url)) {
    showNotification('SwanLab链接无效或尚未生成', 'info');
    return;
  }
  const w = window.open(url, '_blank');
  if (w) showNotification('正在打开SwanLab页面', 'info');
  else showNotification('无法打开新窗口，请检查浏览器设置', 'error');
}

function isValidUrl(url) {
  try {
    new URL(url);
    return true;
  } catch {
    const u = String(url).toLowerCase();
    return u.startsWith('http://') || u.startsWith('https://');
  }
}

