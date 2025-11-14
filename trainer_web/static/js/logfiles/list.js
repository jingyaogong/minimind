import { getLogFiles, getLogFileContent, deleteLogFile as apiDeleteLogFile } from '../services/apiClient.js';
import { el } from '../utils/dom.js';
import { showNotification } from '../ui/notify.js';
import { showConfirmDialog } from '../ui/dialog.js';

export function loadLogFiles() {
  return getLogFiles().then((data) => {
    const list = document.getElementById('logfiles-list');
    list.innerHTML = '';
    if (data.length === 0) {
      list.innerHTML = '<p>暂无日志文件</p>';
      return;
    }
    data.sort((a, b) => new Date(b.modified_time) - new Date(a.modified_time));
    const groups = {};
    data.forEach((f) => {
      let type = '自定义训练';
      const n = f.filename;
      if (n.includes('train_pretrain_')) type = 'pretrain';
      else if (n.includes('train_sft_')) type = 'sft';
      else if (n.includes('train_lora_')) type = 'lora';
      else if (n.includes('train_dpo_')) type = 'dpo';
      else if (n.includes('train_ppo_')) type = 'ppo';
      else if (n.includes('train_grpo_')) type = 'grpo';
      else if (n.includes('train_spo_')) type = 'spo';
      f.train_type = type;
      if (!groups[type]) groups[type] = [];
      groups[type].push(f);
    });
    const order = ['pretrain', 'sft', 'lora', 'dpo', 'ppo', 'grpo', 'spo', '未知'];
    [...order.filter((t) => groups[t]), ...Object.keys(groups).filter((t) => !order.includes(t))].forEach((t) => {
      list.appendChild(createTypeGroupWithToggle(t, groups[t]));
    });
  });
}

function createTypeGroupWithToggle(trainType, files) {
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
  files.forEach((f) => addLogFileItemToGroup(content, f));
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

function addLogFileItemToGroup(parent, logfile) {
  const item = el('div', { class: 'process-item' });
  item.innerHTML = `
    <div class="process-info">
      <div><strong>${logfile.filename}</strong></div>
      <div>
        <span class="process-status status-completed">已保存</span>
        <span style="margin-left: 10px; color: #999; font-size: 0.9em;">${logfile.modified_time}</span>
      </div>
    </div>
    <div>
      <button class="btn-logs" data-view="${logfile.filename}">查看日志</button>
      <button class="btn-delete" data-del="${logfile.filename}">删除</button>
    </div>
    <div id="log-content-${logfile.filename.replace(/\./g, '-') }" class="logs-container hidden"></div>
  `;
  parent.appendChild(item);
  bindItemButtons(item, logfile);
}

function bindItemButtons(item, logfile) {
  const viewBtn = item.querySelector('[data-view]');
  if (viewBtn) viewBtn.addEventListener('click', () => viewLogFile(logfile.filename, viewBtn));
  const delBtn = item.querySelector('[data-del]');
  if (delBtn) delBtn.addEventListener('click', () => deleteLogFile(logfile.filename, delBtn));
}

function deleteLogFile(filename, button) {
  showConfirmDialog(`确定要删除日志文件 "${filename}" 吗？此操作无法恢复。`, () => {
    const item = button.closest('.process-item');
    const content = item.closest('.process-type-content');
    const group = content.closest('.process-type-group');
    const original = button.textContent;
    button.textContent = '删除中...';
    button.disabled = true;
    apiDeleteLogFile(filename)
      .then((data) => {
        if (data.success) {
          item.remove();
          if (content.children.length === 0) group.remove();
          else {
            const header = content.previousElementSibling;
            if (header && header.dataset.expanded === 'true') {
              content.style.maxHeight = 'none';
              const h = content.scrollHeight;
              content.style.maxHeight = h + 'px';
            }
          }
          showNotification(`日志文件 "${filename}" 已成功删除`);
        } else throw new Error(data.message || '删除失败');
      })
      .catch((e) => {
        showNotification(`删除失败: ${e.message}`, 'error');
        button.textContent = original;
        button.disabled = false;
      });
  });
}

function viewLogFile(filename, button) {
  const safe = filename.replace(/[^a-zA-Z0-9_.-]/g, '_').replace(/\./g, '-');
  const item = button.closest('.process-item');
  const container = item.querySelector(`#log-content-${safe}`);
  const content = item.closest('.process-type-content');
  const header = content ? content.previousElementSibling : null;
  if (content && header && header.dataset.expanded !== 'true') toggleGroup(header);
  if (container.classList.contains('hidden')) {
    container.classList.remove('hidden');
    container.textContent = '加载中...';
    getLogFileContent(filename)
      .then((logs) => {
        container.textContent = logs;
        container.scrollTop = 0;
        updateContentHeight(content, header);
      })
      .catch((e) => {
        container.textContent = `获取日志失败: ${e.message}`;
        updateContentHeight(content, header);
      });
  } else {
    container.classList.add('hidden');
    updateContentHeight(content, header);
  }
}

function updateContentHeight(content, header) {
  if (content && header && header.dataset.expanded === 'true') {
    const current = content.style.maxHeight;
    content.style.maxHeight = 'none';
    const h = content.scrollHeight;
    if (current === 'none' || parseInt(current) !== h) {
      content.style.maxHeight = h + 'px';
      setTimeout(() => {
        if (header.dataset.expanded === 'true') content.style.maxHeight = 'none';
      }, 300);
    } else content.style.maxHeight = current;
  }
}

