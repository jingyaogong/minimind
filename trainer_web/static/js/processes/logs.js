import { getLogs } from '../services/apiClient.js';
import { setHidden } from '../utils/dom.js';

const logTimers = new Map();

export function showLogs(processId) {
  const container = document.getElementById(`logs-${processId}`);
  if (!container) return;
  const wasHidden = container.classList.contains('hidden');
  setHidden(container, false);
  if (wasHidden) {
    loadLogContent(processId, container);
    resetTimer(processId, container);
  } else {
    setHidden(container, true);
    clearTimer(processId);
  }
}

export function refreshLog(processId) {
  const container = document.getElementById(`logs-${processId}`);
  if (!container || container.classList.contains('hidden')) return;
  loadLogContent(processId, container);
  resetTimer(processId, container);
}

export function clearLogTimerFor(processId) {
  clearTimer(processId);
}

export function isLogTimerActive(processId) {
  return logTimers.has(processId);
}

function resetTimer(processId, container) {
  clearTimer(processId);
  const item = document.querySelector(`[data-process-id="${processId}"]`);
  const running = item && item.dataset.processStatus === '运行中';
  if (!running) return;
  const id = setInterval(() => {
    if (container.classList.contains('hidden')) {
      clearTimer(processId);
      return;
    }
    const current = document.querySelector(`[data-process-id="${processId}"]`);
    const stillRunning = current && current.dataset.processStatus === '运行中';
    if (stillRunning) loadLogContent(processId, container);
    else clearTimer(processId);
  }, 1000);
  logTimers.set(processId, id);
}

function clearTimer(processId) {
  const id = logTimers.get(processId);
  if (id) {
    clearInterval(id);
    logTimers.delete(processId);
  }
}

function loadLogContent(processId, container) {
  const old = container.textContent;
  const stickBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 10;
  return getLogs(processId)
    .then((logs) => {
      container.textContent = logs;
      if (stickBottom || old === container.textContent) container.scrollTop = container.scrollHeight;
    })
    .catch((err) => {
      if (!container.textContent.includes('加载失败')) container.textContent = `加载日志失败: ${err.message}`;
    });
}

