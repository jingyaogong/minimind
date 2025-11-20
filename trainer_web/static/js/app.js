import { openTab as _openTab } from './ui/tabs.js';
import { initTrainForm } from './train/form.js';
import { startProcessPolling, stopProcessPolling, loadProcesses } from './processes/list.js';
import { loadLogFiles } from './logfiles/list.js';

const hooks = {
  onEnterProcesses: () => {
    loadProcesses();
  },
  onLeaveProcesses: () => {
    stopProcessPolling();
  },
  onEnterLogfiles: () => {
    loadLogFiles();
  },
};

window.openTab = (evt, tabName) => _openTab(evt, tabName, hooks);

// 文件夹选择器功能
window.selectFolder = async (inputId) => {
  try {
    // 检查是否支持 File System Access API
    if ('showDirectoryPicker' in window) {
      const dirHandle = await window.showDirectoryPicker();
      const path = dirHandle.name; // 使用目录名称作为路径
      document.getElementById(inputId).value = `./${path}`;
    } else {
      // 降级方案：使用文件输入模拟
      const input = document.createElement('input');
      input.type = 'file';
      input.webkitdirectory = true;
      input.onchange = (e) => {
        const files = e.target.files;
        if (files.length > 0) {
          // 提取相对路径
          const path = files[0].webkitRelativePath.split('/')[0];
          document.getElementById(inputId).value = `./${path}`;
        }
      };
      input.click();
    }
  } catch (error) {
    console.warn('文件夹选择失败:', error);
    // 如果用户取消选择，不显示错误
    if (error.name !== 'AbortError') {
      alert('文件夹选择失败，请手动输入路径');
    }
  }
};

window.addEventListener('load', () => {
  initTrainForm();
  startProcessPolling();
  loadProcesses();
});

