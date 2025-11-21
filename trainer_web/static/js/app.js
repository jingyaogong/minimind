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

// 文件夹选择器功能 - 支持远程文件浏览
window.selectFolder = async (inputId) => {
  try {
    // 检测是否为远程连接（通过检查主机名或端口）
    const isRemote = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
    
    if (isRemote) {
      // 远程连接：使用服务器端文件浏览
      openRemoteFileBrowser(inputId);
    } else {
      // 本地连接：尝试本地文件系统访问
      await openLocalFileBrowser(inputId);
    }
  } catch (error) {
    console.warn('文件夹选择失败:', error);
    if (error.name !== 'AbortError') {
      // 降级到远程文件浏览
      openRemoteFileBrowser(inputId);
    }
  }
};

// 本地文件浏览器（兼容模式）
async function openLocalFileBrowser(inputId) {
  try {
    // 检查是否支持 File System Access API
    if ('showDirectoryPicker' in window) {
      const dirHandle = await window.showDirectoryPicker();
      const path = dirHandle.name;
      document.getElementById(inputId).value = `./${path}`;
    } else {
      // 降级到远程文件浏览
      await openRemoteFileBrowser(inputId);
    }
  } catch (error) {
    // 如果本地失败，降级到远程文件浏览
    await openRemoteFileBrowser(inputId);
  }
}

// 远程文件浏览器
let currentFileBrowserTarget = null;
let currentBrowsePath = './';

function openRemoteFileBrowser(inputId) {
  currentFileBrowserTarget = inputId;
  document.getElementById('file-browser-modal').classList.remove('hidden');
  
  // 加载初始路径
  loadQuickPaths();
  browsePath('./');
}

function closeFileBrowser() {
  document.getElementById('file-browser-modal').classList.add('hidden');
  currentFileBrowserTarget = null;
  currentBrowsePath = './';
}

function confirmFileSelection() {
  const selectedPath = document.getElementById('selected-path').value;
  if (selectedPath && currentFileBrowserTarget) {
    document.getElementById(currentFileBrowserTarget).value = selectedPath;
  }
  closeFileBrowser();
}

function navigateToParent() {
  if (currentBrowsePath !== './') {
    const parentPath = currentBrowsePath.includes('/') ? 
      currentBrowsePath.substring(0, currentBrowsePath.lastIndexOf('/')) : './';
    browsePath(parentPath || './');
  }
}

async function loadQuickPaths() {
  try {
    const response = await fetch('/api/quick-paths');
    const data = await response.json();
    
    const quickPathsContainer = document.getElementById('quick-paths');
    quickPathsContainer.innerHTML = '';
    
    if (data.paths && data.paths.length > 0) {
      data.paths.forEach(path => {
        const btn = document.createElement('button');
        btn.className = 'quick-path-btn';
        btn.textContent = path.name;
        btn.onclick = () => browsePath(path.path);
        btn.title = path.path;
        quickPathsContainer.appendChild(btn);
      });
    }
  } catch (error) {
    console.warn('加载快捷路径失败:', error);
  }
}

async function browsePath(path) {
  try {
    currentBrowsePath = path;
    document.getElementById('current-path').textContent = path;
    
    const response = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
    const data = await response.json();
    
    if (data.error) {
      alert(`浏览失败: ${data.error}`);
      return;
    }
    
    renderFileList(data);
  } catch (error) {
    console.error('浏览路径失败:', error);
    alert('浏览路径失败，请检查网络连接');
  }
}

function renderFileList(data) {
  const fileList = document.getElementById('file-list');
  fileList.innerHTML = '';
  
  if (!data.items || data.items.length === 0) {
    fileList.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">此目录为空</div>';
    return;
  }
  
  // 先显示目录，再显示文件
  const directories = data.items.filter(item => item.type === 'directory');
  const files = data.items.filter(item => item.type === 'file');
  
  // 渲染目录
  directories.forEach(item => {
    const div = createFileItem(item, '📁');
    fileList.appendChild(div);
  });
  
  // 渲染文件
  files.forEach(item => {
    const div = createFileItem(item, '📄');
    fileList.appendChild(div);
  });
}

function createFileItem(item, icon) {
  const div = document.createElement('div');
  div.className = 'file-item';
  div.onclick = () => selectFileItem(item);
  
  div.innerHTML = `
    <span class="file-icon">${icon}</span>
    <span class="file-name">${item.name}</span>
    <span class="file-info">${item.type === 'file' ? formatFileSize(item.size) : '文件夹'}</span>
  `;
  
  return div;
}

function selectFileItem(item) {
  if (item.type === 'directory') {
    browsePath(item.path);
  } else {
    document.getElementById('selected-path').value = item.path;
    // 文件被选中，可以高亮显示
    document.querySelectorAll('.file-item').forEach(el => el.classList.remove('selected'));
    event.currentTarget.classList.add('selected');
  }
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

window.addEventListener('load', () => {
  initTrainForm();
  startProcessPolling();
  loadProcesses();
});

