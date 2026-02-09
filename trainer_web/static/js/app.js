import { openTab as _openTab } from './ui/tabs.js';
import { initTrainForm } from './train/form.js';
import { startProcessPolling, stopProcessPolling, loadProcesses } from './processes/list.js';
import { loadLogFiles } from './logfiles/list.js';
import { refreshLog } from './processes/logs.js';

const hooks = {
  onEnterProcesses: () => {
    // 当切换到进程标签页时，立即加载一次，然后开始轮询
    loadProcesses().then(() => {
      startProcessPolling();
    });
  },
  onLeaveProcesses: () => {
    stopProcessPolling();
  },
  onEnterLogfiles: () => {
    loadLogFiles();
  },
};

window.openTab = (evt, tabName) => _openTab(evt, tabName, hooks);

// 文件夹选择器功能 - 直接显示服务器端文件浏览器
window.selectFolder = (inputId) => {
  // 直接使用远程文件浏览器，不尝试本地文件系统访问
  openRemoteFileBrowser(inputId);
};

// 远程文件浏览器 - 支持文件和文件夹选择
let currentFileBrowserTarget = null;
let currentBrowsePath = './';
let selectedFilePath = null;
let currentSelectionMode = 'auto'; // 'file', 'folder', or 'auto'

function openRemoteFileBrowser(inputId) {
  console.log('openRemoteFileBrowser called with:', inputId);
  currentFileBrowserTarget = inputId;
  
  // 根据输入框ID确定选择模式
  if (inputId === 'data_path') {
    currentSelectionMode = 'file'; // 数据路径需要文件选择
    console.log('Mode set to: FILE selection');
  } else if (inputId === 'save_dir' || inputId.includes('reward_model_path')) {
    currentSelectionMode = 'folder'; // 保存目录和奖励模型路径需要文件夹选择
    console.log('Mode set to: FOLDER selection');
  } else {
    currentSelectionMode = 'auto'; // 自动模式
    console.log('Mode set to: AUTO selection');
  }
  
  const modal = document.getElementById('file-browser-modal');
  if (modal) {
    modal.classList.remove('hidden');
    console.log('Modal opened successfully');
  } else {
    console.error('Modal element not found!');
    return;
  }
  
  // 重置选择状态
  selectedFilePath = null;
  const selectedPathInput = document.getElementById('selected-path');
  if (selectedPathInput) {
    selectedPathInput.value = '';
    console.log('Selected path input cleared');
  }
  
  // 加载初始路径
  loadQuickPaths();
  browsePath('./');
}

function closeFileBrowser() {
  document.getElementById('file-browser-modal').classList.add('hidden');
  currentFileBrowserTarget = null;
  currentBrowsePath = './';
  selectedFilePath = null;
  currentSelectionMode = 'auto';
}

function confirmFileSelection() {
  console.log('confirmFileSelection called');
  console.log('selectedFilePath:', selectedFilePath);
  console.log('currentFileBrowserTarget:', currentFileBrowserTarget);
  
  if (selectedFilePath && currentFileBrowserTarget) {
    const targetElement = document.getElementById(currentFileBrowserTarget);
    console.log('targetElement:', targetElement);
    
    if (targetElement) {
      targetElement.value = selectedFilePath;
      console.log('Value set successfully');
      closeFileBrowser();
    } else {
      console.error('Target element not found:', currentFileBrowserTarget);
      alert('错误：无法找到目标输入框');
    }
  } else {
    console.log('Missing selection or target');
    alert('请先选择文件或文件夹');
  }
}

function navigateToParent() {
  if (window.currentParentPath) {
    // 使用后端提供的父目录路径（绝对路径）
    browsePath(window.currentParentPath);
  } else if (currentBrowsePath && currentBrowsePath !== './') {
    // 回退到基于当前路径的计算
    const parentPath = currentBrowsePath.includes('/') ? 
      currentBrowsePath.substring(0, currentBrowsePath.lastIndexOf('/')) : './';
    browsePath(parentPath || './');
  }
}

function selectCurrentDirectory() {
  // 选择当前目录
  selectedFilePath = currentBrowsePath;
  document.getElementById('selected-path').value = currentBrowsePath;
  // 可以关闭模态框或让用户继续浏览
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
  console.log('browsePath called with:', path);
  try {
    currentBrowsePath = path;
    selectedFilePath = null; // 重置选中的文件路径
    document.getElementById('selected-path').value = ''; // 清空显示
    
    // 更新帮助文本
    updateHelpText();
    
    const response = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
    const data = await response.json();
    
    if (data.error) {
      alert(`浏览失败: ${data.error}`);
      return;
    }
    
    renderFileList(data);
    console.log('File list rendered successfully');
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
  
  // 更新当前路径显示（使用相对路径用于显示）
  document.getElementById('current-path').textContent = data.relative_path || data.current_path;
  
  // 存储父目录路径供导航使用
  window.currentParentPath = data.parent;
  
  // 先显示目录，再显示文件
  const directories = data.items.filter(item => item.type === 'directory');
  const files = data.items.filter(item => item.type === 'file');
  
  // 渲染目录
  directories.forEach(item => {
    const div = createFileItem(item, '📁');
    fileList.appendChild(div);
  });
  
  // 渲染文件（仅在文件选择模式或自动模式下显示）
  if (currentSelectionMode !== 'folder') {
    files.forEach(item => {
      const div = createFileItem(item, '📄');
      fileList.appendChild(div);
    });
  }
}

function createFileItem(item, icon) {
  const div = document.createElement('div');
  div.className = 'file-item';
  
  // 根据选择模式添加适当的CSS类
  if (currentSelectionMode === 'file' && item.type === 'directory') {
    // 文件选择模式下，文件夹只用于导航，不能选择
    div.classList.add('navigable');
  } else if (currentSelectionMode === 'folder' && item.type === 'file') {
    // 文件夹选择模式下，文件不能被选择
    div.classList.add('disabled');
  }
  
  div.onclick = (event) => selectFileItem(item, event);
  
  div.innerHTML = `
    <span class="file-icon">${icon}</span>
    <span class="file-name">${item.name}</span>
    <span class="file-info">${item.type === 'file' ? formatFileSize(item.size) : '文件夹'}</span>
  `;
  
  return div;
}

function selectFileItem(item, event) {
  console.log('selectFileItem called with:', item);
  console.log('currentSelectionMode:', currentSelectionMode);
  console.log('event:', event);
  
  // 检查是否点击了被禁用的项目
  if (event && event.currentTarget && event.currentTarget.classList.contains('disabled')) {
    console.log('Clicked disabled item, ignoring');
    return;
  }
  
  if (item.type === 'directory') {
    // 文件夹：根据选择模式决定行为
    if (currentSelectionMode === 'file') {
      // 文件选择模式：只能选择文件，点击进入目录
      console.log('File mode: navigating into directory');
      browsePath(item.path);
    } else if (currentSelectionMode === 'folder') {
      // 文件夹选择模式：可以选择文件夹
      console.log('Folder mode: selecting directory');
      selectedFilePath = item.path;
      document.getElementById('selected-path').value = item.path;
      // 高亮显示选中的文件夹
      document.querySelectorAll('.file-item').forEach(el => el.classList.remove('selected'));
      if (event && event.currentTarget) {
        event.currentTarget.classList.add('selected');
      }
      console.log('Directory selected:', selectedFilePath);
    } else {
      // 自动模式：点击进入目录
      console.log('Auto mode: navigating into directory');
      browsePath(item.path);
    }
  } else {
    // 文件：选中文件路径（仅在选择文件或自动模式下）
    if (currentSelectionMode !== 'folder') {
      console.log('Selecting file:', item.path);
      selectedFilePath = item.path;
      document.getElementById('selected-path').value = item.path;
      // 高亮显示选中的文件
      document.querySelectorAll('.file-item').forEach(el => el.classList.remove('selected'));
      if (event && event.currentTarget) {
        event.currentTarget.classList.add('selected');
      }
      console.log('File selected:', selectedFilePath);
    } else {
      console.log('File clicked in folder mode, ignoring');
    }
  }
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function updateHelpText() {
  const helpText = document.querySelector('.file-browser-help');
  const modalTitle = document.getElementById('modal-title');
  
  if (!helpText) return;
  
  let text = '';
  let title = '';
  switch (currentSelectionMode) {
    case 'file':
      text = '💡 请选择文件：点击文件选择，点击文件夹进入目录，使用📍选择当前目录';
      title = '选择文件';
      break;
    case 'folder':
      text = '💡 请选择文件夹：点击文件夹选择，点击文件无效，使用📍选择当前目录';
      title = '选择文件夹';
      break;
    default:
      text = '💡 点击文件夹进入目录，点击文件选择文件，使用📍选择当前目录';
      title = '选择文件或文件夹';
  }
  helpText.textContent = text;
  if (modalTitle) {
    modalTitle.textContent = title;
  }
}

// 添加模态框键盘事件监听
document.addEventListener('keydown', function(event) {
  if (event.key === 'Escape') {
    closeFileBrowser();
  }
});

// 添加模态框点击外部关闭功能
document.addEventListener('DOMContentLoaded', function() {
  const modal = document.getElementById('file-browser-modal');
  if (modal) {
    modal.addEventListener('click', function(event) {
      if (event.target === modal) {
        closeFileBrowser();
      }
    });
  }
});

// 将文件浏览器函数暴露到全局作用域
window.selectFolder = selectFolder;
window.openRemoteFileBrowser = openRemoteFileBrowser;
window.closeFileBrowser = closeFileBrowser;
window.confirmFileSelection = confirmFileSelection;
window.navigateToParent = navigateToParent;
window.selectCurrentDirectory = selectCurrentDirectory;

// 将进程管理函数暴露到全局作用域
window.refreshProcesses = () => {
  // 立即刷新进程数据，然后重置轮询计时器
  return loadProcesses().then(() => {
    // 重置轮询计时器以确保平滑的更新间隔
    stopProcessPolling();
    startProcessPolling();
  });
};
window.refreshLogs = loadLogFiles;
window.refreshLog = refreshLog;

window.addEventListener('load', () => {
  initTrainForm();
  // 不再立即开始轮询，而是等待用户切换到进程标签页
  // startProcessPolling(); // 移动到钩子函数中
  loadProcesses(); // 仍然加载初始进程数据
});

