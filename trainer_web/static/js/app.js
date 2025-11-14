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

window.addEventListener('load', () => {
  initTrainForm();
  startProcessPolling();
  loadProcesses();
});

