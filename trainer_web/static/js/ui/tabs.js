import { qsa } from '../utils/dom.js';

export function openTab(evt, tabName, hooks = {}) {
  const contents = qsa('.tab-content');
  contents.forEach((c) => c.classList.add('hidden'));
  const tabs = qsa('.tab');
  tabs.forEach((t) => t.classList.remove('active'));
  const target = document.getElementById(tabName);
  if (target) target.classList.remove('hidden');
  if (evt && evt.currentTarget) evt.currentTarget.classList.add('active');
  if (tabName !== 'processes' && hooks.onLeaveProcesses) hooks.onLeaveProcesses();
  if (tabName === 'processes' && hooks.onEnterProcesses) hooks.onEnterProcesses();
  if (tabName === 'logfiles' && hooks.onEnterLogfiles) hooks.onEnterLogfiles();
}

