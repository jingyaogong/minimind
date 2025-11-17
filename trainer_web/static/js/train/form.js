import { startTrain } from '../services/apiClient.js';
import { showNotification } from '../ui/notify.js';

export function initTrainForm() {
  const typeSel = document.getElementById('train_type');
  if (typeSel) {
    typeSel.addEventListener('change', onTrainTypeChange);
    typeSel.dispatchEvent(new Event('change'));
  }
  initGpuSelectors();
  const form = document.getElementById('train-form');
  if (form) form.addEventListener('submit', onSubmit);
}

function onTrainTypeChange() {
  const v = this.value;
  const pretrainSft = document.querySelectorAll('.pretrain-sft');
  const fromWeightFields = document.querySelectorAll('.from-weight');
  const loraFields = document.querySelectorAll('.lora');
  const dpoFields = document.querySelectorAll('.dpo');
  const dpoCard = document.querySelector('.parameter-card.dpo');
  const ppoFields = document.querySelectorAll('.ppo');
  const ppoCard = document.querySelector('.parameter-card.ppo');
  const grpoFields = document.querySelectorAll('.grpo');
  const grpoCard = document.querySelector('.parameter-card.grpo');
  const spoFields = document.querySelectorAll('.spo');
  const spoCard = document.querySelector('.parameter-card.spo');
  pretrainSft.forEach((f) => (f.style.display = v === 'pretrain' || v === 'sft' || v === 'dpo' || v === 'ppo' || v === 'grpo' || v === 'spo' ? 'block' : 'none'));
  fromWeightFields.forEach((f) => (f.style.display = v !== 'ppo' && v !== 'grpo' && v !== 'spo' ? 'block' : 'none'));
  loraFields.forEach((f) => (f.style.display = v === 'lora' ? 'block' : 'none'));
  dpoFields.forEach((f) => (f.style.display = v === 'dpo' ? 'block' : 'none'));
  ppoFields.forEach((f) => (f.style.display = v === 'ppo' ? 'block' : 'none'));
  if (dpoCard) dpoCard.style.display = v === 'dpo' ? 'block' : 'none';
  if (ppoCard) ppoCard.style.display = v === 'ppo' ? 'block' : 'none';
  grpoFields.forEach((f) => (f.style.display = v === 'grpo' ? 'block' : 'none'));
  spoFields.forEach((f) => (f.style.display = v === 'spo' ? 'block' : 'none'));
  if (grpoCard) grpoCard.style.display = v === 'grpo' ? 'block' : 'none';
  if (spoCard) spoCard.style.display = v === 'spo' ? 'block' : 'none';
  if (v === 'pretrain') setDefaults({ save_dir: '../out', save_weight: 'pretrain', epochs: '1', batch_size: '32', learning_rate: '5e-4', data_path: '../dataset/pretrain_hq.jsonl', from_weight: 'none', log_interval: '100', save_interval: '100', hidden_size: '512', num_hidden_layers: '8', max_seq_len: '512', use_moe: '0' });
  else if (v === 'sft') setDefaults({ save_dir: '../out', save_weight: 'full_sft', epochs: '2', batch_size: '16', learning_rate: '5e-7', data_path: '../dataset/sft_mini_512.jsonl', from_weight: 'pretrain', log_interval: '100', save_interval: '100', hidden_size: '512', num_hidden_layers: '8', max_seq_len: '512', use_moe: '0' });
  else if (v === 'lora') setDefaults({ save_dir: '../out/lora', lora_name: 'lora_identity', epochs: '50', batch_size: '32', learning_rate: '1e-4', data_path: '../dataset/lora_identity.jsonl', from_weight: 'full_sft', log_interval: '10', save_interval: '1', hidden_size: '512', num_hidden_layers: '8', max_seq_len: '512', use_moe: '0' });
  else if (v === 'dpo') setDefaults({ save_dir: '../out', save_weight: 'dpo', epochs: '1', batch_size: '4', learning_rate: '4e-8', data_path: '../dataset/dpo.jsonl', from_weight: 'full_sft', log_interval: '100', save_interval: '100', beta: '0.1', hidden_size: '512', num_hidden_layers: '8', max_seq_len: '1024', use_moe: '0' });
  else if (v === 'ppo') setDefaults({ save_dir: '../out', save_weight: 'ppo_actor', epochs: '1', batch_size: '2', learning_rate: '8e-8', data_path: '../dataset/rlaif-mini.jsonl', log_interval: '1', save_interval: '10', clip_epsilon: '0.1', vf_coef: '0.5', kl_coef: '0.02', reasoning: '1', update_old_actor_freq: '4', reward_model_path: '../../internlm2-1_8b-reward', hidden_size: '512', num_hidden_layers: '8', max_seq_len: '66', use_moe: '0' });
  else if (v === 'grpo') setDefaults({ save_dir: '../out', save_weight: 'grpo', epochs: '1', batch_size: '2', learning_rate: '8e-8', data_path: '../dataset/rlaif-mini.jsonl', log_interval: '1', save_interval: '10', beta: '0.02', num_generations: '8', reasoning: '1', reward_model_path: '../../internlm2-1_8b-reward', hidden_size: '512', num_hidden_layers: '8', max_seq_len: '66', use_moe: '0' });
  else if (v === 'spo') setDefaults({ save_dir: '../out', save_weight: 'spo', epochs: '1', batch_size: '2', learning_rate: '1e-7', data_path: '../dataset/rlaif-mini.jsonl', log_interval: '1', save_interval: '10', beta: '0.02', reasoning: '1', reward_model_path: '../../internlm2-1_8b-reward', hidden_size: '512', num_hidden_layers: '8', max_seq_len: '66', use_moe: '0' });
}

function setDefaults(map) {
  Object.entries(map).forEach(([name, val]) => {
    const nodes = document.querySelectorAll(`[name="${name}"]`);
    nodes.forEach((node) => {
      const card = node.closest('.parameter-card');
      const visible = !card || card.style.display !== 'none';
      if (visible) node.value = val;
    });
  });
}

function initGpuSelectors() {
  const hasGpu = window.hasGpu === true;
  const gpuCount = Number(window.gpuCount || 0);
  const modeSel = document.getElementById('training_mode');
  const single = document.getElementById('single-gpu-selection');
  const multi = document.getElementById('multi-gpu-selection');
  if (!modeSel) return;
  function updateVisibility() {
    const mode = modeSel.value;
    if (single) single.style.display = mode === 'single_gpu' ? 'block' : 'none';
    if (multi) multi.style.display = mode === 'multi_gpu' ? 'block' : 'none';
  }
  if (!hasGpu) {
    modeSel.value = 'cpu';
    if (single) single.style.display = 'none';
    if (multi) multi.style.display = 'none';
  } else {
    const gpuNumInput = document.getElementById('gpu_num');
    if (gpuNumInput && gpuCount > 0) gpuNumInput.value = gpuCount;
  }
  updateVisibility();
  modeSel.addEventListener('change', updateVisibility);
}

function onSubmit(e) {
  e.preventDefault();
  const form = e.currentTarget;
  const data = {};
  const trainingModeSel = form.querySelector('#training_mode');
  const trainingMode = trainingModeSel ? trainingModeSel.value : 'cpu';
  const inputs = form.querySelectorAll('input, select, textarea');
  inputs.forEach((el) => {
    const name = el.name;
    if (!name || name === 'training_mode') return;
    const card = el.closest('.parameter-card');
    const visible = !card || card.style.display !== 'none';
    if (!visible) return;
    let value = el.value;
    if (el.type === 'checkbox') {
      if (!el.checked) return;
    }
    if (name === 'gpu_num') {
      const multi = document.getElementById('multi-gpu-selection');
      if (!(multi && multi.style.display !== 'none')) return;
    }
    if (name === 'device') {
      if (trainingMode === 'single_gpu') value = `cuda:${value}`;
      else if (trainingMode === 'cpu') value = 'cpu';
      else return;
    }
    data[name] = value;
  });
  showNotification('正在启动训练...', 'info');
  setTimeout(() => {
    startTrain(data)
      .then((result) => {
        if (result.success) {
          showNotification('训练已开始！', 'success');
          setTimeout(() => {
            const processTab = document.querySelector('.tab[onclick*="processes"]');
            if (processTab) processTab.click();
          }, 1000);
        } else showNotification('训练启动失败：' + result.error, 'error');
      })
      .catch(() => {
        showNotification('启动训练中，请耐心等待...', 'info');
      });
  }, 1000);
}

