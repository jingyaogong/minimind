import sys
import os

def build_command(train_type, params, gpu_num, use_torchrun):
    if train_type == 'pretrain':
        script_path = '../trainer/train_pretrain.py'
        cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path] if use_torchrun else [sys.executable, script_path]
        if 'save_weight' in params:
            cmd.extend(['--save_weight', params['save_weight']])
    elif train_type == 'sft':
        script_path = '../trainer/train_full_sft.py'
        cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path] if use_torchrun else [sys.executable, script_path]
        if 'save_weight' in params:
            cmd.extend(['--save_weight', params['save_weight']])
    elif train_type == 'lora':
        script_path = '../trainer/train_lora.py'
        cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path] if use_torchrun else [sys.executable, script_path]
        if 'lora_name' in params:
            cmd.extend(['--lora_name', params['lora_name']])
    elif train_type == 'dpo':
        script_path = '../trainer/train_dpo.py'
        cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path] if use_torchrun else [sys.executable, script_path]
        if 'beta' in params and params['beta']:
            cmd.extend(['--beta', params['beta']])
        if 'accumulation_steps' in params and params['accumulation_steps']:
            cmd.extend(['--accumulation_steps', params['accumulation_steps']])
        if 'grad_clip' in params and params['grad_clip']:
            cmd.extend(['--grad_clip', params['grad_clip']])
    elif train_type == 'ppo':
        script_path = '../trainer/train_ppo.py'
        cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path] if use_torchrun else [sys.executable, script_path]
        if 'clip_epsilon' in params and params['clip_epsilon']:
            cmd.extend(['--clip_epsilon', params['clip_epsilon']])
        if 'vf_coef' in params and params['vf_coef']:
            cmd.extend(['--vf_coef', params['vf_coef']])
        if 'kl_coef' in params and params['kl_coef']:
            cmd.extend(['--kl_coef', params['kl_coef']])
        if 'reasoning' in params and params['reasoning']:
            cmd.extend(['--reasoning', params['reasoning']])
        if 'update_old_actor_freq' in params and params['update_old_actor_freq']:
            cmd.extend(['--update_old_actor_freq', params['update_old_actor_freq']])
        if 'reward_model_path' in params and params['reward_model_path']:
            cmd.extend(['--reward_model_path', params['reward_model_path']])
    elif train_type == 'grpo':
        script_path = '../trainer/train_grpo.py'
        cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path] if use_torchrun else [sys.executable, script_path]
        if 'beta' in params and params['beta']:
            cmd.extend(['--beta', params['beta']])
        if 'num_generations' in params and params['num_generations']:
            cmd.extend(['--num_generations', params['num_generations']])
        if 'reasoning' in params and params['reasoning']:
            cmd.extend(['--reasoning', params['reasoning']])
        if 'reward_model_path' in params and params['reward_model_path']:
            cmd.extend(['--reward_model_path', params['reward_model_path']])
    elif train_type == 'spo':
        script_path = '../trainer/train_spo.py'
        cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path] if use_torchrun else [sys.executable, script_path]
        if 'beta' in params and params['beta']:
            cmd.extend(['--beta', params['beta']])
        if 'reasoning' in params and params['reasoning']:
            cmd.extend(['--reasoning', params['reasoning']])
        if 'reward_model_path' in params and params['reward_model_path']:
            cmd.extend(['--reward_model_path', params['reward_model_path']])
    else:
        return None

    for key, value in params.items():
        if key in ['train_type', 'save_weight', 'lora_name', 'train_monitor', 'beta', 'accumulation_steps', 'grad_clip', 'gpu_num', 'clip_epsilon', 'vf_coef', 'kl_coef', 'reasoning', 'update_old_actor_freq', 'reward_model_path', 'num_generations'] or ((train_type == 'ppo' or train_type == 'grpo' or train_type == 'spo') and key == 'from_weight'):
            continue
        elif key == 'from_resume':
            cmd.extend([f'--{key}', str(value)])
        else:
            cmd.extend([f'--{key}', str(value)])

    if 'train_monitor' in params:
        if params['train_monitor'] == 'wandb' or params['train_monitor'] == 'swanlab':
            cmd.append('--use_wandb')
            if params['train_monitor'] == 'wandb':
                cmd.extend(['--wandb_project', 'minimind_training'])

    return cmd