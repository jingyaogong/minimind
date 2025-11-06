import os
import sys
import subprocess
import threading
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__, template_folder='templates', static_folder='static')

# 存储训练进程的信息
training_processes = {}

# 启动训练进程
def start_training_process(train_type, params):
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用详细的时间戳作为进程ID和日志文件名
    process_id = time.strftime('%Y%m%d_%H%M%S')
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    log_file = os.path.join(log_dir, f"train_{train_type}_{process_id}.log")
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 构建命令
    if train_type == 'pretrain':
        script_path = '../trainer/train_pretrain.py'
        cmd = [sys.executable, script_path]
        if 'save_weight' in params:
            cmd.extend(['--save_weight', params['save_weight']])
    elif train_type == 'sft':
        script_path = '../trainer/train_full_sft.py'
        cmd = [sys.executable, script_path]
        if 'save_weight' in params:
            cmd.extend(['--save_weight', params['save_weight']])
    elif train_type == 'lora':
        script_path = '../trainer/train_lora.py'
        cmd = [sys.executable, script_path]
        if 'lora_name' in params:
            cmd.extend(['--lora_name', params['lora_name']])
    else:
        return None
    
    # 添加通用参数
    for key, value in params.items():
        if key not in ['train_type', 'save_weight', 'lora_name']:
            # 特殊处理布尔标志参数
            if key in ['use_wandb', 'from_resume']:
                if value == '1':  # 只有当值为1时才添加这个标志
                    cmd.append(f'--{key}')
            else:
                # 确保log_interval和save_interval参数正确传递
                cmd.extend([f'--{key}', str(value)])
    
    # 创建日志文件
    with open(log_file, 'w') as f:
        f.write(f"开始训练 {train_type} 进程\n")
        f.write(f"命令: {' '.join(cmd)}\n\n")
    
    # 启动进程
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    # 存储进程信息
    training_processes[process_id] = {
        'process': process,
        'train_type': train_type,
        'log_file': log_file,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'running': True,
        'error': False
    }
    
    # 开始读取输出
    def read_output():
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    with open(log_file, 'a') as f:
                        f.write(output)
            # 检查进程是否成功结束
            if process.returncode != 0:
                training_processes[process_id]['error'] = True
        finally:
            training_processes[process_id]['running'] = False
    
    # 启动线程读取输出
    threading.Thread(target=read_output, daemon=True).start()
    
    return process_id

# Flask路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    train_type = data.get('train_type')
    
    # 移除不相关的参数
    params = data.copy()
    
    # 处理复选框参数
    if 'from_resume' not in params:
        params['from_resume'] = '0'
    if 'use_wandb' not in params:
        params['use_wandb'] = '0'
    
    # 启动训练进程
    process_id = start_training_process(train_type, params)
    
    if process_id:
        return jsonify({'success': True, 'process_id': process_id})
    else:
        return jsonify({'success': False, 'error': '无效的训练类型'})

@app.route('/processes')
def processes():
    result = []
    for process_id, info in training_processes.items():
        result.append({
            'id': process_id,
            'train_type': info['train_type'],
            'start_time': info['start_time'],
            'running': info['running'],
            'error': info['error']
        })
    return jsonify(result)

@app.route('/logs/<process_id>')
def logs(process_id):
    # 直接从本地logfile目录读取日志文件
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            if filename.endswith(f'{process_id}.log'):
                log_file = os.path.join(log_dir, filename)
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            return f.read()
                    except Exception as e:
                        return f'读取日志失败: {str(e)}'
    return '日志文件不存在或已被删除'

@app.route('/logfiles')
def get_logfiles():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    
    logfiles = []
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            if filename.endswith('.log') and filename.startswith('train_'):
                file_path = os.path.join(log_dir, filename)
                try:
                    modified_time = os.path.getmtime(file_path)
                    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modified_time))
                    logfiles.append({
                        'filename': filename,
                        'modified_time': formatted_time,
                        'size': os.path.getsize(file_path)
                    })
                except Exception as e:
                    continue
    return jsonify(logfiles)

@app.route('/logfile-content/<filename>')
def get_logfile_content(filename):
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    
    # 安全检查：防止路径遍历攻击
    if '..' in filename or '/' in filename or '\\' in filename:
        return '非法的文件名'
    
    log_file = os.path.join(log_dir, filename)
    if os.path.exists(log_file) and os.path.isfile(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f'读取日志失败: {str(e)}'
    return '日志文件不存在'


@app.route('/stop/<process_id>', methods=['POST'])
def stop(process_id):
    if process_id in training_processes and training_processes[process_id]['running']:
        process = training_processes[process_id]['process']
        # 在Windows上使用terminate，在Unix上尝试优雅终止
        try:
            process.terminate()
            # 等待进程结束
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # 如果超时，强制杀死
            process.kill()
        
        training_processes[process_id]['running'] = False
        return jsonify({'success': True})
    return jsonify({'success': False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)