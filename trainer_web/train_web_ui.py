import os
import sys
import subprocess
import threading
import json
import socket
import atexit
import signal
import re
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask import g
import time
import psutil
import glob
import pathlib

# 尝试导入torch来检测GPU
try:
    import torch
    HAS_TORCH = True
    # 检测可用的GPU数量和设备信息
    if torch.cuda.is_available():
        GPU_COUNT = torch.cuda.device_count()
        # 获取GPU设备名称
        GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]
    else:
        GPU_COUNT = 0
        GPU_NAMES = []
except ImportError:
    HAS_TORCH = False
    GPU_COUNT = 0
    GPU_NAMES = []

def calculate_training_progress(process_id, process_info):
    """
    计算训练进度信息
    从日志文件中提取训练进度、loss、epoch等信息
    """
    progress = {
        'percentage': 0,
        'current_epoch': 0,
        'total_epochs': 0,
        'remaining_time': '计算中...',
        'current_loss': None,
        'current_lr': None
    }
    
    # 如果进程不在运行，返回空进度
    if not process_info.get('running', False):
        return progress
    
    try:
        # 获取日志文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, '../logfile')
        log_dir = os.path.abspath(log_dir)
        
        log_file = None
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                if filename.endswith(f'{process_id}.log'):
                    log_file = os.path.join(log_dir, filename)
                    break
        
        if not log_file or not os.path.exists(log_file):
            return progress
        
        # 读取日志文件的最后1000行
        def read_last_lines(file_path, n=1000):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # 使用更高效的方法读取最后n行
                    lines = []
                    for line in f:
                        lines.append(line.strip())
                        if len(lines) > n:
                            lines.pop(0)
                    return lines
            except Exception:
                return []
        
        lines = read_last_lines(log_file, 1000)
        
        # 从日志中提取进度信息
        current_epoch = 0
        total_epochs = 0
        current_loss = None
        current_lr = None
        
        for line in reversed(lines):  # 从最新日志开始
            line = line.strip()
            if not line:
                continue
                
            # 提取epoch信息 - 支持多种格式
            if not total_epochs:
                # 格式: epoch 3/10, Epoch 3 of 10, [3/10], 第3轮/共10轮
                epoch_patterns = [
                    r'epoch\s+(\d+)\s*/\s*(\d+)',
                    r'Epoch\s+(\d+)\s*of\s*(\d+)',
                    r'\[(\d+)/(\d+)\]',
                    r'epoch\s*[:：]\s*(\d+)\s*/\s*(\d+)',
                    r'第\s*(\d+)\s*轮\s*/\s*共\s*(\d+)\s*轮'
                ]
                
                for pattern in epoch_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        current_epoch = int(match.group(1))
                        total_epochs = int(match.group(2))
                        break
            
            # 提取loss信息 - 支持多种格式
            if not current_loss:
                # 格式: loss: 4.32, training_loss: 4.32, train_loss: 4.32, Loss: 4.32, 训练损失: 4.32
                loss_patterns = [
                    r'loss[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',           # loss: 4.32
                    r'training_loss[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',  # training_loss: 4.32
                    r'train_loss[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',     # train_loss: 4.32
                    r'Loss[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',          # Loss: 4.32
                    r'训练损失[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',        # 训练损失: 4.32
                    r'损失[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',           # 损失: 4.32
                    r'\s+([\d.]+(?:e[+-]?\d+)?)\s*loss',             # 4.32 loss
                    r'\s+([\d.]+(?:e[+-]?\d+)?)\s*训练损失',         # 4.32 训练损失
                    r'(?:loss|损失|training_loss|train_loss)\s*=\s*([\d.]+(?:e[+-]?\d+)?)'  # loss = 4.32
                ]
                
                for pattern in loss_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    if matches:
                        # 取最后一个匹配的loss值
                        loss_value = float(matches[-1])
                        if 0 < loss_value < 100:  # 合理的loss范围
                            current_loss = loss_value
                            break
            
            # 提取学习率信息 - 支持多种格式
            if not current_lr:
                # 格式: lr: 1e-4, learning_rate: 1e-4, LR: 1e-4, 学习率: 1e-4
                lr_patterns = [
                    r'lr[\s:=]\s*([\d.e+-]+)',
                    r'learning_rate[\s:=]\s*([\d.e+-]+)',
                    r'LR[\s:=]\s*([\d.e+-]+)',
                    r'学习率[\s:=]\s*([\d.e+-]+)'
                ]
                
                for pattern in lr_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    if matches:
                        # 取最后一个匹配的lr值
                        lr_value = float(matches[-1])
                        if 0 < lr_value < 1:  # 合理的lr范围
                            current_lr = f"{lr_value:.2e}"
                            break
            
            # 如果已经收集到足够信息，提前退出
            if total_epochs and current_loss and current_lr:
                break
        
        # 计算进度百分比
        percentage = 0
        if total_epochs > 0:
            percentage = min(100, max(0, int((current_epoch / total_epochs) * 100)))
        
        # 估算剩余时间（增强计算）
        remaining_time = '计算中...'
        if current_epoch > 0 and total_epochs > current_epoch:
            # 从日志中提取时间信息
            for line in reversed(lines):
                # 格式: remaining: 1:30:45, ETA: 1:30:45, 预计剩余: 1小时30分钟
                time_patterns = [
                    r'remaining[\s:=]\s*(\d+):(\d+):(\d+)',      # remaining: 1:30:45
                    r'ETA[\s:=]\s*(\d+):(\d+):(\d+)',            # ETA: 1:30:45
                    r'预计剩余[\s:=]\s*(\d+)[\s小时]*[\s:]?(\d+)?[\s分钟]*',  # 预计剩余: 1小时30分钟
                    r'剩余时间[\s:=]\s*(\d+)[\s小时]*[\s:]?(\d+)?[\s分钟]*',  # 剩余时间: 1小时30分钟
                    r'time left[\s:=]\s*(\d+)[\s:]?(\d+)?[\s:]?(\d+)?',  # time left: 1:30:45
                    r'还需[\s:=]\s*(\d+)[\s小时]*[\s:]?(\d+)?[\s分钟]*'  # 还需: 1小时30分钟
                ]
                
                for pattern in time_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        if len(groups) >= 3 and all(groups[:3]):
                            # 小时:分钟:秒格式
                            hours = int(groups[0])
                            minutes = int(groups[1])
                            seconds = int(groups[2])
                            if hours > 0 or minutes > 0 or seconds > 0:
                                parts = []
                                if hours > 0: parts.append(f"{hours}小时")
                                if minutes > 0: parts.append(f"{minutes}分钟")
                                if seconds > 0 and hours == 0 and minutes == 0:
                                    parts.append(f"{seconds}秒")
                                remaining_time = ''.join(parts)
                                break
                        elif len(groups) >= 2:
                            # 小时和分钟格式
                            hours = int(groups[0])
                            minutes = int(groups[1]) if groups[1] else 0
                            if hours > 0 or minutes > 0:
                                parts = []
                                if hours > 0: parts.append(f"{hours}小时")
                                if minutes > 0: parts.append(f"{minutes}分钟")
                                remaining_time = ''.join(parts)
                                break
                
                if remaining_time != '计算中...':
                    break
            
            # 如果没有找到时间信息，根据进度估算
            if remaining_time == '计算中...':
                # 假设每epoch时间大致相同
                elapsed_time = time.time() - process_info.get('start_timestamp', time.time())
                if current_epoch > 0:
                    time_per_epoch = elapsed_time / current_epoch
                    remaining_epochs = total_epochs - current_epoch
                    remaining_seconds = remaining_epochs * time_per_epoch
                    
                    if remaining_seconds > 3600:
                        remaining_time = f"{remaining_seconds / 3600:.1f}小时"
                    elif remaining_seconds > 60:
                        remaining_time = f"{remaining_seconds / 60:.1f}分钟"
                    else:
                        remaining_time = f"{int(remaining_seconds)}秒"
        
        return {
            'percentage': percentage,
            'current_epoch': current_epoch,
            'total_epochs': total_epochs,
            'remaining_time': remaining_time,
            'current_loss': f"{current_loss:.4f}" if current_loss else None,
            'current_lr': current_lr
        }
        
    except Exception as e:
        print(f"计算进度时出错: {e}")
        return progress

# 训练方式支持检测
def get_supported_training_methods():
    """获取当前环境支持的训练方法"""
    methods = {
        'pretrain': True,  # 预训练总是支持
        'sft': True,       # SFT总是支持
        'lora': True,      # LoRA总是支持
        'dpo': True,       # DPO总是支持
        'multi_gpu': HAS_TORCH and GPU_COUNT > 1  # 多GPU训练需要PyTorch和多个GPU
    }
    return methods

# 获取当前环境支持的训练方法
SUPPORTED_METHODS = get_supported_training_methods()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__, template_folder='templates', static_folder='static')

# 存储训练进程的信息
training_processes = {}

# 进程信息持久化文件
PROCESSES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_processes.json')

# PID文件
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_web_ui.pid')

# Authentication removed - allow anonymous training

# 启动训练进程
def start_training_process(train_type, params, client_id=None):
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
    
    # 获取GPU数量参数，如果存在且大于1，则使用torchrun启动多卡训练
    gpu_num = int(params.get('gpu_num', 0)) if 'gpu_num' in params else 0
    use_torchrun = HAS_TORCH and GPU_COUNT > 0 and gpu_num > 1
    
    try:
        from .dispatcher import build_command
    except ImportError:
        import sys as _sys
        import os as _os
        _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
        from dispatcher import build_command
    cmd = build_command(train_type, params, gpu_num, use_torchrun)
    if cmd is None:
        return None
    
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
        'start_timestamp': time.time(),  # 添加时间戳用于进度计算
        'running': True,
        'error': False,
        'train_monitor': params.get('train_monitor', 'none'),  # 保存训练监控设置
        'swanlab_url': None,
        'next_line_is_swanlab_url': False,
        'client_id': client_id
    }
    
    # 开始读取输出
    def read_output():
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # 检查是否是swanlab链接的行
                    output_stripped = output.strip()
                    if training_processes[process_id]['next_line_is_swanlab_url']:
                        # 保存swanlab链接
                        training_processes[process_id]['swanlab_url'] = output_stripped
                        training_processes[process_id]['next_line_is_swanlab_url'] = False
                    elif 'swanlab: 🚀 View run at' in output_stripped:
                        # 标记下一行是swanlab链接
                        training_processes[process_id]['next_line_is_swanlab_url'] = True
                    
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
    # 传递GPU信息到前端
    return render_template('index.html', has_gpu=HAS_TORCH and GPU_COUNT > 0, gpu_count=GPU_COUNT)

@app.route('/healthz')
def healthz():
    try:
        return jsonify({'status': 'ok', 'gpu': GPU_COUNT, 'methods': SUPPORTED_METHODS}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    train_type = data.get('train_type')
    
    # 移除不相关的参数
    params = data.copy()
    
    # 处理复选框参数
    if 'from_resume' not in params:
        params['from_resume'] = '0'
    
    # 启动训练进程 - 允许匿名训练，不传入client_id
    process_id = start_training_process(train_type, params)
    
    if process_id:
        return jsonify({'success': True, 'process_id': process_id})
    else:
        return jsonify({'success': False, 'error': '无效的训练类型'})

# 测试端点 - 添加模拟训练进程
@app.route('/test/add_process', methods=['POST'])
def add_test_process():
    """添加一个测试进程用于验证自动更新功能"""
    import subprocess
    import threading
    
    process_id = f"test_process_{int(time.time())}"
    
    # 创建测试训练命令
    test_command = [
        'python', '-c', '''
import time
import sys

print("2024-11-21 14:30:00 - Starting pretrain training")
sys.stdout.flush()
time.sleep(1)

print("2024-11-21 14:30:01 - Loading dataset from ../dataset/pretrain_hq.jsonl")
sys.stdout.flush()
time.sleep(1)

print("2024-11-21 14:30:02 - Model initialized with 108M parameters")
sys.stdout.flush()
time.sleep(2)

for epoch in range(1, 6):
    print(f"2024-11-21 14:30:{5 + epoch*5} - Starting epoch {epoch}/5")
    sys.stdout.flush()
    time.sleep(1)
    
    loss = 4.5 - epoch * 0.3
    lr = 1e-4 * (0.9 ** epoch)
    print(f"2024-11-21 14:30:{6 + epoch*5} - Loss: {loss:.4f}, lr: {lr:.2e}")
    sys.stdout.flush()
    time.sleep(2)
    
    remaining = (5 - epoch) * 15
    print(f"2024-11-21 14:30:{8 + epoch*5} - Epoch {epoch}/5 completed, remaining: 0:0{remaining}:00")
    sys.stdout.flush()
    time.sleep(1)

print("2024-11-21 14:30:35 - Training completed successfully")
sys.stdout.flush()
        '''
    ]
    
    # 启动进程
    process = subprocess.Popen(
        test_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # 保存进程信息
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logfile')
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    training_processes[process_id] = {
        'process': process,
        'train_type': 'pretrain',
        'log_file': os.path.join(log_dir, f'{process_id}.log'),
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'start_timestamp': time.time(),
        'running': True,
        'error': False,
        'train_monitor': 'none',
        'swanlab_url': None
    }
    
    # 启动线程读取输出并写入日志文件
    def read_output():
        try:
            log_file = training_processes[process_id]['log_file']
            with open(log_file, 'w') as f:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        f.write(line)
                        f.flush()
            process.wait()
            training_processes[process_id]['running'] = False
            if process.returncode != 0:
                training_processes[process_id]['error'] = True
        except Exception as e:
            print(f"读取测试进程输出时出错: {e}")
            training_processes[process_id]['running'] = False
            training_processes[process_id]['error'] = True
    
    threading.Thread(target=read_output, daemon=True).start()
    
    return jsonify({
        'success': True,
        'process_id': process_id,
        'message': '测试进程已添加'
    })

@app.route('/processes')
def processes():
    result = []
    for process_id, info in training_processes.items():
        # 确定状态
        status = '运行中' if info['running'] else \
                '手动停止' if 'manually_stopped' in info and info['manually_stopped'] else \
                '出错' if info['error'] else '已完成'
        
        # 计算训练进度信息
        progress = calculate_training_progress(process_id, info)
                
        result.append({
            'id': process_id,
            'train_type': info['train_type'],
            'start_time': info['start_time'],
            'running': info['running'],
            'error': info['error'],
            'status': status,
            'train_monitor': info.get('train_monitor', 'none'),  # 添加train_monitor字段
            'swanlab_url': info.get('swanlab_url'),  # 添加swanlab_url字段
            'progress': progress  # 添加进度信息
        })
    return jsonify(result)

@app.route('/api/browse')
def browse_files():
    """
    浏览服务器文件系统
    支持远程文件选择功能
    """
    try:
        # 获取请求的路径参数
        path = request.args.get('path', './')
        
        # 安全检查：限制访问范围
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        
        # 解析请求的路径
        if path.startswith('./'):
            # 相对路径，基于项目根目录
            full_path = os.path.abspath(os.path.join(project_root, path[2:]))
        elif path.startswith('/'):
            # 绝对路径，检查是否在项目目录内
            full_path = os.path.abspath(path)
        else:
            # 相对路径，基于项目根目录
            full_path = os.path.abspath(os.path.join(project_root, path))
        
        # 安全检查：确保路径在项目目录内
        if not full_path.startswith(project_root):
            full_path = project_root
        
        # 检查路径是否存在
        if not os.path.exists(full_path):
            return jsonify({'error': '路径不存在', 'path': path})
        
        # 获取目录内容
        if os.path.isdir(full_path):
            items = []
            try:
                # 列出目录内容
                for item in sorted(os.listdir(full_path)):
                    item_path = os.path.join(full_path, item)
                    
                    # 跳过隐藏文件和系统文件
                    if item.startswith('.') or item.startswith('__'):
                        continue
                    
                    try:
                        stat = os.stat(item_path)
                        items.append({
                            'name': item,
                            'path': item_path,  # 返回绝对路径
                            'relative_path': os.path.relpath(item_path, project_root),  # 同时返回相对路径用于显示
                            'type': 'directory' if os.path.isdir(item_path) else 'file',
                            'size': stat.st_size if os.path.isfile(item_path) else 0,
                            'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                        })
                    except (OSError, PermissionError):
                        # 跳过无法访问的项目
                        continue
                
                return jsonify({
                    'current_path': full_path,  # 返回绝对路径
                    'relative_path': os.path.relpath(full_path, project_root),  # 相对路径用于显示
                    'absolute_path': full_path,
                    'items': items,
                    'parent': os.path.dirname(full_path) if full_path != project_root else None
                })
            except (OSError, PermissionError) as e:
                return jsonify({'error': f'无法访问目录: {str(e)}', 'path': path})
        
        else:
            # 如果是文件，返回文件信息
            stat = os.stat(full_path)
            return jsonify({
                'name': os.path.basename(full_path),
                'path': full_path,  # 返回绝对路径
                'relative_path': os.path.relpath(full_path, project_root),  # 相对路径用于显示
                'type': 'file',
                'size': stat.st_size,
                'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            })
            
    except Exception as e:
        return jsonify({'error': f'浏览文件时出错: {str(e)}'})

@app.route('/api/quick-paths')
def quick_paths():
    """
    返回常用路径快捷方式
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        
        quick_paths = [
            {'name': '项目根目录', 'path': './', 'type': 'directory'},
            {'name': '数据集目录', 'path': './dataset', 'type': 'directory'},
            {'name': '模型检查点', 'path': './checkpoints', 'type': 'directory'},
            {'name': '日志文件', 'path': './logfile', 'type': 'directory'}
        ]
        
        # 验证路径是否存在
        valid_paths = []
        for item in quick_paths:
            full_path = os.path.join(project_root, item['path'][2:] if item['path'].startswith('./') else item['path'])
            if os.path.exists(full_path):
                valid_paths.append(item)
        
        return jsonify({'paths': valid_paths})
        
    except Exception as e:
        return jsonify({'error': f'获取快捷路径时出错: {str(e)}'})

@app.route('/logs/<process_id>')
def logs(process_id):
    # 直接从本地logfile目录读取日志文件
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    
    # 查找匹配的日志文件
    log_file = None
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            if filename.endswith(f'{process_id}.log'):
                log_file = os.path.join(log_dir, filename)
                break
    
    if not log_file or not os.path.exists(log_file):
        return '日志文件不存在或已被删除'
    
    try:
        # 使用高效且健壮的方法读取文件的最后200行
        def read_last_n_lines(file_path, n=200):
            # 使用二进制模式读取文件，避免编码问题
            with open(file_path, 'rb') as f:
                # 获取文件大小
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                
                # 如果文件很小，直接读取整个文件
                if file_size < 1024 * 1024:  # 小于1MB的文件直接读取
                    f.seek(0)
                    content = f.read()
                    return process_content(content)
                
                # 对于大文件，使用缓冲读取末尾部分
                # 估计需要读取的字节数（假设每行平均100字节）
                buffer_size = n * 200  # 为了保险，读取更多字节
                
                # 定位到适当的位置
                position = max(0, file_size - buffer_size)
                f.seek(position)
                
                # 读取缓冲区内容
                buffer = f.read(file_size - position)
                
                # 处理缓冲区内容
                lines = process_content(buffer)
                
                # 确保我们获取到完整的行
                # 如果缓冲区不是从文件开头开始，第一个行可能不完整
                if position > 0:
                    # 跳过第一个可能不完整的行
                    if len(lines) > 1:
                        lines = lines[1:]
                    else:
                        # 如果只有一行且不在文件开头，可能需要读取更多
                        # 这里简单处理，直接读取整个文件（罕见情况）
                        f.seek(0)
                        content = f.read()
                        lines = process_content(content)
                
                # 返回最后n行
                return lines[-n:] if len(lines) > n else lines
        
        def process_content(content):
            # 尝试多种编码方式解码内容
            encodings = ['utf-8', 'latin-1', 'gbk', 'gb2312']
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    # 使用True参数保留换行符，确保行分隔符正确
                    return text.splitlines(True)
                except UnicodeDecodeError:
                    continue
            # 如果所有编码都失败，使用错误替换模式
            text = content.decode('utf-8', errors='replace')
            return text.splitlines(True)
        
        # 读取最后200行
        last_200_lines = read_last_n_lines(log_file, 200)
        
        # 确保返回的内容顺序正确，并且不包含空行
        return ''.join(last_200_lines)
    except Exception as e:
        return f'读取日志失败: {str(e)}'

@app.route('/logfiles')
def get_logfiles():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    
    logfiles = []
    # 获取所有进程ID用于关联
    process_pids = set(training_processes.keys())
    
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            if filename.endswith('.log') and filename.startswith('train_'):
                file_path = os.path.join(log_dir, filename)
                try:
                    modified_time = os.path.getmtime(file_path)
                    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modified_time))
                    # 提取进程ID
                    pid = filename.split('.')[-2].split('_')[-1] if filename.endswith('.log') else None
                    logfiles.append({
                        'filename': filename,
                        'modified_time': formatted_time,
                        'size': os.path.getsize(file_path),
                        'process_id': pid,
                        'has_process': pid in process_pids
                    })
                except Exception as e:
                    continue
    # 按修改时间倒序排序，最新的在前面
    logfiles.sort(key=lambda x: x['modified_time'], reverse=True)
    return jsonify(logfiles)

@app.route('/logfile-content/<filename>')
def get_logfile_content(filename):
    # 安全检查：确保文件名不包含路径遍历字符
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'Invalid filename'}), 400
    
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径，train_web_ui.py在scripts目录下
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    log_file = os.path.join(log_dir, filename)
    
    try:
        # 使用二进制模式读取文件，可以更可靠地保留原始换行符
        with open(log_file, 'rb') as f:
            content_bytes = f.read()
        
        # 尝试多种编码方式解码，确保正确处理换行符
        encodings = ['utf-8', 'latin-1', 'gbk', 'gb2312']
        content = None
        
        for encoding in encodings:
            try:
                # 解码文件内容，保留原始换行符
                content = content_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        
        # 如果所有编码都失败，使用errors='replace'参数处理不可解码的字符
        if content is None:
            content = content_bytes.decode('utf-8', errors='replace')
        
        # 确保返回的内容正确保留所有换行符
        return content
    except FileNotFoundError:
        return jsonify({'error': 'Log file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete-logfile/<filename>', methods=['DELETE'])
def delete_logfile(filename):
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    
    # 安全检查：防止路径遍历攻击
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'success': False, 'message': '非法的文件名'})
    
    log_file = os.path.join(log_dir, filename)
    if os.path.exists(log_file) and os.path.isfile(log_file):
        try:
            os.remove(log_file)
            return jsonify({'success': True, 'message': '日志文件删除成功'})
        except Exception as e:
            print(f"删除日志文件失败: {str(e)}")
            return jsonify({'success': False, 'message': f'删除失败: {str(e)}'})
    return jsonify({'success': False, 'message': '日志文件不存在'})


@app.route('/stop/<process_id>', methods=['POST'])
def stop(process_id):
    if process_id in training_processes and training_processes[process_id]['running']:
        process = training_processes[process_id]['process']
        # 在Windows上使用terminate，在Unix上尝试优雅终止
        try:
            process.terminate()
            # 等待进程结束
            process.wait(timeout=5)
            # 标记为手动停止
            training_processes[process_id]['running'] = False
            training_processes[process_id]['manually_stopped'] = True
        except subprocess.TimeoutExpired:
            # 如果超时，强制杀死
            process.kill()
            # 标记为手动停止
            training_processes[process_id]['running'] = False
            training_processes[process_id]['manually_stopped'] = True
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/delete/<process_id>', methods=['POST'])
def delete(process_id):
    if process_id in training_processes:
        # 确保进程已经停止
        if training_processes[process_id]['running']:
            # 如果进程还在运行，先停止它
            try:
                process = training_processes[process_id]['process']
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            except Exception as e:
                print(f"停止进程失败: {str(e)}")
        
        # 从进程字典中删除
        del training_processes[process_id]
        
        # 可选：删除对应的日志文件
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(script_dir, '../logfile')
            log_dir = os.path.abspath(log_dir)
            
            if os.path.exists(log_dir):
                for filename in os.listdir(log_dir):
                    if filename.endswith(f'{process_id}.log'):
                        os.remove(os.path.join(log_dir, filename))
        except Exception as e:
            print(f"删除日志文件失败: {str(e)}")
        
        return jsonify({'success': True})
    return jsonify({'success': False})

def find_available_port(start_port=12581, max_attempts=100):
    """查找可用的端口号
    
    Args:
        start_port: 起始端口号
        max_attempts: 最大尝试次数
        
    Returns:
        可用的端口号，如果没有找到则返回None
    """
    for port in range(start_port, start_port + max_attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result != 0:  # 端口可用
            return port
    return None

def save_processes_info():
    """保存训练进程信息到文件"""
    try:
        # 创建一个不包含进程对象的可序列化版本
        serializable_processes = {}
        for pid, info in training_processes.items():
            serializable_processes[pid] = {
                'pid': info.get('pid', info.get('process').pid) if isinstance(info.get('process'), subprocess.Popen) else info.get('pid'),
                'train_type': info['train_type'],
                'log_file': info['log_file'],
                'start_time': info['start_time'],
                'running': info['running'],
                'error': info.get('error', False),
                'manually_stopped': info.get('manually_stopped', False),
                'train_monitor': info.get('train_monitor', 'none'),  # 保存train_monitor
                'swanlab_url': info.get('swanlab_url'),
                'client_id': info.get('client_id')
            }
        
        with open(PROCESSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_processes, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存进程信息失败: {str(e)}")

def load_processes_info():
    """从文件加载训练进程信息"""
    global training_processes
    try:
        if os.path.exists(PROCESSES_FILE):
            with open(PROCESSES_FILE, 'r', encoding='utf-8') as f:
                loaded_processes = json.load(f)
            
            # 检查每个进程是否还在运行
            for pid, info in loaded_processes.items():
                # 确保所有需要的字段都存在
                if 'swanlab_url' not in info:
                    info['swanlab_url'] = None
                if 'manually_stopped' not in info:
                    info['manually_stopped'] = False
                if 'error' not in info:
                    info['error'] = False
                if 'train_monitor' not in info:
                    info['train_monitor'] = 'none'
                if 'client_id' not in info:
                    info['client_id'] = None
                
                if info['running']:
                    try:
                        # 检查进程是否还在运行
                        proc = psutil.Process(info['pid'])
                        if proc.is_running() and proc.status() != 'zombie':
                            # 进程仍在运行，恢复信息
                            training_processes[pid] = info
                        else:
                            # 进程已停止
                            info['running'] = False
                            # 如果进程未被明确标记为完成或出错，则默认为手动停止
                            if not info['error']:
                                info['manually_stopped'] = True
                            training_processes[pid] = info
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # 进程不存在或无权限访问
                        info['running'] = False
                        # 如果进程未被明确标记为完成或出错，则默认为手动停止
                        if not info['error']:
                            info['manually_stopped'] = True
                        training_processes[pid] = info
                else:
                    # 进程已停止，直接恢复
                    training_processes[pid] = info
    except Exception as e:
        print(f"加载进程信息失败: {str(e)}")

def handle_exit(signum, frame):
    """处理程序退出信号，保存进程信息"""
    print("正在保存进程信息...")
    save_processes_info()
    # 删除PID文件
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except:
            pass
    sys.exit(0)

# 注册退出处理器
signal.signal(signal.SIGINT, handle_exit)  # Ctrl+C
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, handle_exit)  # 终止信号

# 注册程序退出时的处理函数
atexit.register(save_processes_info)

if __name__ == '__main__':
    # 加载已保存的进程信息
    load_processes_info()
    
    # 创建PID文件，用于标识web进程
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    
    # 尝试使用默认端口12581，如果被占用则自动寻找可用端口
    port = find_available_port(12581)
    if port is not None:
        print(f"启动Flask服务器在 http://0.0.0.0:{port}")
        print(f"使用nohup启动可保持服务持续运行: nohup python -u scripts/train_web_ui.py &")
        # 使用0.0.0.0作为host以兼容VSCode的端口转发功能
        app.run(host='0.0.0.0', port=port, debug=False)  # 生产环境关闭debug
    else:
        print("无法找到可用的端口，请检查系统端口占用情况")
        # 删除PID文件
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except:
                pass
        sys.exit(1)
# Registration endpoint removed - allow anonymous training