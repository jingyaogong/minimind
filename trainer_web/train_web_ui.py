import os
import sys
import subprocess
import threading
import json
import socket
import atexit
import signal
from flask import Flask, render_template, request, jsonify, redirect, url_for
import time
import psutil

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
    
    # 获取GPU数量参数，如果存在且大于1，则使用torchrun启动多卡训练
    gpu_num = int(params.get('gpu_num', 0)) if 'gpu_num' in params else 0
    use_torchrun = HAS_TORCH and GPU_COUNT > 0 and gpu_num > 1
    
    # 构建命令
    if train_type == 'pretrain':
        script_path = '../trainer/train_pretrain.py'
        if use_torchrun:
            cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path]
        else:
            cmd = [sys.executable, script_path]
        if 'save_weight' in params:
            cmd.extend(['--save_weight', params['save_weight']])
    elif train_type == 'sft':
        script_path = '../trainer/train_full_sft.py'
        if use_torchrun:
            cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path]
        else:
            cmd = [sys.executable, script_path]
        if 'save_weight' in params:
            cmd.extend(['--save_weight', params['save_weight']])
    elif train_type == 'lora':
        script_path = '../trainer/train_lora.py'
        if use_torchrun:
            cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path]
        else:
            cmd = [sys.executable, script_path]
        if 'lora_name' in params:
            cmd.extend(['--lora_name', params['lora_name']])
    elif train_type == 'dpo':
        script_path = '../trainer/train_dpo.py'
        if use_torchrun:
            cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path]
        else:
            cmd = [sys.executable, script_path]
        # 添加DPO特定参数
        if 'beta' in params and params['beta']:
            cmd.extend(['--beta', params['beta']])
        if 'accumulation_steps' in params and params['accumulation_steps']:
            cmd.extend(['--accumulation_steps', params['accumulation_steps']])
        if 'grad_clip' in params and params['grad_clip']:
            cmd.extend(['--grad_clip', params['grad_clip']])
    elif train_type == 'ppo':
        script_path = '../trainer/train_ppo.py'
        if use_torchrun:
            cmd = ['torchrun', '--nproc_per_node', str(gpu_num), script_path]
        else:
            cmd = [sys.executable, script_path]
        # 添加PPO特定参数
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
    else:
        return None
    
    # 添加通用参数
    for key, value in params.items():
        # 跳过特殊参数和DPO、PPO特有参数，以及gpu_num参数（因为已经在torchrun命令中使用）
        # 对于PPO训练，跳过--from_weight参数
        if key in ['train_type', 'save_weight', 'lora_name', 'train_monitor', 'beta', 'accumulation_steps', 'grad_clip', 'gpu_num', 'clip_epsilon', 'vf_coef', 'kl_coef', 'reasoning', 'update_old_actor_freq', 'reward_model_path'] or (train_type == 'ppo' and key == 'from_weight'):
            continue
        # 对于from_resume参数，需要正确传递参数值
        elif key == 'from_resume':
            # 确保传递参数名和参数值
            cmd.extend([f'--{key}', str(value)])
        else:
            # 确保log_interval和save_interval参数正确传递
            cmd.extend([f'--{key}', str(value)])
    
    # 单独处理训练监控参数，确保它不会被错误地添加值
    if 'train_monitor' in params:
        if params['train_monitor'] == 'wandb' or params['train_monitor'] == 'swanlab':
            cmd.append('--use_wandb')  # 对于wandb和swanlab，只添加标志，不添加值
            if params['train_monitor'] == 'wandb':
                cmd.extend(['--wandb_project', 'minimind_training'])
    
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
        'error': False,
        'train_monitor': params.get('train_monitor', 'none'),  # 保存训练监控设置
        'swanlab_url': None,
        'next_line_is_swanlab_url': False
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

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    train_type = data.get('train_type')
    
    # 移除不相关的参数
    params = data.copy()
    
    # 处理复选框参数
    if 'from_resume' not in params:
        params['from_resume'] = '0'
    
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
        # 确定状态
        status = '运行中' if info['running'] else \
                '手动停止' if 'manually_stopped' in info and info['manually_stopped'] else \
                '出错' if info['error'] else '已完成'
                
        result.append({
            'id': process_id,
            'train_type': info['train_type'],
            'start_time': info['start_time'],
            'running': info['running'],
            'error': info['error'],
            'status': status,
            'train_monitor': info.get('train_monitor', 'none'),  # 添加train_monitor字段
            'swanlab_url': info.get('swanlab_url')  # 添加swanlab_url字段
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
                'swanlab_url': info.get('swanlab_url')  # 保存swanlab_url
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
    print("正在保存进程信息...  save at 'trainer_web/training_processes.json'...")
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