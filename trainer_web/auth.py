import os
import json
import hashlib
import secrets
import time

CLIENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clients.json')

def _load_clients():
    if os.path.exists(CLIENTS_FILE):
        with open(CLIENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def _save_clients(clients):
    with open(CLIENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(clients, f, ensure_ascii=False, indent=2)

def _hash_api_key(api_key):
    return hashlib.sha256(api_key.encode('utf-8')).hexdigest()

def register_client(name, email):
    clients = _load_clients()
    client_id = secrets.token_hex(8) + str(int(time.time()))
    api_key = secrets.token_hex(32)
    clients[client_id] = {
        'client_id': client_id,
        'name': name,
        'email': email,
        'api_key_hash': _hash_api_key(api_key),
        'created_at': int(time.time())
    }
    _save_clients(clients)
    return {'client_id': client_id, 'api_key': api_key}

def verify_api_key(api_key):
    clients = _load_clients()
    target = _hash_api_key(api_key)
    for _, info in clients.items():
        if info.get('api_key_hash') == target:
            return info
    return None

def get_client_from_request(request):
    auth = request.headers.get('Authorization')
    if not auth or not auth.startswith('Bearer '):
        return None
    token = auth.split(' ', 1)[1].strip()
    return verify_api_key(token)