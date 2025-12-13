import json
import urllib.request
import urllib.error

class MinimindClient:
    def __init__(self, base_url, api_key=None, timeout=10):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or ''
        self.timeout = timeout

    def _request(self, method, path, body=None, expect_text=False):
        url = f"{self.base_url}{path}"
        headers = {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache'
        }
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        data = None
        if body is not None:
            data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                if expect_text:
                    return raw.decode('utf-8', errors='replace')
                return json.loads(raw.decode('utf-8'))
        except urllib.error.HTTPError as e:
            msg = e.read().decode('utf-8', errors='replace')
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except urllib.error.URLError as e:
            raise RuntimeError(str(e))

    def register(self, name, email):
        res = self._request('POST', '/api/register', {'name': name, 'email': email})
        self.api_key = res.get('api_key', self.api_key)
        return res

    def start_training(self, train_type, **params):
        payload = {'train_type': train_type}
        payload.update(params or {})
        res = self._request('POST', '/train', payload)
        return res

    def get_processes(self):
        return self._request('GET', '/processes', None)

    def get_logs(self, process_id):
        return self._request('GET', f"/logs/{process_id}", None, expect_text=True)

    def stop(self, process_id):
        return self._request('POST', f"/stop/{process_id}", None)

    def delete(self, process_id):
        return self._request('POST', f"/delete/{process_id}", None)

    def get_logfiles(self):
        return self._request('GET', '/logfiles', None)

    def get_logfile_content(self, filename):
        return self._request('GET', f"/logfile-content/{filename}", None, expect_text=True)

    def delete_logfile(self, filename):
        return self._request('DELETE', f"/delete-logfile/{filename}", None)