const KEY = 'minimind_api_key';

export function getApiKey() {
  try {
    return localStorage.getItem(KEY) || '';
  } catch (_) {
    return '';
  }
}

export function setApiKey(k) {
  try {
    localStorage.setItem(KEY, k || '');
  } catch (_) {}
}

export function registerClient(payload) {
  return fetch('/api/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' },
    body: JSON.stringify(payload || {}),
  }).then((r) => {
    if (!r.ok) throw new Error('register_failed');
    return r.json();
  }).then((res) => {
    if (res && res.api_key) setApiKey(res.api_key);
    return res;
  });
}