import { getApiKey } from './authClient.js';
const defaultTimeout = 10000;

export function fetchWithTimeoutAndRetry(url, options = {}, timeout = defaultTimeout, retries = 3) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  const apiKey = getApiKey();
  const fetchOptions = {
    ...options,
    headers: {
      ...options.headers,
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      Pragma: 'no-cache',
      Expires: '0',
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
    },
    signal: controller.signal,
  };
  return fetch(url, fetchOptions)
    .then((response) => {
      clearTimeout(timeoutId);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response;
    })
    .catch((error) => {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') throw new Error('请求超时');
      if (retries > 0) {
        return new Promise((resolve) => {
          setTimeout(() => {
            resolve(fetchWithTimeoutAndRetry(url, options, timeout, retries - 1));
          }, timeout / 2);
        });
      }
      throw error;
    });
}

export function getProcesses() {
  return fetchWithTimeoutAndRetry('/processes').then((r) => r.json());
}

export function getLogs(processId) {
  return fetchWithTimeoutAndRetry(`/logs/${processId}`).then((r) => r.text());
}

export function startTrain(payload) {
  return fetchWithTimeoutAndRetry('/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' },
    body: JSON.stringify(payload),
  }).then((r) => r.json());
}

export function stopProcess(processId) {
  return fetchWithTimeoutAndRetry(`/stop/${processId}`, { method: 'POST' }).then((r) => r.json().catch(() => ({})));
}

export function deleteProcess(processId) {
  return fetchWithTimeoutAndRetry(`/delete/${processId}`, { method: 'POST' }).then((r) => r.json().catch(() => ({})));
}

export function getLogFiles() {
  return fetchWithTimeoutAndRetry('/logfiles').then((r) => r.json());
}

export function getLogFileContent(filename) {
  return fetchWithTimeoutAndRetry(`/logfile-content/${encodeURIComponent(filename)}`).then((r) => r.text());
}

export function deleteLogFile(filename) {
  return fetchWithTimeoutAndRetry(`/delete-logfile/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    headers: { 'Cache-Control': 'no-cache' },
  }).then((r) => r.json());
}

