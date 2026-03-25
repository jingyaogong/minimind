export function showNotification(message, type = 'success') {
  const n = document.createElement('div');
  n.className = `notification notification-${type}`;
  n.textContent = message;
  document.body.appendChild(n);
  setTimeout(() => {
    n.classList.add('show');
  }, 10);
  setTimeout(() => {
    n.classList.remove('show');
    setTimeout(() => {
      if (n.parentNode) document.body.removeChild(n);
    }, 300);
  }, 3000);
}

