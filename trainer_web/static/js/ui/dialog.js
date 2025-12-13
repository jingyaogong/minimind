export function showConfirmDialog(message, onConfirm, onCancel = null) {
  const existing = document.querySelector('.custom-dialog');
  if (existing && existing.parentNode && existing.parentNode.classList.contains('dialog-overlay')) {
    document.body.removeChild(existing.parentNode);
  }
  const overlay = document.createElement('div');
  overlay.className = 'dialog-overlay';
  const container = document.createElement('div');
  container.className = 'custom-dialog';
  container.innerHTML = `
    <div class="dialog-content">
      <div class="dialog-message">${message}</div>
      <div class="dialog-actions">
        <button class="dialog-button dialog-cancel">取消</button>
        <button class="dialog-button dialog-confirm">确认</button>
      </div>
    </div>
  `;
  overlay.appendChild(container);
  document.body.appendChild(overlay);
  setTimeout(() => {
    overlay.classList.add('show');
    container.classList.add('show');
  }, 10);
  const confirmBtn = container.querySelector('.dialog-confirm');
  confirmBtn.addEventListener('click', () => {
    if (onConfirm) onConfirm();
    closeDialog(overlay);
  });
  const cancelBtn = container.querySelector('.dialog-cancel');
  cancelBtn.addEventListener('click', () => {
    if (onCancel) onCancel();
    closeDialog(overlay);
  });
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      if (onCancel) onCancel();
      closeDialog(overlay);
    }
  });
}

export function closeDialog(overlay) {
  overlay.classList.remove('show');
  const container = overlay.querySelector('.custom-dialog');
  if (container) container.classList.remove('show');
  setTimeout(() => {
    if (overlay.parentNode) document.body.removeChild(overlay);
  }, 300);
}

