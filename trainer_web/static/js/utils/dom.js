export function qs(selector, scope = document) {
  return scope.querySelector(selector);
}

export function qsa(selector, scope = document) {
  return Array.from(scope.querySelectorAll(selector));
}

export function el(tag, attrs = {}) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else node.setAttribute(k, v);
  }
  return node;
}

export function setHidden(node, hidden) {
  if (!node) return;
  if (hidden) node.classList.add('hidden');
  else node.classList.remove('hidden');
}

export function setText(node, text) {
  if (!node) return;
  node.textContent = text;
}

export function clearChildren(node) {
  if (!node) return;
  while (node.firstChild) node.removeChild(node.firstChild);
}

