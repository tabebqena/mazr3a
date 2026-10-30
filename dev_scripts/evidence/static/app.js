const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  items: [],
  byKey: new Map(),
  filtered: [],
  page: 1,
  pageSize: 1,
  selected: null,
};

let boxObserver = null;

function esc(value) {
  const text = value == null ? '' : String(value);
  return text.replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function imgUrl(rel) {
  return '/evidence/' + String(rel).split('/').map(encodeURIComponent).join('/');
}

function findCard(key) {
  return $$('.card').find(card => card.dataset.key === key) || null;
}

function pages() {
  return Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
}

async function init() {
  try {
    const res = await fetch('/api/items');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    state.items = data.items || [];
    for (const item of state.items) {
      item.verdict = item.verdict || '';
      state.byKey.set(item.key, item);
    }
    fillSelects();
    bind();
    refresh(true);
  } catch (err) {
    $('#summary').textContent = 'Failed to load: ' + err.message;
  }
}

function fillSelects() {
  const cameraSelect = $('#camera');
  const cameras = [...new Set(state.items.map(item => item.camera).filter(Boolean))].sort();
  for (const camera of cameras) {
    const option = document.createElement('option');
    option.value = camera;
    option.textContent = camera;
    cameraSelect.appendChild(option);
  }
  const labelSelect = $('#label');
  const labels = [...new Set(state.items.flatMap(item => item.labels || []))].sort();
  for (const label of labels) {
    const option = document.createElement('option');
    option.value = label;
    option.textContent = label;
    labelSelect.appendChild(option);
  }
}

function bind() {
  for (const id of ['camera', 'verdict', 'label', 'min-boxes', 'sort']) {
    $('#' + id).addEventListener('change', () => refresh(true));
  }
  for (const id of ['alerted', 'orphans', 'hide-classified']) {
    $('#' + id).addEventListener('change', () => refresh(true));
  }
  for (const id of ['show-boxes', 'show-annotated']) {
    $('#' + id).addEventListener('change', render);
  }
  $('#prev').addEventListener('click', () => {
    if (state.page > 1) { state.page--; render(); selectCurrent(); }
  });
  $('#next').addEventListener('click', () => {
    if (state.page < pages()) { state.page++; render(); selectCurrent(); }
  });
  const pageInput = $('#page-input');
  pageInput.addEventListener('focus', () => pageInput.select());
  pageInput.addEventListener('change', () => goToPage(pageInput.value));
  pageInput.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      goToPage(pageInput.value);
    } else if (event.key === 'Escape') {
      pageInput.value = String(state.page);
      pageInput.blur();
    }
  });
  $('#next-unclassified').addEventListener('click', nextUnclassified);
  $('#grid').addEventListener('click', onGridClick);
  document.addEventListener('keydown', onKey);
  window.addEventListener('resize', () => {
    const img = $('#grid img');
    if (img) drawBoxes(img);
  });
}

function refresh(resetPage) {
  if (resetPage) state.page = 1;
  const camera = $('#camera').value;
  const verdict = $('#verdict').value;
  const label = $('#label').value;
  const minBoxes = Number.parseInt($('#min-boxes').value, 10) || 0;
  const alerted = $('#alerted').checked;
  const orphans = $('#orphans').checked;
  const hideClassified = $('#hide-classified').checked;
  let list = state.items.filter(item => {
    if (camera && item.camera !== camera) return false;
    if (alerted && !item.alerted) return false;
    if (orphans && !item.orphan) return false;
    if (label && !(item.labels || []).includes(label)) return false;
    if (minBoxes && (item.detections || []).length < minBoxes) return false;
    if (verdict === 'unclassified' && item.verdict) return false;
    if (verdict === 'true' && item.verdict !== 'true') return false;
    if (verdict === 'false' && item.verdict !== 'false') return false;
    if (hideClassified && item.verdict) return false;
    return true;
  });
  const sort = $('#sort').value;
  list = list.slice().sort((a, b) => {
    if (sort === 'oldest') return (a.captured_at - b.captured_at) || a.key.localeCompare(b.key);
    if (sort === 'high') return (b.best_score - a.best_score) || (b.captured_at - a.captured_at);
    if (sort === 'low') return (a.best_score - b.best_score) || (b.captured_at - a.captured_at);
    return (b.captured_at - a.captured_at) || a.key.localeCompare(b.key);
  });
  state.filtered = list;
  if (!list.some(item => item.key === state.selected)) {
    state.selected = list.length ? list[0].key : null;
  }
  const maxPage = pages();
  if (state.page > maxPage) state.page = maxPage;
  render();
}

function selectCurrent() {
  const item = state.filtered[state.page - 1];
  if (item) select(item.key);
}

function goToPage(value) {
  const requested = Number.parseInt(value, 10);
  const totalPages = pages();
  if (!Number.isFinite(requested)) {
    $('#page-input').value = String(state.page);
    return;
  }
  state.page = Math.min(Math.max(requested, 1), totalPages);
  render();
  selectCurrent();
  $('#page-input').blur();
}

function render() {
  const grid = $('#grid');
  const start = (state.page - 1) * state.pageSize;
  const pageItems = state.filtered.slice(start, start + state.pageSize);
  grid.innerHTML = pageItems.map(cardHtml).join('');
  grid.classList.toggle('hidden', state.filtered.length === 0);
  $('#empty').classList.toggle('hidden', state.filtered.length !== 0);
  const totalPages = pages();
  const pageInput = $('#page-input');
  pageInput.max = String(totalPages);
  pageInput.disabled = state.filtered.length === 0;
  if (document.activeElement !== pageInput) pageInput.value = String(state.page);
  $('#page-total').textContent = String(totalPages);
  $('#match-count').textContent = String(state.filtered.length);
  $('#prev').disabled = state.page <= 1;
  $('#next').disabled = state.page >= totalPages;
  $$('.imgwrap img', grid).forEach(img => {
    const draw = () => drawBoxes(img);
    img.addEventListener('load', draw);
    if (img.complete) draw();
  });
  observeBoxes();
  updateStats();
}

function observeBoxes() {
  if (!('ResizeObserver' in window)) return;
  if (!boxObserver) {
    boxObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        const img = entry.target.querySelector('img');
        if (img) drawBoxes(img);
      }
    });
  }
  boxObserver.disconnect();
  $$('.imgwrap', $('#grid')).forEach(wrap => boxObserver.observe(wrap));
}

function cardHtml(item) {
  const labelTags = (item.labels || []).map(label => {
    const cls = String(label).toLowerCase() === 'smoke' ? 'smoke' : 'fire';
    return '<span class="tag ' + cls + '">' + esc(label) + '</span>';
  }).join('');
  const score = Number(item.best_score || 0).toFixed(2);
  const useAnnotated = $('#show-annotated').checked && item.annotated;
  const src = imgUrl(useAnnotated ? item.annotated : item.file);
  const cls = ['card'];
  if (item.verdict) cls.push('v-' + item.verdict);
  if (state.selected === item.key) cls.push('selected');
  return '<article class="' + cls.join(' ') + '" data-key="' + esc(item.key) + '" tabindex="0">' +
    '<div class="imgwrap">' +
      '<img loading="lazy" src="' + src + '" data-key="' + esc(item.key) + '" alt="' +
        esc(item.camera) + ' ' + esc(item.ts_utc) + '">' +
      '<div class="boxes"></div>' +
      (item.alerted ? '<span class="badge alert">alerted</span>' : '') +
      (item.orphan ? '<span class="badge orphan" title="On disk with no DB row">orphan</span>' : '') +
      (item.verdict ? '<span class="badge verdict ' + item.verdict + '">' + item.verdict + '</span>' : '') +
    '</div>' +
    '<div class="meta">' +
      labelTags +
      '<span class="score">best ' + score + '</span>' +
      '<span class="tag">' + esc(item.camera) + '</span>' +
      '<span class="ts">' + esc(item.ts_utc) + '</span>' +
    '</div>' +
    '<div class="actions">' +
      '<button type="button" data-verdict="true" class="' + (item.verdict === 'true' ? 'active' : '') + '" title="' + (item.verdict === 'true' ? 'Remove the true mark' : 'Mark this image as true') + '">' + (item.verdict === 'true' ? 'Clear true' : 'Mark true') + '</button>' +
      '<button type="button" data-verdict="false" class="' + (item.verdict === 'false' ? 'active' : '') + '" title="' + (item.verdict === 'false' ? 'Remove the false mark' : 'Mark this image as false') + '">' + (item.verdict === 'false' ? 'Clear false' : 'Mark false') + '</button>' +
    '</div>' +
  '</article>';
}

function drawBoxes(img) {
  const wrap = img.parentElement;
  const layer = wrap.querySelector('.boxes');
  if (!layer) return;
  layer.innerHTML = '';
  const naturalWidth = img.naturalWidth || 640;
  const naturalHeight = img.naturalHeight || 360;
  const scale = Math.min(wrap.clientWidth / naturalWidth, wrap.clientHeight / naturalHeight);
  const displayWidth = naturalWidth * scale;
  const displayHeight = naturalHeight * scale;
  layer.style.right = 'auto';
  layer.style.bottom = 'auto';
  layer.style.left = ((wrap.clientWidth - displayWidth) / 2) + 'px';
  layer.style.top = ((wrap.clientHeight - displayHeight) / 2) + 'px';
  layer.style.width = displayWidth + 'px';
  layer.style.height = displayHeight + 'px';
  if (!$('#show-boxes').checked || $('#show-annotated').checked) return;
  const item = state.byKey.get(img.dataset.key);
  if (!item) return;
  for (const det of item.detections || []) {
    const x1 = det.x1 / naturalWidth * 100;
    const y1 = det.y1 / naturalHeight * 100;
    const x2 = det.x2 / naturalWidth * 100;
    const y2 = det.y2 / naturalHeight * 100;
    const box = document.createElement('div');
    box.className = 'box ' + (String(det.label).toLowerCase() === 'smoke' ? 'smoke' : 'fire');
    box.style.left = x1 + '%';
    box.style.top = y1 + '%';
    box.style.width = Math.max(0, x2 - x1) + '%';
    box.style.height = Math.max(0, y2 - y1) + '%';
    box.title = det.label + ' ' + Number(det.score || 0).toFixed(2);
    layer.appendChild(box);
  }
}

function onGridClick(event) {
  const card = event.target.closest('.card');
  if (!card) return;
  const button = event.target.closest('button[data-verdict]');
  const key = card.dataset.key;
  if (button) {
    event.preventDefault();
    setVerdict(key, button.dataset.verdict, false);
    return;
  }
  select(key);
}

function select(key) {
  state.selected = key;
  $$('.card.selected').forEach(card => card.classList.remove('selected'));
  const card = findCard(key);
  if (card) {
    card.classList.add('selected');
    card.focus({ preventScroll: true });
  }
}

function updateCard(item) {
  const card = findCard(item.key);
  if (!card) return;
  card.classList.remove('v-true', 'v-false');
  if (item.verdict) card.classList.add('v-' + item.verdict);
  $$('button[data-verdict]', card).forEach(button => {
    const value = button.dataset.verdict;
    const active = value === item.verdict;
    button.classList.toggle('active', active);
    button.textContent = active ? 'Clear ' + value : 'Mark ' + value;
    button.title = active
      ? 'Remove the ' + value + ' mark'
      : 'Mark this image as ' + value;
  });
  const oldBadge = $('.badge.verdict', card);
  if (oldBadge) oldBadge.remove();
  if (item.verdict) {
    const badge = document.createElement('span');
    badge.className = 'badge verdict ' + item.verdict;
    badge.textContent = item.verdict;
    $('.imgwrap', card).appendChild(badge);
  }
}

async function setVerdict(key, verdict, advance) {
  const item = state.byKey.get(key);
  if (!item) return;
  if (verdict && item.verdict === verdict) verdict = '';
  const previous = item.verdict;
  const previousIndex = state.filtered.findIndex(entry => entry.key === key);
  item.verdict = verdict;
  try {
    const res = await fetch('/api/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: key, verdict: verdict }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
  } catch (err) {
    item.verdict = previous;
    $('#hint').textContent = 'Could not save: ' + err.message;
    return;
  }
  $('#hint').textContent = 'T/F toggle · click active again to clear · ←/→ move';
  const filterVerdict = $('#verdict').value;
  const leaves = filterVerdict && filterVerdict !== (verdict || 'unclassified');
  if (leaves || $('#hide-classified').checked) {
    refresh(false);
    if (state.filtered.some(item => item.key === key)) {
      select(key);
    } else {
      const index = Math.min(Math.max(previousIndex, 0), state.filtered.length - 1);
      if (state.filtered[index]) focusItem(state.filtered[index].key);
    }
    return;
  }
  updateCard(item);
  updateStats();
  if (advance) move(1);
  else select(key);
}

function move(delta) {
  if (!state.filtered.length) return;
  let index = state.filtered.findIndex(item => item.key === state.selected);
  if (index < 0) index = delta > 0 ? -1 : state.filtered.length;
  index = Math.min(Math.max(index + delta, 0), state.filtered.length - 1);
  focusItem(state.filtered[index].key);
}

function focusItem(key) {
  const index = state.filtered.findIndex(item => item.key === key);
  if (index < 0) return;
  const page = Math.floor(index / state.pageSize) + 1;
  if (page !== state.page) {
    state.page = page;
    render();
  }
  select(key);
  const card = findCard(key);
  if (card) card.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

function nextUnclassified() {
  $('#verdict').value = 'unclassified';
  $('#hide-classified').checked = true;
  refresh(true);
  if (state.filtered.length) focusItem(state.filtered[0].key);
}

function updateStats() {
  const total = state.items.length;
  let yes = 0;
  let no = 0;
  for (const item of state.items) {
    if (item.verdict === 'true') yes++;
    else if (item.verdict === 'false') no++;
  }
  const unclassified = total - yes - no;
  $('#stats').innerHTML =
    '<span class="stat"><b>' + total + '</b> total</span>' +
    '<span class="stat true"><b>' + yes + '</b> true</span>' +
    '<span class="stat false"><b>' + no + '</b> false</span>' +
    '<span class="stat"><b>' + unclassified + '</b> unclassified</span>';
  $('#summary').textContent =
    state.filtered.length + ' matching · ' + total + ' in evidence';
  const pct = total ? (yes + no) / total * 100 : 0;
  $('#progress-bar').style.width = pct.toFixed(1) + '%';
}

function currentItem() {
  const selected = state.filtered.find(item => item.key === state.selected);
  return selected || state.filtered[state.page - 1] || null;
}

function onKey(event) {
  const tag = (event.target.tagName || '').toLowerCase();
  const key = event.key.toLowerCase();
  if (tag === 'input' || tag === 'textarea' || event.target.isContentEditable) return;
  const isVerdictKey = key === 't' || key === 'f' || key === '0';
  if (tag === 'select' && !isVerdictKey) return;
  if (isVerdictKey) {
    event.preventDefault();
    const item = currentItem();
    if (!item) return;
    if (state.selected !== item.key) select(item.key);
    setVerdict(item.key, key === 't' ? 'true' : key === 'f' ? 'false' : '', true);
  } else if (key === 'arrowright' || key === 'j' || key === 'n') {
    event.preventDefault();
    move(1);
  } else if (key === 'arrowleft' || key === 'k' || key === 'p') {
    event.preventDefault();
    move(-1);
  }
}

init();
