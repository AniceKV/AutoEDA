/* AutoEDA Pro — app.js v2
   Modules: Tabs, Accordion, VariableSearch, DataSourceToggle,
            FileDropZone, StatusBar, LiveLog, Toast, DataPreview,
            RunForm, ResetBtn, QuestionModal, ReportIframe,
            SidebarToggle
*/
"use strict";

// ── CSRF helper ──────────────────────────────────────────────
function getCsrfToken() {
  const el = document.querySelector('[name=csrfmiddlewaretoken]');
  if (el) return el.value;
  const cookie = document.cookie.split(';').find(c => c.trim().startsWith('csrftoken='));
  return cookie ? cookie.split('=')[1].trim() : '';
}

function fetchPost(url, formData) {
  formData.append('csrfmiddlewaretoken', getCsrfToken());
  return fetch(url, { method: 'POST', body: formData });
}

// ── Toast system ─────────────────────────────────────────────
const Toast = (() => {
  const ICONS = { success: '✓', error: '⚠', info: 'ℹ' };

  function show(msg, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span class="toast-icon">${ICONS[type] || 'ℹ'}</span>
                       <span class="toast-msg">${msg}</span>`;
    container.appendChild(toast);

    const dismiss = () => {
      toast.classList.add('toast-out');
      toast.addEventListener('animationend', () => toast.remove(), { once: true });
    };
    setTimeout(dismiss, duration);
    toast.addEventListener('click', dismiss);
  }

  return { show };
})();

// ── Tab system ───────────────────────────────────────────────
function initTabs(barSelector, panelSelector, storageKey) {
  const btns   = document.querySelectorAll(barSelector);
  const panels = document.querySelectorAll(panelSelector);
  if (!btns.length) return;

  function activate(tabId) {
    btns.forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
    panels.forEach(p => p.classList.toggle('active', p.id === tabId));
    if (storageKey) sessionStorage.setItem(storageKey, tabId);
    // Trigger specific tab hooks
    if (tabId === 'tab-report') initReportIframe();
    if (tabId === 'tab-preview') DataPreview.load();
  }

  btns.forEach(btn => btn.addEventListener('click', () => activate(btn.dataset.tab)));

  const saved = storageKey ? sessionStorage.getItem(storageKey) : null;
  const first = saved && document.getElementById(saved) ? saved : btns[0].dataset.tab;
  activate(first);
}

// ── Accordion ────────────────────────────────────────────────
function initAccordions() {
  document.querySelectorAll('.accordion-trigger').forEach(trigger => {
    trigger.addEventListener('click', () => {
      const item = trigger.closest('.accordion-item');
      item.classList.toggle('open');
    });
  });
}

// ── Variable search ──────────────────────────────────────────
function initVariableSearch() {
  const searchInput = document.getElementById('variable-search');
  if (!searchInput) return;

  searchInput.addEventListener('input', () => {
    const q = searchInput.value.toLowerCase().trim();
    document.querySelectorAll('.accordion-item[data-col]').forEach(item => {
      const col = item.dataset.col.toLowerCase();
      item.style.display = (!q || col.includes(q)) ? '' : 'none';
    });
  });
}

// ── Data source toggle ───────────────────────────────────────
function initDataSourceToggle() {
  const radios     = document.querySelectorAll('input[name="data_source"]');
  const sampleWrap = document.getElementById('sample-section');
  const uploadWrap = document.getElementById('upload-section');
  if (!radios.length) return;

  function update() {
    const val = document.querySelector('input[name="data_source"]:checked').value;
    if (sampleWrap) sampleWrap.style.display = val === 'sample' ? '' : 'none';
    if (uploadWrap) uploadWrap.style.display  = val === 'upload' ? '' : 'none';
  }
  radios.forEach(r => r.addEventListener('change', update));
  update();
}

// ── File drop zone ───────────────────────────────────────────
function initFileDropZone() {
  const zone = document.getElementById('file-drop-zone');
  if (!zone) return;
  const input   = zone.querySelector('input[type="file"]');
  const display = zone.querySelector('.file-name-display');
  if (!input) return;

  // Prevent click event bubbling when native file dialog is triggered
  input.addEventListener('click', e => e.stopPropagation());

  zone.addEventListener('click', e => {
    if (e.target !== input) {
      input.click();
    }
  });
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      input.files = e.dataTransfer.files;
      if (display) display.textContent = e.dataTransfer.files[0].name;
    }
  });
  input.addEventListener('change', () => {
    if (display) display.textContent = input.files && input.files[0] ? input.files[0].name : '';
  });
}

// ── Live Log module ───────────────────────────────────────────
const LiveLog = (() => {
  let pollInterval = null;
  let lastLineCount = 0;

  const panel = () => document.getElementById('live-log-panel');

  function renderLine(line) {
    const span = document.createElement('div');
    if (line.startsWith('✅')) span.className = 'log-line-ok';
    else if (line.startsWith('❌')) span.className = 'log-line-err';
    else if (line.startsWith('❓')) span.className = 'log-line-q';
    span.textContent = line;
    return span;
  }

  function poll() {
    fetch('/api/log/')
      .then(r => r.json())
      .then(data => {
        const lines = data.lines || [];
        const p = panel();
        if (!p) return;
        if (lines.length !== lastLineCount) {
          // Only append new lines
          const newLines = lines.slice(lastLineCount);
          newLines.forEach(l => p.appendChild(renderLine(l)));
          lastLineCount = lines.length;
          p.scrollTop = p.scrollHeight;
        }
      })
      .catch(() => {});
  }

  function start() {
    const p = panel();
    if (p) { p.classList.add('visible'); p.innerHTML = ''; }
    lastLineCount = 0;
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(poll, 1500);
    poll(); // immediate first fetch
  }

  function stop() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
    // Final fetch to catch last lines
    setTimeout(() => {
      fetch('/api/log/').then(r => r.json()).then(data => {
        const lines = data.lines || [];
        const p = panel();
        if (!p) return;
        const newLines = lines.slice(lastLineCount);
        newLines.forEach(l => p.appendChild(renderLine(l)));
        lastLineCount = lines.length;
        if (lines.length) p.scrollTop = p.scrollHeight;
      }).catch(() => {});
    }, 500);
  }

  return { start, stop };
})();

// ── Status bar & polling ──────────────────────────────────────
const StatusBar = (() => {
  let pollInterval = null;

  function setNav(state, msg) {
    const el = document.getElementById('nav-status');
    if (!el) return;
    el.className = 'topnav-status ' + state;
    el.innerHTML = state === 'running'
      ? `<span class="pulse-dot"></span>${msg}`
      : msg;
  }

  function setProgress(visible, msg) {
    const container = document.getElementById('progress-container');
    const label     = document.getElementById('progress-label');
    if (!container) return;
    container.classList.toggle('visible', visible);
    if (label && msg) label.textContent = msg;
  }

  function setRunBtn(disabled) {
    const btn = document.getElementById('run-btn');
    if (btn) { btn.disabled = disabled; btn.textContent = disabled ? 'Running…' : 'Execute Pipeline'; }
  }

  function stop() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
    LiveLog.stop();
    setProgress(false, '');
    setRunBtn(false);
  }

  let errCount = 0;

  function poll() {
    fetch('/api/status/')
      .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(data => {
        errCount = 0;
        const status = data.status;
        const msg    = data.message || '';

        if (status === 'running') {
          setNav('running', msg || 'Running pipeline…');
          setProgress(true, msg);

        } else if (status === 'done') {
          stop();
          setNav('done', '✓ Analysis complete');
          Toast.show('Pipeline completed successfully! Results are ready.', 'success', 5000);
          setTimeout(() => window.location.reload(), 1200);

        } else if (status === 'question') {
          stop();
          setNav('', '');
        } else if (status === 'error') {
          stop();
          setNav('error', '⚠ Execution failed');
          Toast.show(msg || 'Pipeline failed during execution.', 'error', 8000);

        } else {
          setNav('', '');
          setProgress(false, '');
        }
      })
      .catch(() => {
        errCount++;
        if (errCount > 5) {
          stop();
        }
      });
  }

  function start() {
    setRunBtn(true);
    setNav('running', 'Initialising…');
    setProgress(true, 'Starting pipeline…');
    LiveLog.start();
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(poll, 1800);
  }

  return { start, stop, poll };
})();

// ── Run pipeline form ─────────────────────────────────────────
function initRunForm() {
  const form = document.getElementById('run-form');
  if (!form) return;

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData(form);
    StatusBar.start();
    try {
      const resp = await fetch('/run/', { method: 'POST', body: fd });
      const text = await resp.text();
      let data = {};
      try {
        data = JSON.parse(text);
      } catch (_) {
        if (!resp.ok) {
          throw new Error(`Server returned HTTP ${resp.status} (${resp.statusText})`);
        }
      }
      if (data.error) {
        StatusBar.stop();
        Toast.show('Error: ' + data.error, 'error');
      }
    } catch (err) {
      StatusBar.stop();
      Toast.show('Network error: ' + err.message, 'error');
    }
  });
}

// ── Reset session ─────────────────────────────────────────────
function initResetBtn() {
  const btn = document.getElementById('reset-btn');
  if (!btn) return;
  btn.addEventListener('click', () => {
    if (!confirm('Reset session and clear all results?')) return;
    const fd = new FormData();
    fd.append('csrfmiddlewaretoken', getCsrfToken());
    fetch('/reset/', { method: 'POST', body: fd })
      .then(() => {
        Toast.show('Session reset.', 'info', 2000);
        setTimeout(() => window.location.reload(), 600);
      });
  });
}

// ── Agent clarifying question modal ───────────────────────────
function openQuestionModal(question) {
  const overlay = document.getElementById('question-modal');
  const text    = document.getElementById('question-text');
  if (!overlay) return;
  if (text) text.textContent = question;
  overlay.classList.add('visible');
}

function initQuestionModal() {
  const overlay = document.getElementById('question-modal');
  if (!overlay) return;

  const q = overlay.dataset.question;
  if (q) openQuestionModal(q);

  const form = document.getElementById('question-form');
  if (!form) return;

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData(form);
    fd.append('csrfmiddlewaretoken', getCsrfToken());
    overlay.classList.remove('visible');
    StatusBar.start();
    try {
      await fetch('/submit-answer/', { method: 'POST', body: fd });
    } catch (err) {
      StatusBar.stop();
      Toast.show('Error: ' + err.message, 'error');
    }
  });
}

// ── Iframe report tab: lazy load ──────────────────────────────
function initReportIframe() {
  const iframe = document.getElementById('report-iframe');
  if (!iframe || iframe.dataset.loaded) return;
  const src = iframe.dataset.src;
  if (src) { iframe.src = src; iframe.dataset.loaded = 'true'; }
}

// ── Data Preview tab ──────────────────────────────────────────
const DataPreview = (() => {
  let loaded = false;

  const DTYPE_BADGE = {
    numeric:    '<span class="badge badge-numeric">N</span>',
    categorical:'<span class="badge badge-categ">C</span>',
    bool:       '<span class="badge badge-bool">B</span>',
    datetime:   '<span class="badge badge-datetime">D</span>',
  };

  function load() {
    if (loaded) return;
    const container = document.getElementById('preview-table-container');
    const loading   = document.getElementById('preview-loading');
    const errBox    = document.getElementById('preview-error');
    const errMsg    = document.getElementById('preview-error-msg');
    if (!container) return;

    fetch('/api/preview/')
      .then(r => r.json())
      .then(data => {
        if (loading) loading.style.display = 'none';

        if (data.error && !data.rows.length) {
          if (errBox) { errBox.style.display = 'flex'; }
          if (errMsg) errMsg.textContent = data.error;
          return;
        }

        const columns = data.columns || [];
        const rows    = data.rows || [];

        // Info bar
        const info = document.createElement('div');
        info.className = 'alert-banner info';
        info.style.marginBottom = '12px';
        info.innerHTML = `<span class="alert-icon">ℹ</span>
          <span>Showing <strong>${data.preview_rows}</strong> of <strong>${data.total_rows.toLocaleString()}</strong> total rows
          &nbsp;·&nbsp; <strong>${columns.length}</strong> columns</span>`;
        container.appendChild(info);

        // Build table
        const wrap = document.createElement('div');
        wrap.className = 'preview-table-wrap';

        const table = document.createElement('table');
        table.className = 'data-table';

        // Header
        const thead = document.createElement('thead');
        const hrow  = document.createElement('tr');
        hrow.innerHTML = '<th>#</th>';
        columns.forEach(col => {
          const th = document.createElement('th');
          th.innerHTML = `${DTYPE_BADGE[col.dtype] || ''} ${col.name}`;
          th.title = `${col.name} (${col.pandas_dtype})`;
          hrow.appendChild(th);
        });
        thead.appendChild(hrow);
        table.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');
        rows.forEach((row, idx) => {
          const tr = document.createElement('tr');
          const rowNum = document.createElement('td');
          rowNum.textContent = idx + 1;
          rowNum.style.color = 'var(--text-dim)';
          rowNum.style.fontSize = '0.68rem';
          tr.appendChild(rowNum);
          columns.forEach(col => {
            const td = document.createElement('td');
            const v = row[col.name];
            if (v === null || v === undefined) {
              td.textContent = 'null';
              td.className = 'td-null';
            } else {
              td.textContent = String(v);
            }
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
        table.appendChild(tbody);

        wrap.appendChild(table);
        container.appendChild(wrap);
        container.style.display = 'block';
        loaded = true;
      })
      .catch(err => {
        if (loading) loading.style.display = 'none';
        if (errBox) errBox.style.display = 'flex';
        if (errMsg) errMsg.textContent = 'Failed to load preview: ' + err.message;
      });
  }

  function reset() { loaded = false; }

  return { load, reset };
})();

// ── Mobile sidebar toggle ─────────────────────────────────────
function initSidebarToggle() {
  const toggle  = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  if (!toggle || !sidebar) return;

  function close() {
    sidebar.classList.remove('open');
    if (overlay) overlay.style.display = 'none';
  }

  toggle.addEventListener('click', () => {
    const isOpen = sidebar.classList.toggle('open');
    if (overlay) overlay.style.display = isOpen ? 'block' : 'none';
  });

  if (overlay) overlay.addEventListener('click', close);
}

// ── Theme Toggle module ──────────────────────────────────────
function initThemeToggle() {
  const checkbox = document.getElementById('theme-toggle-checkbox');

  function applyTheme(theme) {
    if (theme === 'light') {
      document.documentElement.setAttribute('data-theme', 'light');
      if (checkbox) checkbox.checked = true;
    } else {
      document.documentElement.removeAttribute('data-theme');
      if (checkbox) checkbox.checked = false;
    }

    // Live sync to embedded report iframe
    const reportIframe = document.getElementById('report-iframe');
    if (reportIframe && reportIframe.contentWindow) {
      try {
        if (reportIframe.contentWindow.toggleReportTheme) {
          reportIframe.contentWindow.toggleReportTheme(theme === 'light');
        }
      } catch (e) {}
    }
  }

  const savedTheme = localStorage.getItem('autoeda-theme') || 'dark';
  applyTheme(savedTheme);

  if (checkbox) {
    checkbox.addEventListener('change', () => {
      const nextTheme = checkbox.checked ? 'light' : 'dark';
      localStorage.setItem('autoeda-theme', nextTheme);
      applyTheme(nextTheme);
    });
  }
}

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();

  // Primary tabs
  initTabs('.tab-btn[data-tab]', '.tab-panel[id]', 'eda-active-tab');

  // Sub-tabs
  document.querySelectorAll('.sub-tab-bar').forEach(bar => {
    const scopeId = bar.dataset.scope;
    const btns    = bar.querySelectorAll('.sub-tab-btn');
    const panels  = scopeId
      ? document.querySelectorAll(`.sub-tab-panel[data-scope="${scopeId}"]`)
      : bar.closest('.tab-panel').querySelectorAll('.sub-tab-panel');

    btns.forEach(btn => btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.toggle('active', b === btn));
      panels.forEach(p => p.classList.toggle('active', p.dataset.subtab === btn.dataset.subtab));
    }));
    if (btns[0]) btns[0].click();
  });

  initAccordions();
  initVariableSearch();
  initDataSourceToggle();
  initFileDropZone();
  initRunForm();
  initResetBtn();
  initQuestionModal();
  initSidebarToggle();

  // If page has a running pipeline already (e.g. page refreshed mid-run)
  const runningIndicator = document.getElementById('pipeline-running-indicator');
  if (runningIndicator) {
    StatusBar.start();
  }
});
