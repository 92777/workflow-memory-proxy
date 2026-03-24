from __future__ import annotations


def render_dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Memory Proxy Dashboard</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255, 250, 241, 0.92);
      --panel-strong: #fff;
      --line: #d9ccb5;
      --text: #1f1b16;
      --muted: #6c6255;
      --accent: #0a7b83;
      --accent-soft: #d8f0f2;
      --warn: #b35c1e;
      --good: #27724b;
      --shadow: 0 12px 32px rgba(31, 27, 22, 0.08);
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      --sans: "Avenir Next", "Segoe UI", "PingFang SC", "Hiragino Sans GB", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(10, 123, 131, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(179, 92, 30, 0.12), transparent 22%),
        linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
    }
    a { color: inherit; }
    button, input, select, textarea {
      font: inherit;
      color: inherit;
    }
    .page {
      max-width: 1520px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .hero, .panel, .drawer {
      background: var(--panel);
      border: 1px solid rgba(217, 204, 181, 0.9);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    .hero {
      padding: 22px 24px;
      display: flex;
      gap: 16px;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: 32px;
      line-height: 1;
      letter-spacing: -0.03em;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 860px;
    }
    .hero-actions {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .panel {
      padding: 18px;
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: -0.02em;
    }
    .subtle {
      color: var(--muted);
      font-size: 13px;
    }
    .row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 12px;
    }
    .stat {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }
    .stat .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .stat .value {
      margin-top: 6px;
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }
    .main-grid {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 18px;
      align-items: start;
    }
    .stack {
      display: grid;
      gap: 18px;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 12px;
    }
    input, select, textarea, button {
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
    }
    input, select, textarea {
      width: 100%;
      padding: 10px 12px;
    }
    textarea {
      min-height: 88px;
      resize: vertical;
    }
    button {
      cursor: pointer;
      padding: 10px 14px;
      font-weight: 600;
      transition: transform 120ms ease, background 120ms ease;
    }
    button:hover {
      transform: translateY(-1px);
    }
    button.primary {
      background: linear-gradient(135deg, #0a7b83, #146b71);
      color: #fff;
      border-color: #0a7b83;
    }
    button.secondary {
      background: var(--accent-soft);
      color: var(--accent);
      border-color: rgba(10, 123, 131, 0.25);
    }
    .list {
      display: grid;
      gap: 10px;
      max-height: 330px;
      overflow: auto;
    }
    .list button {
      text-align: left;
      width: 100%;
      background: var(--panel-strong);
      border-radius: 16px;
      padding: 12px 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #fff;
    }
    .pill.good {
      color: var(--good);
      border-color: rgba(39, 114, 75, 0.25);
      background: rgba(39, 114, 75, 0.08);
    }
    .pill.warn {
      color: var(--warn);
      border-color: rgba(179, 92, 30, 0.25);
      background: rgba(179, 92, 30, 0.08);
    }
    .meta-card {
      border: 1px dashed var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.6);
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(120px, 1fr));
      gap: 10px;
    }
    .bar {
      height: 12px;
      border-radius: 999px;
      background: rgba(31, 27, 22, 0.08);
      overflow: hidden;
    }
    .bar > span {
      display: block;
      height: 100%;
      background: linear-gradient(90deg, #0a7b83, #b35c1e);
      width: 0%;
      transition: width 160ms ease;
    }
    .compare-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }
    pre {
      margin: 0;
      padding: 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #161412;
      color: #f7efe4;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.5;
      overflow: auto;
      min-height: 360px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .muted-box {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.5);
      color: var(--muted);
    }
    .fab {
      position: fixed;
      right: 22px;
      bottom: 22px;
      z-index: 30;
      box-shadow: 0 18px 32px rgba(10, 123, 131, 0.26);
    }
    .drawer {
      position: fixed;
      right: 22px;
      bottom: 88px;
      width: min(420px, calc(100vw - 24px));
      max-height: min(76vh, 820px);
      padding: 16px;
      z-index: 29;
      display: none;
      overflow: hidden;
    }
    .drawer.open {
      display: grid;
      gap: 12px;
    }
    .drawer-header {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
    }
    .drawer h2 {
      margin: 0;
      font-size: 18px;
      letter-spacing: -0.02em;
    }
    .chat-log {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      min-height: 220px;
      max-height: 280px;
      overflow: auto;
      display: grid;
      gap: 12px;
    }
    .bubble {
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #fffdf8;
    }
    .bubble.user {
      background: #eef7f8;
      border-color: rgba(10, 123, 131, 0.24);
    }
    .bubble.assistant {
      background: #fff9f3;
      border-color: rgba(179, 92, 30, 0.18);
    }
    .bubble .role {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .drawer-body {
      display: grid;
      gap: 12px;
      overflow: auto;
      padding-right: 2px;
    }
    @media (max-width: 1180px) {
      .main-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 900px) {
      .stats { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .compare-grid, .meta-grid { grid-template-columns: 1fr; }
      .drawer {
        right: 12px;
        left: 12px;
        width: auto;
        bottom: 84px;
      }
      .fab {
        right: 12px;
        bottom: 12px;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <h1 data-i18n="hero_title">Memory Proxy Dashboard</h1>
        <p data-i18n="hero_desc">Inspect recent request audits, watch compression reasons and token savings, and keep the chat tester tucked away in a small popup.</p>
      </div>
      <div class="hero-actions">
        <button class="secondary" id="lang-toggle" type="button">中文</button>
        <div class="pill" id="store-status">Store status: loading</div>
        <div class="pill" id="compact-alias-status">Compact alias: loading</div>
      </div>
    </section>

    <section class="panel">
      <h2 data-i18n="overview_title">Overview</h2>
      <div class="row" style="margin-bottom:12px">
        <span class="subtle" id="retention-note">Retention: loading</span>
      </div>
      <div class="stats">
        <div class="stat"><div class="label" data-i18n="stat_sessions">Sessions</div><div class="value" id="stat-sessions">0</div></div>
        <div class="stat"><div class="label" data-i18n="stat_requests">Requests</div><div class="value" id="stat-requests">0</div></div>
        <div class="stat"><div class="label" data-i18n="stat_compressed">Compressed</div><div class="value" id="stat-compressed">0</div></div>
        <div class="stat"><div class="label" data-i18n="stat_avg_savings">Avg Savings</div><div class="value" id="stat-savings">0%</div></div>
      </div>
    </section>

    <div class="main-grid">
      <div class="stack">
        <section class="panel">
          <h2 data-i18n="recent_sessions_title">Recent Sessions</h2>
          <div class="list" id="sessions-list"></div>
        </section>

        <section class="panel">
          <h2 data-i18n="recent_requests_title">Recent Requests</h2>
          <div class="toolbar">
            <input id="request-session-filter" data-i18n-placeholder="placeholder_filter_session" placeholder="Filter by session id" />
            <button class="secondary" id="filter-button" type="button" data-i18n="button_apply">Apply</button>
            <button class="secondary" id="clear-filter-button" type="button" data-i18n="button_clear">Clear</button>
          </div>
          <div class="list" id="requests-list"></div>
        </section>
      </div>

      <section class="panel">
        <h2 data-i18n="request_detail_title">Request Detail</h2>
        <div id="detail-empty" class="muted-box">Choose a request from the list to inspect the compression summary.</div>
        <div id="detail-content" style="display:none">
          <div class="meta-card">
            <div class="meta-grid">
              <div><div class="subtle" data-i18n="detail_api">API</div><div id="detail-api-kind">-</div></div>
              <div><div class="subtle" data-i18n="detail_model">Model</div><div id="detail-model">-</div></div>
              <div><div class="subtle" data-i18n="detail_status">Status</div><div id="detail-status">-</div></div>
              <div><div class="subtle" data-i18n="detail_compressed">Compressed</div><div id="detail-compressed">-</div></div>
              <div><div class="subtle" data-i18n="detail_estimated_tokens">Estimated Tokens</div><div id="detail-estimated-tokens">-</div></div>
              <div><div class="subtle" data-i18n="detail_upstream_usage">Upstream Usage</div><div id="detail-upstream-usage">-</div></div>
            </div>
          </div>
          <div class="meta-card" style="margin-top:12px">
            <div class="meta-grid">
              <div><div class="subtle" data-i18n="detail_reason">Reason</div><div id="detail-reason">-</div></div>
              <div><div class="subtle" data-i18n="detail_session">Session</div><div id="detail-session">-</div></div>
              <div><div class="subtle" data-i18n="detail_request">Request</div><div id="detail-request">-</div></div>
              <div><div class="subtle" data-i18n="detail_dropped">Dropped History</div><div id="detail-dropped">-</div></div>
              <div><div class="subtle" data-i18n="detail_recent">Recent Kept</div><div id="detail-recent">-</div></div>
              <div><div class="subtle" data-i18n="detail_snapshot">Snapshot</div><div id="detail-snapshot">-</div></div>
            </div>
          </div>
          <div class="subtle" style="margin:12px 0 6px" data-i18n="detail_savings_bar">Estimated input token savings</div>
          <div class="bar"><span id="savings-bar"></span></div>
          <div id="detail-savings-text" class="subtle" style="margin:8px 0 14px">-</div>
        </div>
      </section>
    </div>
  </div>

  <button class="primary fab" id="open-chat-button" type="button">Chat Test</button>

  <aside class="drawer" id="chat-drawer" aria-hidden="true">
    <div class="drawer-header">
      <div>
        <h2 data-i18n="chat_window_title">Chat Test</h2>
        <div class="subtle" id="chat-status">Ready.</div>
      </div>
      <button class="secondary" id="close-chat-button" type="button" data-i18n="button_close">Close</button>
    </div>
    <div class="drawer-body">
      <div class="toolbar" style="margin:0">
        <div style="flex:1 1 180px">
          <label class="subtle" for="model-select" data-i18n="label_model">Model</label>
          <select id="model-select"></select>
        </div>
        <div style="flex:1 1 180px">
          <label class="subtle" for="session-id" data-i18n="label_session_id">Session ID</label>
          <input id="session-id" data-i18n-placeholder="placeholder_session_id" placeholder="web_session_..." />
        </div>
      </div>
      <div>
        <label class="subtle" for="system-prompt" data-i18n="label_system_prompt">System Prompt (optional)</label>
        <textarea id="system-prompt" data-i18n-placeholder="placeholder_system_prompt" placeholder="You are a helpful assistant."></textarea>
      </div>
      <div class="chat-log" id="chat-log"></div>
      <div>
        <label class="subtle" for="user-input" data-i18n="label_next_user_message">Next User Message</label>
        <textarea id="user-input" data-i18n-placeholder="placeholder_user_input" placeholder="Ask something long enough to trigger compression..."></textarea>
      </div>
      <div class="row">
        <button class="primary" id="send-button" type="button" data-i18n="button_send">Send</button>
        <button class="secondary" id="reset-button" type="button" data-i18n="button_reset">Reset Conversation</button>
      </div>
      <div class="meta-card">
        <div class="meta-grid">
          <div><div class="subtle" data-i18n="meta_compressed">Compressed</div><div id="last-compressed">-</div></div>
          <div><div class="subtle" data-i18n="meta_dropped_history">Dropped History</div><div id="last-dropped">-</div></div>
          <div><div class="subtle" data-i18n="meta_estimated_savings">Estimated Savings</div><div id="last-savings">-</div></div>
          <div><div class="subtle" data-i18n="meta_reason">Reason</div><div id="last-reason">-</div></div>
          <div><div class="subtle" data-i18n="meta_request_id">Request ID</div><div id="last-request-id">-</div></div>
          <div><div class="subtle" data-i18n="meta_snapshot_id">Snapshot ID</div><div id="last-snapshot-id">-</div></div>
        </div>
      </div>
    </div>
  </aside>

  <script>
    const I18N = {
      en: {
        page_title: 'Memory Proxy Dashboard',
        hero_title: 'Memory Proxy Dashboard',
        hero_desc: 'Inspect recent request audits, watch compression reasons and token savings, and keep the chat tester tucked away in a small popup.',
        overview_title: 'Overview',
        stat_sessions: 'Sessions',
        stat_requests: 'Requests',
        stat_compressed: 'Compressed',
        stat_avg_savings: 'Avg Savings',
        recent_sessions_title: 'Recent Sessions',
        recent_requests_title: 'Recent Requests',
        request_detail_title: 'Request Detail',
        chat_window_title: 'Chat Test',
        label_model: 'Model',
        label_session_id: 'Session ID',
        label_system_prompt: 'System Prompt (optional)',
        label_next_user_message: 'Next User Message',
        button_send: 'Send',
        button_reset: 'Reset Conversation',
        button_apply: 'Apply',
        button_clear: 'Clear',
        button_close: 'Close',
        button_open_chat: 'Chat Test',
        meta_compressed: 'Compressed',
        meta_dropped_history: 'Dropped History',
        meta_estimated_savings: 'Estimated Savings',
        meta_reason: 'Reason',
        meta_request_id: 'Request ID',
        meta_snapshot_id: 'Snapshot ID',
        detail_api: 'API',
        detail_model: 'Model',
        detail_status: 'Status',
        detail_compressed: 'Compressed',
        detail_estimated_tokens: 'Estimated Tokens',
        detail_upstream_usage: 'Upstream Usage',
        detail_savings_bar: 'Estimated input token savings',
        detail_reason: 'Reason',
        detail_session: 'Session',
        detail_request: 'Request',
        detail_dropped: 'Dropped History',
        detail_recent: 'Recent Kept',
        detail_snapshot: 'Snapshot',
        placeholder_session_id: 'web_session_...',
        placeholder_system_prompt: 'You are a helpful assistant.',
        placeholder_user_input: 'Ask something long enough to trigger compression...',
        placeholder_filter_session: 'Filter by session id',
        store_enabled: 'Store enabled',
        store_disabled: 'Store disabled',
        compact_alias_deprecated: 'Legacy compact alias deprecated',
        compact_alias_unknown: 'Compact alias status unavailable',
        compact_alias_title: 'Legacy alias {path} sunsets at {sunset}. Use {successor}.',
        retention: 'Retention: latest {count} request audits',
        ready: 'Ready.',
        sending: 'Sending...',
        reply_received: 'Reply received.',
        request_failed: 'Request failed.',
        conversation_cleared: 'Conversation cleared.',
        no_messages: 'No messages yet. Send a message to test compression behavior.',
        no_sessions: 'No persisted sessions yet. Enable store mode to inspect history.',
        no_requests: 'No request audits to show.',
        detail_empty: 'Choose a request from the list to inspect the compression summary.',
        compressed_label: 'compressed',
        raw_label: 'raw',
        session_label: 'session',
        est_label: 'est',
        last_label: 'last',
        no_estimated_savings: 'No estimated savings available.',
        estimated_reduction: '{pct}% estimated input reduction using {counter}.',
        role_user: 'user',
        role_assistant: 'assistant',
        role_system: 'system',
        role_tool: 'tool',
        lang_toggle: '中文'
      },
      zh: {
        page_title: '记忆压缩代理面板',
        hero_title: '记忆压缩代理面板',
        hero_desc: '主页面专注查看请求记录、压缩原因和 token 变化，对话测试则收纳到右下角的小弹窗里。',
        overview_title: '概览',
        stat_sessions: '会话数',
        stat_requests: '请求数',
        stat_compressed: '已压缩',
        stat_avg_savings: '平均压缩率',
        recent_sessions_title: '最近会话',
        recent_requests_title: '最近请求',
        request_detail_title: '请求详情',
        chat_window_title: '对话测试',
        label_model: '模型',
        label_session_id: '会话 ID',
        label_system_prompt: '系统提示词（可选）',
        label_next_user_message: '下一条用户消息',
        button_send: '发送',
        button_reset: '重置对话',
        button_apply: '应用',
        button_clear: '清空',
        button_close: '关闭',
        button_open_chat: '对话测试',
        meta_compressed: '是否压缩',
        meta_dropped_history: '裁剪历史数',
        meta_estimated_savings: '估算压缩率',
        meta_reason: '原因',
        meta_request_id: '请求 ID',
        meta_snapshot_id: '快照 ID',
        detail_api: '接口',
        detail_model: '模型',
        detail_status: '状态码',
        detail_compressed: '是否压缩',
        detail_estimated_tokens: '估算 Tokens',
        detail_upstream_usage: '上游 Usage',
        detail_savings_bar: '估算输入 Token 压缩率',
        detail_reason: '原因',
        detail_session: '会话',
        detail_request: '请求',
        detail_dropped: '裁剪历史数',
        detail_recent: '保留最近数',
        detail_snapshot: '快照',
        placeholder_session_id: 'web_session_...',
        placeholder_system_prompt: '你是一个有帮助的助手。',
        placeholder_user_input: '输入一段足够长、可能触发压缩的对话内容...',
        placeholder_filter_session: '按 session id 过滤',
        store_enabled: '存储已开启',
        store_disabled: '存储未开启',
        compact_alias_deprecated: '旧 compact 别名已废弃',
        compact_alias_unknown: 'compact 别名状态未知',
        compact_alias_title: '旧别名 {path} 将在 {sunset} 下线，请改用 {successor}。',
        retention: '保留策略：仅保留最近 {count} 条请求审计',
        ready: '准备就绪。',
        sending: '发送中...',
        reply_received: '已收到回复。',
        request_failed: '请求失败。',
        conversation_cleared: '对话已清空。',
        no_messages: '还没有消息，先发一条来测试压缩效果。',
        no_sessions: '还没有持久化会话。开启 store 模式后，这里会显示历史。',
        no_requests: '暂无请求审计记录。',
        detail_empty: '从左侧选择一条请求，就能在这里查看压缩摘要。',
        compressed_label: '已压缩',
        raw_label: '未压缩',
        session_label: '会话',
        est_label: '估算',
        last_label: '最近',
        no_estimated_savings: '暂无估算压缩率。',
        estimated_reduction: '使用 {counter} 估算，输入减少约 {pct}%。',
        role_user: '用户',
        role_assistant: '助手',
        role_system: '系统',
        role_tool: '工具',
        lang_toggle: 'EN'
      }
    };

    const state = {
      messages: [],
      filterSessionId: '',
      lang: localStorage.getItem('memory_proxy_lang') || 'zh',
      storeMaxRequests: null,
      compactAlias: null,
      chatOpen: false,
    };

    function t(key, params = {}) {
      const dict = I18N[state.lang] || I18N.en;
      let text = dict[key] || I18N.en[key] || key;
      for (const [paramKey, paramValue] of Object.entries(params)) {
        text = text.replaceAll(`{${paramKey}}`, String(paramValue));
      }
      return text;
    }

    function pretty(value) {
      if (value === null || value === undefined || value === '') return '-';
      if (typeof value === 'string') return value;
      return JSON.stringify(value, null, 2);
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    function roleLabel(role) {
      return t(`role_${role}`) || role;
    }

    function applyTranslations() {
      document.documentElement.lang = state.lang === 'zh' ? 'zh-CN' : 'en';
      document.title = t('page_title');
      document.getElementById('lang-toggle').textContent = t('lang_toggle');
      document.getElementById('open-chat-button').textContent = t('button_open_chat');
      document.querySelectorAll('[data-i18n]').forEach((element) => {
        element.textContent = t(element.dataset.i18n);
      });
      document.querySelectorAll('[data-i18n-placeholder]').forEach((element) => {
        element.placeholder = t(element.dataset.i18nPlaceholder);
      });
      if (state.storeMaxRequests !== null) {
        document.getElementById('retention-note').textContent = t('retention', {count: state.storeMaxRequests});
      }
      applyCompactAliasStatus();
      renderChat();
      if (document.getElementById('detail-content').style.display === 'none' || !document.getElementById('detail-content').style.display) {
        document.getElementById('detail-empty').textContent = t('detail_empty');
      }
      const status = document.getElementById('chat-status');
      if (!status.dataset.state || status.dataset.state === 'ready') {
        status.textContent = t('ready');
      }
    }

    function setChatDrawer(open) {
      state.chatOpen = open;
      const drawer = document.getElementById('chat-drawer');
      drawer.classList.toggle('open', open);
      drawer.setAttribute('aria-hidden', open ? 'false' : 'true');
    }

    function ensureSessionId() {
      const input = document.getElementById('session-id');
      if (!input.value.trim()) {
        input.value = 'web_' + Date.now().toString(36);
      }
      return input.value.trim();
    }

    function renderChat() {
      const log = document.getElementById('chat-log');
      if (!state.messages.length) {
        log.innerHTML = `<div class="muted-box">${escapeHtml(t('no_messages'))}</div>`;
        return;
      }
      log.innerHTML = state.messages.map((msg) => `
        <div class="bubble ${msg.role}">
          <div class="role">${escapeHtml(roleLabel(msg.role))}</div>
          <div>${escapeHtml(msg.content)}</div>
        </div>
      `).join('');
      log.scrollTop = log.scrollHeight;
    }

    async function loadModels() {
      const select = document.getElementById('model-select');
      try {
        const response = await fetch('/v1/models');
        const body = await response.json();
        const models = Array.isArray(body.data) ? body.data : [];
        if (!models.length) {
          select.innerHTML = '<option value="gpt-5.4-mini">gpt-5.4-mini</option>';
          return;
        }
        select.innerHTML = models
          .map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)}</option>`)
          .join('');
        const preferred = models.find((item) => item.id.includes('mini')) || models[0];
        select.value = preferred.id;
      } catch (error) {
        select.innerHTML = '<option value="gpt-5.4-mini">gpt-5.4-mini</option>';
      }
    }

    async function loadSummary() {
      const response = await fetch('/api/dashboard/summary');
      const body = await response.json();
      document.getElementById('stat-sessions').textContent = String(body.summary.sessions || 0);
      document.getElementById('stat-requests').textContent = String(body.summary.requests || 0);
      document.getElementById('stat-compressed').textContent = String(body.summary.compressed_requests || 0);
      const avg = body.summary.avg_estimated_savings_pct;
      document.getElementById('stat-savings').textContent = avg === null ? '-' : `${avg}%`;
      state.storeMaxRequests = body.store_max_requests;
      document.getElementById('retention-note').textContent = t('retention', {
        count: body.store_max_requests === null ? '-' : body.store_max_requests
      });
      const status = document.getElementById('store-status');
      if (body.store_enabled) {
        status.textContent = t('store_enabled');
        status.className = 'pill good';
      } else {
        status.textContent = t('store_disabled');
        status.className = 'pill warn';
      }
    }

    function applyCompactAliasStatus() {
      const pill = document.getElementById('compact-alias-status');
      if (!state.compactAlias) {
        pill.textContent = t('compact_alias_unknown');
        pill.className = 'pill warn';
        pill.title = '';
        return;
      }
      if (state.compactAlias.deprecated) {
        pill.textContent = t('compact_alias_deprecated');
        pill.className = 'pill warn';
      } else {
        pill.textContent = t('compact_alias_unknown');
        pill.className = 'pill';
      }
      pill.title = t('compact_alias_title', {
        path: state.compactAlias.path || '/v1/responses/compact',
        successor: state.compactAlias.successor || '/v1/proxy/responses/compact',
        sunset: state.compactAlias.sunset || '-'
      });
    }

    async function loadHealth() {
      try {
        const response = await fetch('/health');
        const body = await response.json();
        state.compactAlias = body.legacy_compact_alias || null;
      } catch (error) {
        state.compactAlias = null;
      }
      applyCompactAliasStatus();
    }

    async function loadSessions() {
      const response = await fetch('/api/dashboard/sessions');
      const body = await response.json();
      const list = document.getElementById('sessions-list');
      if (!body.items || !body.items.length) {
        list.innerHTML = `<div class="muted-box">${escapeHtml(t('no_sessions'))}</div>`;
        return;
      }
      list.innerHTML = body.items.map((item) => `
        <button data-session-id="${escapeHtml(item.session_id)}">
          <div class="row" style="justify-content:space-between">
            <strong>${escapeHtml(item.session_id)}</strong>
            <span class="pill ${item.compressed_request_count > 0 ? 'good' : ''}">${item.compressed_request_count}/${item.request_count} ${escapeHtml(t('compressed_label'))}</span>
          </div>
          <div class="subtle" style="margin-top:6px">${escapeHtml(t('last_label'))} ${escapeHtml(item.last_request_at || item.updated_at || '-')}</div>
        </button>
      `).join('');
      list.querySelectorAll('button').forEach((button) => {
        button.addEventListener('click', () => {
          const sessionId = button.getAttribute('data-session-id') || '';
          document.getElementById('request-session-filter').value = sessionId;
          document.getElementById('session-id').value = sessionId;
          state.filterSessionId = sessionId;
          loadRequests();
        });
      });
    }

    async function loadRequests() {
      const query = state.filterSessionId ? `?session_id=${encodeURIComponent(state.filterSessionId)}` : '';
      const response = await fetch(`/api/dashboard/requests${query}`);
      const body = await response.json();
      const list = document.getElementById('requests-list');
      if (!body.items || !body.items.length) {
        list.innerHTML = `<div class="muted-box">${escapeHtml(t('no_requests'))}</div>`;
        return;
      }
      list.innerHTML = body.items.map((item) => `
        <button data-request-id="${escapeHtml(item.request_id)}">
          <div class="row" style="justify-content:space-between">
            <strong>${escapeHtml(item.request_id)}</strong>
            <span class="pill ${item.compressed ? 'good' : 'warn'}">${escapeHtml(item.compressed ? t('compressed_label') : t('raw_label'))}</span>
          </div>
          <div class="subtle" style="margin-top:6px">${escapeHtml(item.api_kind)} | ${escapeHtml(item.upstream_model || '-')}</div>
          <div class="subtle">${escapeHtml(t('session_label'))} ${escapeHtml(item.session_id)} | ${escapeHtml(t('est_label'))} ${item.estimated_savings_pct === null ? '-' : `${item.estimated_savings_pct}%`}</div>
        </button>
      `).join('');
      list.querySelectorAll('button').forEach((button) => {
        button.addEventListener('click', () => loadRequestDetail(button.getAttribute('data-request-id')));
      });
    }

    async function loadRequestDetail(requestId) {
      if (!requestId) return;
      const response = await fetch(`/api/dashboard/requests/${encodeURIComponent(requestId)}`);
      if (!response.ok) return;
      const body = await response.json();
      const request = body.request;
      document.getElementById('detail-empty').style.display = 'none';
      document.getElementById('detail-content').style.display = 'block';
      document.getElementById('detail-api-kind').textContent = request.api_kind;
      document.getElementById('detail-model').textContent = request.upstream_model || '-';
      document.getElementById('detail-status').textContent = request.status_code || '-';
      document.getElementById('detail-compressed').textContent = request.compressed ? 'true' : 'false';
      document.getElementById('detail-estimated-tokens').textContent = `${request.estimated_input_tokens_before || '-'} -> ${request.estimated_input_tokens_after || '-'}`;
      document.getElementById('detail-upstream-usage').textContent = request.upstream_total_tokens === null
        ? '-'
        : `${request.upstream_input_tokens || '-'} in / ${request.upstream_output_tokens || '-'} out / ${request.upstream_total_tokens} total`;
      document.getElementById('detail-reason').textContent = request.compression_reason || '-';
      document.getElementById('detail-session').textContent = request.session_id || '-';
      document.getElementById('detail-request').textContent = request.request_id || '-';
      document.getElementById('detail-dropped').textContent = request.dropped_message_count ?? '-';
      document.getElementById('detail-recent').textContent = request.recent_message_count ?? '-';
      document.getElementById('detail-snapshot').textContent = request.snapshot_id || '-';
      const savings = typeof request.estimated_savings_pct === 'number' ? request.estimated_savings_pct : 0;
      document.getElementById('savings-bar').style.width = `${Math.max(0, Math.min(100, savings))}%`;
      document.getElementById('detail-savings-text').textContent = request.estimated_savings_pct === null
        ? t('no_estimated_savings')
        : t('estimated_reduction', {pct: request.estimated_savings_pct, counter: request.token_counter || 'unknown'});
    }

    async function sendMessage() {
      const userInput = document.getElementById('user-input');
      const model = document.getElementById('model-select').value;
      const systemPrompt = document.getElementById('system-prompt').value.trim();
      const sessionId = ensureSessionId();
      const content = userInput.value.trim();
      if (!content) return;

      state.messages.push({role: 'user', content});
      renderChat();
      userInput.value = '';
      const chatStatus = document.getElementById('chat-status');
      chatStatus.dataset.state = 'sending';
      chatStatus.textContent = t('sending');

      const messages = [];
      if (systemPrompt) {
        messages.push({role: 'system', content: systemPrompt});
      }
      for (const message of state.messages) {
        messages.push({role: message.role, content: message.content});
      }

      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-session-id': sessionId
        },
        body: JSON.stringify({model, messages})
      });
      const body = await response.json();
      const reply = (((body || {}).choices || [])[0] || {}).message || {};
      const assistantText = typeof reply.content === 'string' ? reply.content : JSON.stringify(reply.content || '');
      state.messages.push({role: 'assistant', content: assistantText});
      renderChat();

      document.getElementById('last-compressed').textContent = response.headers.get('x-memory-proxy-compressed') || '-';
      document.getElementById('last-dropped').textContent = response.headers.get('x-memory-proxy-history-dropped') || '-';
      const requestId = response.headers.get('x-memory-proxy-request-id') || '-';
      document.getElementById('last-request-id').textContent = requestId;
      document.getElementById('last-snapshot-id').textContent = response.headers.get('x-memory-proxy-snapshot-id') || '-';
      document.getElementById('last-reason').textContent = response.headers.get('x-memory-proxy-reason') || '-';
      const estimatedSavings = response.headers.get('x-memory-proxy-estimated-savings-pct') || '0';
      const beforeTokens = response.headers.get('x-memory-proxy-estimated-before-tokens') || '0';
      const afterTokens = response.headers.get('x-memory-proxy-estimated-after-tokens') || '0';
      document.getElementById('last-savings').textContent = `${estimatedSavings}% (${beforeTokens} -> ${afterTokens})`;
      chatStatus.dataset.state = response.ok ? 'ready' : 'failed';
      chatStatus.textContent = response.ok ? t('reply_received') : t('request_failed');

      await Promise.all([loadSummary(), loadSessions(), loadRequests()]);
      if (requestId && requestId !== '-') {
        await loadRequestDetail(requestId);
      }
    }

    function resetConversation() {
      state.messages = [];
      renderChat();
      const chatStatus = document.getElementById('chat-status');
      chatStatus.dataset.state = 'ready';
      chatStatus.textContent = t('conversation_cleared');
    }

    function applyFilter() {
      state.filterSessionId = document.getElementById('request-session-filter').value.trim();
      loadRequests();
    }

    function toggleLanguage() {
      state.lang = state.lang === 'zh' ? 'en' : 'zh';
      localStorage.setItem('memory_proxy_lang', state.lang);
      applyTranslations();
      loadSessions();
      loadRequests();
    }

    async function bootstrap() {
      ensureSessionId();
      applyTranslations();
      renderChat();
      await Promise.all([loadModels(), loadHealth(), loadSummary(), loadSessions(), loadRequests()]);
      document.getElementById('open-chat-button').addEventListener('click', () => setChatDrawer(true));
      document.getElementById('close-chat-button').addEventListener('click', () => setChatDrawer(false));
      document.getElementById('send-button').addEventListener('click', sendMessage);
      document.getElementById('reset-button').addEventListener('click', resetConversation);
      document.getElementById('filter-button').addEventListener('click', applyFilter);
      document.getElementById('clear-filter-button').addEventListener('click', () => {
        state.filterSessionId = '';
        document.getElementById('request-session-filter').value = '';
        loadRequests();
      });
      document.getElementById('lang-toggle').addEventListener('click', toggleLanguage);
      document.getElementById('user-input').addEventListener('keydown', (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
          sendMessage();
        }
      });
    }

    bootstrap();
  </script>
</body>
</html>
"""
