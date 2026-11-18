// Round 4j / S-39: loads the REAL <script> of sophon_live/static/index.html in a vm with a stubbed DOM / WebSocket / network / virtual timers,
// feeds it `session.delegation.created` events through its own onEvent(), advances virtual time by a minute and reports everything the page did.
// Usage: node page_harness.js <path-to-index.html>   -> one JSON object on stdout.
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) { console.log(JSON.stringify({error: "no <script> in page"})); process.exit(0); }

const sentWs = [], network = [], timers = [];
let now = 0, tid = 0;
const els = {};
function mkEl(sel) {
  const t = {value: "", textContent: "", innerHTML: "", className: "", style: {}, children: [], dataset: {}, onclick: null, oninput: null, onchange: null,
             classList: {add() {}, remove() {}, toggle() {}, contains() { return false; }}, scrollTop: 0, scrollHeight: 0, width: 0, height: 0, sel};
  return new Proxy(t, {get(o, p) { if (p in o) return o[p]; if (typeof p === "symbol") return undefined; return function () { return mkEl(sel + "." + String(p)); }; },
                       set(o, p, v) { o[p] = v; return true; }});
}
const el = sel => els[sel] || (els[sel] = mkEl(sel));
class FakeWS { constructor(u) { this.url = u; this.readyState = 1; FakeWS.last = this; } send(d) { sentWs.push(JSON.parse(d)); } close() {} }
const sandbox = {
  console: {log() {}, warn() {}, error() {}},
  document: {querySelector: el, querySelectorAll: () => [], getElementById: el,
             createElement: () => { const e = mkEl("created"); e.querySelector = s => el("card" + s); return e; }, addEventListener() {}, body: el("body")},
  location: {protocol: "http:", host: "h:1", pathname: "/", href: "http://h:1/"},
  WebSocket: FakeWS,
  fetch: (...a) => { network.push("fetch " + a[0]); return Promise.resolve({json: async () => ({}), text: async () => ""}); },
  XMLHttpRequest: function () { this.open = (mt, u) => network.push("xhr " + u); this.send = () => {}; this.setRequestHeader = () => {}; },
  navigator: {mediaDevices: {getUserMedia: async () => ({getTracks: () => []})}, sendBeacon: u => { network.push("beacon " + u); return true; }},
  performance: {now: () => now}, atob, btoa, Blob: function () {}, URL: {createObjectURL: () => "blob:x"}, AudioContext: function () {}, AudioWorkletNode: function () {},
  setTimeout: (f, ms = 0) => { const id = ++tid; timers.push({id, at: now + (+ms || 0), f, every: null}); return id; },
  setInterval: (f, ms = 0) => { const id = ++tid; timers.push({id, at: now + Math.max(1, +ms || 0), f, every: Math.max(1, +ms || 0)}); return id; },
  clearTimeout: id => { const i = timers.findIndex(t => t.id === id); if (i >= 0) timers.splice(i, 1); },
  clearInterval: id => { const i = timers.findIndex(t => t.id === id); if (i >= 0) timers.splice(i, 1); },
  requestAnimationFrame: f => { timers.push({id: ++tid, at: now + 16, f, every: null}); return tid; },
  queueMicrotask,
};
sandbox.window = sandbox; sandbox.globalThis = sandbox; sandbox.self = sandbox;
vm.createContext(sandbox);

async function settle(ms) {                       // run microtasks and every timer due within `ms` of virtual time
  const end = now + ms;
  for (let guard = 0; guard < 5000; guard++) {
    for (let i = 0; i < 20; i++) await Promise.resolve();
    timers.sort((a, b) => a.at - b.at);
    const t = timers[0];
    if (!t || t.at > end) break;
    now = Math.max(now, t.at);
    if (t.every) t.at = now + t.every; else timers.shift();
    try { await t.f(); } catch (e) {}
  }
  now = end;
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

(async () => {
  const report = {};
  try {
    vm.runInContext(m[1], sandbox);                                           // the page's own top-level code
    vm.runInContext("ws = new WebSocket('ws://x/v1/live/sessions'); running = true; curUser = {textContent: '今天北京天气怎么样'};", sandbox);
    const cases = JSON.parse(process.argv[3] || "[]");
    for (const c of cases) {
      sentWs.length = 0; network.length = 0; els["#manual"] = mkEl("#manual"); els["#mtext"] = mkEl("#mtext");
      vm.runInContext("pendingDeleg = null;", sandbox);
      sandbox.__ev = {type: "session.delegation.created", delegation: Object.assign({id: "item_x", task: "北京今天天气怎么样"}, c.delegation)};
      vm.runInContext("onEvent(__ev)", sandbox);                                // the page's real event entry point
      await settle(60000);
      const mid = {sent: sentWs.slice(), network: network.slice(), manual_box_shown: els["#manual"].style.display === "", pending: vm.runInContext("pendingDeleg !== null", sandbox)};
      if (c.type_manual) {                                                      // positive control: the user's own manual answer IS sent
        els["#mtext"].value = "手动回答";
        sentWs.length = 0;
        await els["#msend"].onclick();
        await settle(1000);
        mid.after_manual_click = sentWs.slice();
      }
      report[c.name] = mid;
    }
  } catch (e) { report.error = String(e && e.stack || e); }
  console.log(JSON.stringify(report));
})();
