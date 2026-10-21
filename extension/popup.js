const DEFAULT_SERVER = "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);

async function currentTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function checkServer(server) {
  const out = $("server-status");
  try {
    const res = await fetch(`${server}/api/status`);
    const s = await res.json();
    const state = { siap: "siap", memuat: "sedang memuat model...", belum: "model belum dimuat", gagal: `gagal: ${s.error}` };
    out.textContent = `Server terhubung. ${s.llm_model}: ${state[s.state] || s.state}.`;
    out.className = s.state === "gagal" ? "error" : "muted";
  } catch (e) {
    out.textContent = `Server tidak bisa dihubungi di ${server}. Jalankan: python cli.py serve`;
    out.className = "error";
  }
}

async function showStatus(tabId) {
  const key = `status_${tabId}`;
  const data = (await chrome.storage.session.get(key))[key];
  if (data) $("status").textContent = data.text;
}

(async () => {
  const saved = await chrome.storage.local.get(["server", "position", "autoNext"]);
  $("server").value = saved.server || DEFAULT_SERVER;
  $("position").value = saved.position || "";
  $("auto-next").checked = saved.autoNext !== false; // default aktif

  $("auto-next").addEventListener("change", async () => {
    const on = $("auto-next").checked;
    await chrome.storage.local.set({ autoNext: on });
    const tab = await currentTab();
    if (tab) chrome.runtime.sendMessage({ action: "setAuto", tabId: tab.id, on });
  });
  checkServer($("server").value);
  const tab = await currentTab();
  if (tab) showStatus(tab.id);

  $("server").addEventListener("change", () => checkServer($("server").value.replace(/\/$/, "")));

  $("go").onclick = async () => {
    const server = $("server").value.trim().replace(/\/$/, "") || DEFAULT_SERVER;
    await chrome.storage.local.set({ server, position: $("position").value.trim() });
    const tab = await currentTab();
    if (!tab) return;
    $("go").disabled = true;
    await chrome.runtime.sendMessage({ action: "draft", tabId: tab.id });
    $("status").textContent = "Berjalan. Lihat banner di halaman; popup ini boleh ditutup.";
  };
})();
