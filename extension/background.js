// Penghubung antara halaman lamaran dan server RAG lokal. Semua jawaban disusun oleh server di laptop
// (python cli.py serve); extension hanya membaca label kolom dan menulis draf. Next/Submit tetap ditekan pengguna.
const DEFAULTS = { server: "http://127.0.0.1:8000", position: "", autoNext: true };
const MAX_PER_REQUEST = 20; // batas parse_questions() di server
const running = new Set(); // tab yang sedang diproses
const pending = new Set(); // tab yang dapat kolom baru saat sedang diproses

async function getSettings() {
  return { ...DEFAULTS, ...(await chrome.storage.local.get(["server", "position", "autoNext"])) };
}

async function runInTab(tabId, func, ...args) {
  const [res] = await chrome.scripting.executeScript({ target: { tabId }, func, args });
  return res ? res.result : undefined;
}

async function status(tabId, text, state) {
  await runInTab(tabId, (t, s) => window.__ragLamaran.banner(t, s), text, state).catch(() => {});
  await chrome.storage.session.set({ [`status_${tabId}`]: { text, state, at: Date.now() } });
}

// Service worker MV3 bisa dimatikan Chrome setelah ~30 detik tanpa aktivitas extension. Menunggu jawaban
// LLM bisa lebih dari 1 menit, jadi selama request berjalan kita memanggil API ringan secara berkala.
function keepAlive() {
  const id = setInterval(() => chrome.runtime.getPlatformInfo(() => {}), 20000);
  return () => clearInterval(id);
}

// Kirim label ke /api/draft lalu tulis jawabannya ke kolom. Mengembalikan jumlah per status.
async function draftFields(tabId, fields) {
  const { server, position } = await getSettings();
  const questions = [...new Set(fields.map((f) => f.label))];
  const longQuestions = [...new Set(fields.filter((f) => f.tag === "textarea").map((f) => f.label))];
  const stop = keepAlive();
  try {
    const results = [];
    for (let i = 0; i < questions.length; i += MAX_PER_REQUEST) {
      const res = await fetch(`${server}/api/draft`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ questions_text: questions.slice(i, i + MAX_PER_REQUEST).join("\n"), position,
                               long_questions: longQuestions }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      results.push(...data);
    }
    return await runInTab(tabId, (f, r) => window.__ragLamaran.fill(f, r), fields, results);
  } finally {
    stop();
  }
}

const summary = (c) => `${c.ok || 0} terisi (hijau), ${c.needs_review || 0} perlu dicek (kuning), `
  + `${c.not_found || 0} tidak ditemukan atau isi sendiri (abu-abu)`;

// Isi kolom wajib, lalu tanyakan kolom tidak wajib lewat panel di halaman.
async function draftTab(tabId, { auto = false } = {}) {
  if (running.has(tabId)) { pending.add(tabId); return; }
  running.add(tabId);
  try {
    try {
      await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
    } catch (e) {
      // Mis. halaman chrome://, Chrome Web Store, PDF viewer, atau tab sudah pindah ke situs lain.
      await chrome.storage.session.set({ [`status_${tabId}`]: {
        text: `Extension tidak bisa berjalan di halaman ini (${e.message}).`, state: "error", at: Date.now() } });
      return;
    }
    const { autoNext } = await getSettings();
    await runInTab(tabId, (on) => window.__ragLamaran.watch(on), autoNext);

    const { fields, pageMarksRequired } = await runInTab(tabId, () => window.__ragLamaran.collect());
    if (!fields.length) {
      if (!auto) await status(tabId, "Tidak ada kolom kosong yang bisa diisi di halaman ini.", "warn");
      return;
    }
    // Kalau halaman tidak menandai kolom wajib sama sekali, semua kolom ditanyakan dulu.
    const required = pageMarksRequired ? fields.filter((f) => f.required) : [];
    const optional = pageMarksRequired ? fields.filter((f) => !f.required) : fields;

    let text = "";
    if (required.length) {
      await status(tabId, `${auto ? "Halaman baru terdeteksi. " : ""}Menyusun draf untuk ${required.length} kolom wajib. `
        + "Bisa makan 1-2 menit; boleh pindah ke tab lain.", "busy");
      text = `Kolom wajib selesai: ${summary(await draftFields(tabId, required))}.`;
    }
    if (optional.length) {
      text += ` ${optional.length} kolom tidak wajib menunggu pilihanmu di panel kanan bawah.`;
      await runInTab(tabId, (f, m) => window.__ragLamaran.askOptional(f, m), optional, pageMarksRequired);
    }
    await status(tabId, `${text.trim()} Arahkan kursor ke kolom untuk melihat sumbernya. `
      + "Cek semua isian, lalu tekan Next/Submit sendiri.", "done");
  } catch (e) {
    const { server } = await getSettings();
    await status(tabId, `Gagal: ${e.message}. Pastikan server jalan (python cli.py serve) di ${server}.`, "error");
  } finally {
    running.delete(tabId);
    if (pending.delete(tabId)) draftTab(tabId, { auto: true });
  }
}

// Kolom tidak wajib yang dipilih "Isi dari dokumen" di panel.
async function draftSelected(tabId, fields) {
  try {
    await status(tabId, `Menyusun draf untuk ${fields.length} kolom tidak wajib yang kamu pilih...`, "busy");
    const counts = await draftFields(tabId, fields);
    await status(tabId, `Kolom tidak wajib selesai: ${summary(counts)}. Cek lagi sebelum Next/Submit.`, "done");
  } catch (e) {
    await status(tabId, `Gagal: ${e.message}.`, "error");
  }
}

async function activeTabId() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab && tab.id;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  const tabId = msg.tabId || sender.tab?.id;
  if (!tabId) return;
  if (msg.action === "draft") draftTab(tabId); // tetap jalan walau popup ditutup
  if (msg.action === "draftSelected") draftSelected(tabId, msg.fields || []);
  if (msg.action === "newFields") {
    getSettings().then(({ autoNext }) => { if (autoNext) draftTab(tabId, { auto: true }); });
  }
  if (msg.action === "setAuto") {
    runInTab(tabId, (on) => window.__ragLamaran?.watch(on), msg.on).catch(() => {});
  }
  sendResponse({ ok: true });
});

chrome.commands.onCommand.addListener(async (command) => {
  if (command === "isi-draf") {
    const tabId = await activeTabId();
    if (tabId) draftTab(tabId);
  }
});
