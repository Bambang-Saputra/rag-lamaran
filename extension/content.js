// Disuntikkan ke halaman lamaran oleh background.js. Tugasnya hanya membaca label kolom, menulis draf, dan
// menanyakan kolom tidak wajib. Tidak pernah menekan tombol halaman (Next, Submit, dll.) dan tidak menyentuh
// upload file atau CAPTCHA.
(() => {
  if (window.__ragLamaran) return;

  const SKIP_TYPES = new Set(["hidden", "submit", "button", "reset", "checkbox", "radio", "file", "password",
                              "image", "range", "color"]);
  const OWN_UI = "#rag-lamaran-banner, #rag-lamaran-optional";
  const COLORS = { ok: "#16a34a", needs_review: "#d97706", not_found: "#9ca3af", self: "#2563eb", skipped: "#9ca3af" };
  const STATUS_LABEL = { ok: "Terisi", needs_review: "Perlu dicek", not_found: "Tidak ditemukan / isi sendiri" };

  // Sama dengan parse_questions() di server: buang nomor di depan, tanda wajib (*) dan titik dua di belakang.
  const clean = (t) => (t || "").replace(/\s+/g, " ").replace(/^\s*\d+[.)]\s*/, "").replace(/[\s*:]+$/, "").trim();

  const textWithoutFields = (node) => {
    const copy = node.cloneNode(true);
    copy.querySelectorAll("input, textarea, select, button, option").forEach((n) => n.remove());
    return copy.textContent;
  };

  function labelFor(el) {
    if (el.id) {
      const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (l && clean(l.textContent)) return l.textContent;
    }
    const wrap = el.closest("label");
    if (wrap && clean(textWithoutFields(wrap))) return textWithoutFields(wrap);
    const by = el.getAttribute("aria-labelledby");
    if (by) {
      const t = by.split(/\s+/).map((id) => document.getElementById(id)?.textContent || "").join(" ");
      if (clean(t)) return t;
    }
    if (el.getAttribute("aria-label")) return el.getAttribute("aria-label");
    // Teks terdekat di container yang hanya berisi kolom ini (mis. <div>Asal universitas</div><input>).
    let node = el.parentElement;
    for (let i = 0; i < 3 && node; i++, node = node.parentElement) {
      if (node.querySelectorAll("input, textarea, select").length !== 1) break;
      const raw = textWithoutFields(node); // teks mentah: tanda * harus tetap ada untuk isRequired()
      const t = clean(raw);
      if (t && t.length <= 200) return raw;
    }
    return el.placeholder || el.name || "";
  }

  const visible = (el) => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);

  // Wajib = ada tanda * di label, atau atribut required / aria-required.
  const isRequired = (el, rawLabel) => el.required || el.getAttribute("aria-required") === "true" || /\*/.test(rawLabel);

  // Kolom yang bisa diisi di halaman ini (terlihat, aktif, bukan UI extension).
  function candidates() {
    const out = [];
    document.querySelectorAll("input, textarea, select").forEach((el) => {
      const type = (el.getAttribute("type") || "").toLowerCase();
      if (SKIP_TYPES.has(type) || el.disabled || el.readOnly || !visible(el) || el.closest(OWN_UI)) return;
      out.push(el);
    });
    return out;
  }

  // Kolom kosong yang belum pernah diproses. pageMarksRequired: apakah halaman ini memakai penanda wajib sama sekali.
  let counter = 0;
  function collect() {
    const fields = [];
    let pageMarksRequired = false;
    for (const el of candidates()) {
      const raw = labelFor(el);
      const required = isRequired(el, raw);
      pageMarksRequired = pageMarksRequired || required;
      if ((el.value || "").trim() || el.dataset.ragDone) continue; // jangan menimpa isian atau pilihan pengguna
      const label = clean(raw);
      if (!label) continue;
      el.dataset.ragKey = el.dataset.ragKey || `f${counter++}`;
      fields.push({ key: el.dataset.ragKey, label, tag: el.tagName.toLowerCase(), required });
    }
    return { fields, pageMarksRequired };
  }

  // Set value lewat setter bawaan lalu kirim event input/change, supaya framework (React, Vue) ikut mencatat.
  function setValue(el, value) {
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype
      : el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(el, value);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  // Dropdown: pilih opsi yang teksnya sama dengan jawaban (atau saling memuat). Kalau tidak ada, biarkan kosong.
  function matchOption(select, answer) {
    const a = answer.toLowerCase().trim();
    const opts = [...select.options].filter((o) => o.value !== "");
    return opts.find((o) => o.text.toLowerCase().trim() === a)
      || opts.find((o) => a.includes(o.text.toLowerCase().trim()) || o.text.toLowerCase().includes(a));
  }

  const fieldEl = (key) => document.querySelector(`[data-rag-key="${key}"]`);

  function mark(el, status, tooltip, dashed = false) {
    el.style.outline = `2px ${dashed ? "dashed" : "solid"} ${COLORS[status] || COLORS.not_found}`;
    el.style.outlineOffset = "2px";
    el.title = tooltip;
    el.dataset.ragDone = "1";
  }

  function describe(r) {
    const parts = [`RAG Lamaran: ${STATUS_LABEL[r.status] || r.status}`];
    const src = [...new Set((r.sources || []).map((s) => (s.page ? `${s.file} hlm ${s.page}` : s.file)))].join(", ");
    if (src) parts.push(`Sumber: ${src}`);
    if (r.flags && r.flags.length) parts.push(`Flags: ${r.flags.join(", ")}`);
    return parts.join("\n");
  }

  function fill(fields, results) {
    const byQuestion = new Map(results.map((r) => [clean(r.question), r]));
    const counts = { ok: 0, needs_review: 0, not_found: 0 };
    for (const f of fields) {
      const el = fieldEl(f.key);
      const r = byQuestion.get(f.label);
      if (!el || !r) continue;
      let status = r.value ? r.status : "not_found";
      if (r.value && el instanceof HTMLSelectElement) {
        const opt = matchOption(el, r.value);
        if (opt) setValue(el, opt.value); else status = "not_found";
      } else if (r.value) {
        setValue(el, r.value);
      }
      counts[status] = (counts[status] || 0) + 1;
      mark(el, status, describe({ ...r, status }));
    }
    return counts;
  }

  function notify(action, payload = {}) {
    if (globalThis.chrome?.runtime?.sendMessage) {
      chrome.runtime.sendMessage({ action, ...payload });
    } else {
      // Dipakai saat menguji content.js langsung di halaman, tanpa extension terpasang.
      window.dispatchEvent(new CustomEvent(`rag-lamaran-${action}`, { detail: payload }));
    }
  }

  function box(id, css) {
    let el = document.getElementById(id);
    if (!el) {
      el = document.createElement("div");
      el.id = id;
      el.style.cssText = "position:fixed;right:12px;z-index:2147483647;max-width:360px;padding:10px 12px;"
        + "border-radius:10px;font:13px/1.5 system-ui,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,.2);" + css;
      document.body.append(el);
    }
    return el;
  }

  function banner(text, state) {
    const el = box("rag-lamaran-banner", "top:12px;");
    const palette = { busy: ["#eef2ff", "#3730a3"], done: ["#ecfdf5", "#065f46"], warn: ["#fffbeb", "#92400e"],
                      error: ["#fef2f2", "#991b1b"] }[state] || ["#f3f4f6", "#111827"];
    el.style.background = palette[0];
    el.style.color = palette[1];
    el.replaceChildren();
    const close = document.createElement("button");
    close.textContent = "×";
    close.setAttribute("aria-label", "Tutup");
    close.style.cssText = "float:right;border:0;background:transparent;font-size:16px;cursor:pointer;color:inherit;";
    close.onclick = () => el.remove();
    const body = document.createElement("div");
    body.id = "rag-lamaran-banner-text";
    body.textContent = text;
    el.append(close, body);
  }

  // Panel pilihan untuk kolom tidak wajib. Default "Kosongkan" supaya tidak ada isian yang dikarang tanpa persetujuan.
  const CHOICES = [["dokumen", "Isi dari dokumen"], ["sendiri", "Isi sendiri"], ["kosong", "Kosongkan"]];
  function askOptional(fields, pageMarksRequired) {
    const el = box("rag-lamaran-optional", "bottom:12px;max-height:60vh;overflow:auto;background:#ffffff;color:#111827;");
    el.replaceChildren();
    const title = document.createElement("div");
    title.style.cssText = "font-weight:600;margin-bottom:4px;";
    title.textContent = pageMarksRequired ? `${fields.length} kolom tidak wajib (tanpa tanda *)`
      : "Halaman ini tidak menandai kolom wajib";
    const hint = document.createElement("div");
    hint.style.cssText = "color:#6b7280;font-size:12px;margin-bottom:8px;";
    hint.textContent = "Pilih untuk tiap kolom. Default: kosongkan, supaya sistem tidak mengarang isian yang tidak diminta.";
    el.append(title, hint);

    fields.forEach((f, i) => {
      const row = document.createElement("div");
      row.style.cssText = "border-top:1px solid #e5e7eb;padding:6px 0;";
      const label = document.createElement("div");
      label.textContent = f.label;
      label.style.cssText = "cursor:pointer;text-decoration:underline dotted;";
      label.title = "Klik untuk melihat kolomnya";
      label.onclick = () => fieldEl(f.key)?.scrollIntoView({ behavior: "smooth", block: "center" });
      const opts = document.createElement("div");
      opts.style.cssText = "display:flex;gap:10px;flex-wrap:wrap;font-size:12px;margin-top:2px;";
      for (const [value, text] of CHOICES) {
        const lab = document.createElement("label");
        lab.style.cssText = "display:inline-flex;gap:4px;align-items:center;cursor:pointer;";
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = `rag-opt-${i}`;
        radio.value = value;
        radio.checked = value === "kosong";
        lab.append(radio, document.createTextNode(text));
        opts.append(lab);
      }
      row.append(label, opts);
      el.append(row);
    });

    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;gap:8px;margin-top:8px;";
    const btn = (text, primary) => {
      const b = document.createElement("button");
      b.textContent = text;
      b.style.cssText = `flex:1;padding:6px;border-radius:6px;cursor:pointer;font:inherit;border:1px solid #0f766e;`
        + (primary ? "background:#0f766e;color:#fff;" : "background:#fff;color:#0f766e;");
      return b;
    };
    const all = btn("Semua dari dokumen", false);
    all.onclick = () => el.querySelectorAll('input[value="dokumen"]').forEach((r) => { r.checked = true; });
    const apply = btn("Terapkan pilihan", true);
    apply.onclick = () => {
      const toDraft = [];
      let firstSelf = null;
      fields.forEach((f, i) => {
        const choice = el.querySelector(`input[name="rag-opt-${i}"]:checked`)?.value || "kosong";
        const target = fieldEl(f.key);
        if (!target) return;
        if (choice === "dokumen") toDraft.push(f);
        else if (choice === "sendiri") { mark(target, "self", "RAG Lamaran: kamu memilih mengisi sendiri"); firstSelf ||= target; }
        else mark(target, "skipped", "RAG Lamaran: dikosongkan sesuai pilihanmu", true);
      });
      el.remove();
      if (firstSelf) firstSelf.focus();
      if (toDraft.length) notify("draftSelected", { fields: toDraft });
    };
    actions.append(all, apply);
    el.append(actions);
  }

  // Setelah pengguna menekan Next di form yang tidak reload, kolom baru muncul. Pantau DOM dan beri tahu background.
  let observer = null;
  let timer = null;
  function watch(on) {
    if (!on) { observer?.disconnect(); observer = null; return; }
    if (observer) return;
    // Perubahan dari UI extension sendiri (banner, panel) diabaikan. Hati-hati: perubahan atribut (mis. `hidden`
    // saat pindah langkah) tidak punya addedNodes, dan [].every() selalu true, jadi harus dicek terpisah.
    const isOwn = (m) => m.target.closest?.(OWN_UI)
      || (m.type === "childList" && m.addedNodes.length > 0 && [...m.addedNodes].every((n) => n.closest?.(OWN_UI)));
    observer = new MutationObserver((mutations) => {
      if (mutations.every(isOwn)) return;
      clearTimeout(timer);
      timer = setTimeout(() => {
        const fresh = candidates().filter((el) => !el.dataset.ragKey && !(el.value || "").trim());
        if (fresh.length) notify("newFields");
      }, 1200);
    });
    observer.observe(document.body, { childList: true, subtree: true, attributes: true,
                                      attributeFilter: ["hidden", "style", "class", "aria-hidden"] });
  }

  window.__ragLamaran = { collect, fill, banner, askOptional, watch, clean, labelFor };
})();
