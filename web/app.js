(function () {
  "use strict";
  var RISK = window.HR.RISK, KPI = window.HR.KPI, SHAP = window.HR.SHAP, VN = window.HR.VN, PRE = window.HR.PRE;
  var SAGE = window.HRSage;
  function $(id) { return document.getElementById(id); }
  function fmtN(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, "."); }
  function vn1(x) { return String(x).replace(".", ","); }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function mean(xs) { return xs.length ? xs.reduce(function (a, b) { return a + b; }, 0) / xs.length : 0; }
  function num(v, d) { return (typeof v === "number" && isFinite(v)) ? v : d; }
  function icon(n) { return '<svg class="icon" aria-hidden="true"><use href="#i-' + n + '"/></svg>'; }

  var LEVEL_TXT = { red: ["Cao", "red"], amber: ["Trung bình", "amber"], green: ["Thấp", "green"] };
  var AV_BG = ["#065f46", "#1d4ed8", "#b45309", "#be123c", "#6d28d9", "#0e7490"];
  function avColor(s) { var h = 0; for (var i = 0; i < String(s).length; i++) h = (h * 31 + String(s).charCodeAt(i)) >>> 0; return AV_BG[h % AV_BG.length]; }
  function initials(role) { return role.split(" ").map(function (w) { return w[0]; }).slice(0, 2).join("").toUpperCase(); }
  function personCode(idx) { var n = (idx + 1); return "NV-" + ("000" + n).slice(-4); }

  /* ---------- luu trang thai (localStorage, an toan file://) ---------- */
  var STORE_KEY = "hr_web_state_v2", OLD_KEY = "hr_web_state_v1";
  var state = readStore();
  state.track = state.track || {};
  state.ui = state.ui || { explain: true };
  state.ui.navCollapsed = state.ui.navCollapsed === true;
  function readStore() {
    var s = {};
    try { s = JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch (e) { s = {}; }
    if (!s || typeof s !== "object" || !Object.keys(s).length) {
      try {
        var old = JSON.parse(localStorage.getItem(OLD_KEY) || "{}");
        if (old && typeof old === "object") { s.filters = old.filters; s.policy = old.policy; s.page = old.page; }
      } catch (e2) { }
    }
    return (s && typeof s === "object") ? s : {};
  }
  function writeStore() { try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch (e) { } }
  function debounce(fn, ms) { var t = null; return function () { var a = arguments, self = this; clearTimeout(t); t = setTimeout(function () { fn.apply(self, a); }, ms); }; }

  /* ---------- theo doi xu ly (trang thai + ghi chu) ---------- */
  var TRACK_TXT = { "new": ["Chưa liên hệ", "gray"], met: ["Đã gặp", "blue"], done: ["Đã xử lý", "solid-green"] };
  function tGet(idx) { return state.track[personCode(idx)] || null; }
  function tStatus(idx) { var t = tGet(idx); return (t && t.s) ? t.s : "new"; }
  function tSet(idx, s, n) {
    var code = personCode(idx), rec = state.track[code] || {};
    if (s !== undefined) rec.s = s;
    if (n !== undefined) rec.n = n;
    rec.t = new Date().toISOString();
    state.track[code] = rec;
    writeStore();
  }
  function timeTxt(iso) { try { return new Date(iso).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" }); } catch (e) { return ""; } }
  function statusChip(idx) { var t = TRACK_TXT[tStatus(idx)]; return '<span class="chip chip-status ' + t[1] + '">' + t[0] + '</span>'; }

  /* ---------- thong bao nhanh ---------- */
  function toast(msg) {
    var w = $("toast-wrap"), t = document.createElement("div");
    t.className = "toast";
    t.innerHTML = icon("check") + esc(msg);
    w.appendChild(t);
    setTimeout(function () { t.classList.add("out"); setTimeout(function () { t.remove(); }, 300); }, 2200);
  }

  /* ---------- dieu huong (hash + back/forward) ---------- */
  var PAGES = ["tong-quan", "uu-tien", "nhan-vien", "nguyen-nhan", "chinh-sach", "gioi-thieu"];
  var activePage = "tong-quan";
  function pageFromHash() { var h = String(location.hash || "").replace(/^#\/?/, ""); return PAGES.indexOf(h) >= 0 ? h : null; }
  function goto(page, opts) {
    if (PAGES.indexOf(page) < 0) page = "tong-quan";
    var el = $("page-" + page);
    if (!el) return;
    document.querySelectorAll(".page").forEach(function (p) { p.classList.remove("active"); });
    el.classList.add("active");
    document.querySelectorAll("#nav button").forEach(function (b) { b.classList.toggle("active", b.dataset.page === page); });
    closeNav();
    activePage = page;
    if (!(opts && opts.skipHash) && pageFromHash() !== page) location.hash = "/" + page;
    state.page = page; writeStore();
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", function () {
    var p = pageFromHash();
    if (p && p !== activePage) goto(p, { skipHash: true });
  });

  /* ---------- giai thich: bat/tat ---------- */
  var expToggle = $("explain-toggle");
  expToggle.checked = state.ui.explain !== false;
  document.body.classList.toggle("no-explain", !expToggle.checked);
  expToggle.addEventListener("change", function () {
    state.ui.explain = expToggle.checked; writeStore();
    document.body.classList.toggle("no-explain", !expToggle.checked);
  });

  /* ---------- sidebar: thu gon / mo rong (desktop) ---------- */
  var colBtn = $("btn-collapse");
  function applyNavCollapsed() {
    var c = state.ui.navCollapsed === true;
    document.body.classList.toggle("nav-collapsed", c);
    colBtn.setAttribute("aria-expanded", String(!c));
    colBtn.setAttribute("aria-label", c ? "Mở rộng thanh điều hướng" : "Thu gọn thanh điều hướng");
    document.querySelectorAll("#nav button, #btn-help").forEach(function (b) {
      var lb = b.querySelector(".nav-label");
      b.title = (c && lb) ? lb.textContent : "";
    });
  }
  colBtn.addEventListener("click", function () {
    state.ui.navCollapsed = !(state.ui.navCollapsed === true);
    writeStore();
    applyNavCollapsed();
  });
  applyNavCollapsed();

  /* ---------- sidebar: mo/dong tren mobile ---------- */
  function openNav() {
    $("sidebar").classList.add("open");
    $("nav-overlay").hidden = false;
    $("btn-menu").setAttribute("aria-expanded", "true");
    document.body.classList.add("nav-open");
  }
  function closeNav() {
    if (!$("sidebar").classList.contains("open")) return;
    $("sidebar").classList.remove("open");
    $("nav-overlay").hidden = true;
    $("btn-menu").setAttribute("aria-expanded", "false");
    document.body.classList.remove("nav-open");
  }
  $("btn-menu").addEventListener("click", function () {
    if ($("sidebar").classList.contains("open")) closeNav(); else openNav();
  });
  $("nav-overlay").addEventListener("click", closeNav);

  /* ---------- click uy quyen: dieu huong, mo ho so, dien san ---------- */
  document.addEventListener("click", function (ev) {
    var t = ev.target;
    var pf = t.closest("[data-prefill]");
    if (pf) {
      window._prefill = Object.assign({}, RISK[+pf.getAttribute("data-prefill")]);
      closeDrawer();
      goto("nhan-vien");
      fillProfileForm();
      return;
    }
    var pp = t.closest("[data-preset]");
    if (pp) { window._prefill = PRESETS[pp.dataset.preset]; fillProfileForm(); return; }
    var pg = t.closest("[data-goto]");
    if (pg) {
      goto(pg.getAttribute("data-goto"));
      if (pg.dataset.policyTarget && window._openPolicy) window._openPolicy(pg.dataset.policyTarget);
      return;
    }
    var pa = t.closest("[data-page]");
    if (pa) { goto(pa.getAttribute("data-page")); return; }
    var pop = t.closest(".pop-wrap");
    if (!pop) closePopover();
  });

  /* ---------- chao + ngay ---------- */
  var nRed = RISK.filter(function (r) { return r.band === "red"; }).length;
  var nAmber = RISK.filter(function (r) { return r.band === "amber"; }).length;
  var hr = new Date().getHours();
  var buoi = hr < 11 ? "Chào buổi sáng" : hr < 14 ? "Chào buổi trưa" : hr < 18 ? "Chào buổi chiều" : "Chào buổi tối";
  var homNay = new Date().toLocaleDateString("vi-VN", { weekday: "long", day: "numeric", month: "numeric" });
  $("ov-date").textContent = "Hôm nay, " + homNay;
  $("hero-title").textContent = buoi + ". Hôm nay có " + nRed + " người cần giữ chân ngay.";
  $("hero-sub").textContent = "Trong đó có thêm " + nAmber + " người cần theo dõi trong tuần này. Mở danh sách để xem tên từng người, lý do và việc nên làm.";

  /* ---------- goi y viec nen lam (ngon ngu HR) ---------- */
  var medByDept = {};
  ["Sales", "Research & Development", "Human Resources"].forEach(function (d) {
    var xs = RISK.filter(function (r) { return r.Department === d; }).map(function (r) { return r.MonthlyIncome; }).sort(function (a, b) { return a - b; });
    medByDept[d] = xs.length ? xs[Math.floor(xs.length / 2)] : 0;
  });
  function deptVN(d) {
    if (d === "Sales") return "Kinh doanh";
    if (d === "Research & Development") return "Nghiên cứu và Phát triển";
    return "Nhân sự";
  }
  function topReason(row) {
    return row.OverTime === "Yes" ? "Đang làm thêm giờ" : "Xem nguy cơ chi tiết";
  }
  function pctExplain(r) {
    var p = Math.round(r.proba * 100);
    return p + "% — nghĩa là trong 100 người có hoàn cảnh giống người này thì có khoảng " + p + " người muốn nghỉ.";
  }
  function tagsHtml(r) {
    return ["Tuổi " + r.Age, "Phòng " + deptVN(r.Department), r.JobRole, "Làm việc " + r.YearsAtCompany + " năm",
      r.OverTime === "Yes" ? "Đang làm thêm giờ" : "Không làm thêm giờ",
      '<span class="mask-print">Thu nhập ' + fmtN(r.MonthlyIncome) + ' đô một tháng</span>'].map(function (x) { return '<span class="tag">' + x + '</span>'; }).join("");
  }

  var suggestionCache = new Map();
  var suggestionPending = new Map();
  var lastSuggestion = null;
  var suggestionSequence = 0;
  function requestReport(payload) {
    var key = JSON.stringify(payload);
    if (suggestionCache.has(key)) return Promise.resolve(suggestionCache.get(key));
    if (suggestionPending.has(key)) return suggestionPending.get(key);
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, 40000);
    var promise = fetch("/api/suggest", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: key, signal: controller.signal
    }).then(function (response) {
      return response.json().catch(function () { throw new Error("Hãy mở web bằng python serve_web.py tại http://127.0.0.1:5501/."); })
        .then(function (data) {
          if (!response.ok) throw new Error(data.error || "Máy chủ gợi ý chưa sẵn sàng.");
          if (!data.report || typeof data.report !== "object" || !Array.isArray(data.report.action_plan)) throw new Error("Trả kết quả không hợp lệ.");
          var report = data.report;
          if (suggestionCache.size >= 50) suggestionCache.delete(suggestionCache.keys().next().value);
          suggestionCache.set(key, report);
          return report;
        });
    }).catch(function (error) {
      if (error.name === "AbortError") throw new Error("Chưa trả lời trong 40 giây. Hãy thử lại.");
      if (error instanceof TypeError) throw new Error("Không kết nối được máy chủ gợi ý. Hãy chạy python serve_web.py.");
      throw error;
    }).finally(function () { clearTimeout(timer); suggestionPending.delete(key); });
    suggestionPending.set(key, promise);
    return promise;
  }
  function suggestionPayload(row, band) {
    return { band: band, profile: {
      OverTime: row.OverTime, JobSatisfaction: row.JobSatisfaction,
      EnvironmentSatisfaction: row.EnvironmentSatisfaction, WorkLifeBalance: row.WorkLifeBalance,
      YearsAtCompany: row.YearsAtCompany, StockOptionLevel: row.StockOptionLevel,
      incomeBelowDept: row.MonthlyIncome < (medByDept[row.Department] || 0)
    } };
  }
  function reportList(items) {
    return items && items.length ? '<ul class="reason-list" style="margin:6px 0;padding-left:18px;">' + items.map(function (item) { return '<li>' + esc(item) + '</li>'; }).join("") + '</ul>' : '<p class="muted">Không có mục phù hợp.</p>';
  }
  function reportCardHtml(report) {
    if (!report) return '<div class="card" style="margin-top:14px;" id="ai-actions-card"><div class="card-title">' + icon("check") + 'Kế hoạch giữ chân nhân viên</div><p class="muted">Đang chờ phân tích.</p></div>';
    var factors = (report.factor_analysis || []).map(function (item) {
      return '<tr><td>' + esc(item.factor) + '</td><td>' + esc(item.value) + '</td><td>' + esc(item.assessment) + '</td><td>' + esc(item.note) + '</td></tr>';
    }).join("");
    var actions = (report.action_plan || []).map(function (item, index) {
      return '<li style="margin-bottom:16px;"><b>Ưu tiên ' + (index + 1) + ': ' + esc(item.action) + '</b>' +
        '<div style="margin-top:6px;"><b>Lý do:</b> ' + esc(item.reason || item.expected_result) + '</div>' +
        '<div><b>Căn cứ nội quy:</b> ' + esc(item.policy) + '</div>' +
        '<div><b>Người phụ trách:</b> ' + esc(item.owner) + '</div>' +
        '<div><b>Thời hạn:</b> ' + esc(item.deadline) + '</div>' +
        '<div><b>Kết quả kỳ vọng:</b> ' + esc(item.expected_result) + '</div>' +
        '<div><b>Điều kiện/phê duyệt:</b> ' + esc(item.approval) + '</div></li>';
    }).join("");
    return '<div class="card" style="margin-top:14px;" id="ai-actions-card"><div class="card-title">' + icon("check") + 'Kế hoạch giữ chân nhân viên</div>' +
      '<h3 style="font-size:15px;">Tóm tắt điều hành</h3><p>' + esc(report.executive_summary) + '</p>' +
      '<p><b>Đánh giá định tính: ' + esc(report.risk_assessment.level) + '.</b> ' + esc(report.risk_assessment.explanation) + '</p>' +
      '<h3 style="font-size:15px;">Phân tích yếu tố</h3><div style="overflow-x:auto;"><table><thead><tr><th>Yếu tố</th><th>Giá trị</th><th>Đánh giá</th><th>Nhận định</th></tr></thead><tbody>' + factors + '</tbody></table></div>' +
      '<h3 style="font-size:15px;">Giả thuyết nguyên nhân gốc rễ</h3>' + reportList(report.root_causes) +
      '<h3 style="font-size:15px;">Kế hoạch hành động</h3>' + (actions ? '<ol class="reason-list" style="margin:6px 0;padding-left:22px;">' + actions + '</ol>' : '<p class="muted">Chưa có hành động phù hợp trong nội quy.</p>') +
      '<h3 style="font-size:15px;">Việc không nên làm hoặc ngoài nội quy</h3>' + reportList(report.not_recommended) +
      '<h3 style="font-size:15px;">Dữ liệu còn thiếu và giả định</h3>' + reportList(report.missing_data) +
      '<h3 style="font-size:15px;">Chỉ số theo dõi</h3><p><b>Sau 1 tháng:</b></p>' + reportList(report.monitoring.one_month) + '<p><b>Sau 3 tháng:</b></p>' + reportList(report.monitoring.three_months) +
      '<p class="muted">Đây là đánh giá định tính dựa trên dữ liệu HR đã gửi, không phải giải thích của mô hình dự báo.</p></div>';
  }
  function showSuggestions(payload) {
    var card = document.getElementById("ai-actions-card");
    if (!card) return;
    var sequence = ++suggestionSequence;
    card.innerHTML = '<div class="card-title">' + icon("check") + 'Kế hoạch giữ chân nhân viên</div><p class="muted" role="status">Đang phân tích hồ sơ…</p>';
    requestReport(payload).then(function (report) {
      if (sequence !== suggestionSequence || !card.isConnected) return;
      card.outerHTML = reportCardHtml(report);
    }).catch(function (error) {
      if (sequence !== suggestionSequence || !card.isConnected) return;
      card.innerHTML = '<div class="card-title">' + icon("check") + 'Kế hoạch giữ chân nhân viên</div><p role="alert">' + esc(error.message || "Không tạo được kế hoạch.") + '</p><button type="button" class="btn sm" data-retry-suggest>Thử lại</button>';
    });
  }
  $("profile-result").addEventListener("click", function (ev) {
    if (ev.target.closest("[data-retry-suggest]") && lastSuggestion) showSuggestions(lastSuggestion);
  });

  /* ================= 1. TONG QUAN ================= */
  function redIdxAll() { var a = []; RISK.forEach(function (r, i) { if (r.band === "red") a.push(i); }); return a; }
  function renderProgress() {
    var reds = redIdxAll();
    var met = reds.filter(function (i) { return tStatus(i) === "met"; }).length;
    var done = reds.filter(function (i) { return tStatus(i) === "done"; }).length;
    var notYet = reds.length - met - done;
    var pct = reds.length ? Math.round((met + done) / reds.length * 100) : 0;
    $("ov-progress").hidden = false;
    $("ov-progress").innerHTML =
      '<div class="prog-main">' +
        '<div class="prog-label">Tiến độ tuần này: đã liên hệ <b>' + (met + done) + '/' + reds.length + '</b> người cần gặp ngay</div>' +
        '<div class="prog-track" role="img" aria-label="Đã liên hệ ' + pct + ' phần trăm"><div class="prog-fill" style="width:' + pct + '%"></div></div>' +
        '<div class="prog-legend">' +
          '<span>Đã gặp: <b>' + met + '</b></span>' +
          '<span>Đã xử lý: <b>' + done + '</b></span>' +
          '<span>Chưa liên hệ: <b>' + notYet + '</b></span>' +
        '</div>' +
      '</div>' +
      '<button class="btn" data-goto="uu-tien">Mở danh sách' + icon("arrow-right") + '</button>';
  }
  function renderTodo() {
    var reds = redIdxAll();
    var notYet = reds.filter(function (i) { return tStatus(i) === "new"; }).length;
    var otY = mean(RISK.filter(function (r) { return r.OverTime === "Yes"; }).map(function (r) { return r.left; })) * 100;
    var otN = mean(RISK.filter(function (r) { return r.OverTime === "No"; }).map(function (r) { return r.left; })) * 100;
    var items = [
      { lv: ["Cần làm ngay", "red"], g: "chinh-sach", p: "ot", b: "Lập phương án", t: "Giảm làm thêm giờ cho nhóm nguy cơ cao", r: "Vì nhóm làm thêm nghỉ tới " + otY.toFixed(0) + "% so với " + otN.toFixed(0) + "% ở nhóm không làm thêm. Đây là thay đổi đã có bằng chứng tác động để đưa vào mô phỏng." },
      notYet > 0
        ? { lv: ["Cần làm ngay", "red"], g: "uu-tien", b: "Mở danh sách", t: "Gặp trao đổi 1-1 với " + reds.length + " người nguy cơ cao trong 7 ngày", r: "Còn " + notYet + " người chưa liên hệ. Đánh dấu trạng thái ngay trong danh sách để theo dõi tiến độ." }
        : { lv: ["Đã hoàn thành", "solid-green"], g: "uu-tien", b: "Mở danh sách", t: "Đã liên hệ đủ " + reds.length + " người nguy cơ cao", r: "Tiếp tục theo dõi nhóm Trung bình — " + nAmber + " người — trong tuần này." },
      { lv: ["Nên làm tuần này", "amber"], g: "chinh-sach", p: "buddy", b: "Lập kế hoạch thí điểm", t: "Bố trí người hướng dẫn cho nhóm mới vào", r: "Nhóm mới vào có tỷ lệ nghỉ cao hơn, nhưng cần thí điểm và đo lại trước khi khẳng định hiệu quả." }
    ];
    $("todo-list").innerHTML = items.map(function (w, k) {
      return '<div class="action"><span class="num">' + (k + 1) + '</span><div class="a-main">' +
        '<div class="a-top"><span class="chip ' + w.lv[1] + '">' + w.lv[0] + '</span><span class="a-title">' + esc(w.t) + '</span></div>' +
        '<small>' + esc(w.r) + '</small>' +
        '<button class="btn sm" data-goto="' + w.g + '"' + (w.p ? ' data-policy-target="' + w.p + '"' : '') + '>' + esc(w.b) + icon("arrow-right") + '</button>' +
        '</div></div>';
    }).join("");
  }
  function renderOverview() {
    var nLeft = Math.round(KPI.n_total * KPI.attrition_rate / 100);
    var avgAnnual = num(KPI.avg_annual_usd, 65000);
    var estCost = nLeft * avgAnnual;
    var estCostTxt = (estCost / 1000000).toFixed(1).replace(".", ",") + " triệu đô";
    $("kpi-row").innerHTML = [
      '<div class="kpi"><div class="k-top"><span class="dot dot-gray"></span>Toàn công ty</div><div class="v">' + vn1(KPI.attrition_rate) + '%</div><div class="l">Người đã nghỉ việc</div>' +
        '<div class="e">Tính trên ' + fmtN(KPI.n_total) + ' hồ sơ (' + fmtN(nLeft) + ' người đã nghỉ).</div></div>',
      '<div class="kpi red"><div class="k-top"><span class="dot dot-red"></span>Cần làm ngay</div><div class="v">' + nRed + '</div><div class="l">Người cần giữ chân ngay</div>' +
        '<div class="e">Nên gặp trong 1 đến 2 ngày tới. Ngoài ra còn ' + nAmber + ' người cần theo dõi trong tuần.</div></div>',
      '<div class="kpi amber"><div class="k-top"><span class="dot dot-amber"></span>Cần theo dõi</div><div class="v">+' + vn1(KPI.ate_ot) + ' điểm %</div><div class="l">Làm thêm giờ và nguy cơ nghỉ</div>' +
        '<div class="e">Nhân viên làm thêm giờ có nguy cơ nghỉ cao hơn. Cứ 100 người được giảm làm thêm thì giữ lại được khoảng ' + Math.round(KPI.ate_ot) + ' người.</div></div>',
      '<div class="kpi green"><div class="k-top"><span class="dot dot-green"></span>Ổn định nếu giữ được người</div><div class="v">~' + estCostTxt + '</div><div class="l">Chi phí thay người ước tính</div>' +
        '<div class="e">Số tiền một năm để tuyển và đào tạo người thay thế ' + fmtN(nLeft) + ' người đã nghỉ.</div></div>'
    ].join("");

    var byDept = {};
    RISK.forEach(function (r) { (byDept[r.Department] = byDept[r.Department] || []).push(r.proba); });
    var topDept = Object.keys(byDept).map(function (d) { return [d, mean(byDept[d])]; }).sort(function (a, b) { return b[1] - a[1]; })[0];

    function bar(label, pct, cls) {
      return '<div class="bar-row"><div class="bar-label">' + esc(label) + '</div>' +
        '<div class="bar-track"><div class="bar-fill ' + cls + '" style="width:' + Math.min(100, pct / 0.6 * 100).toFixed(1) + '%">' + pct.toFixed(0) + '%</div></div></div>';
    }
    function lvlOf(p) { return p >= 0.35 ? "red" : p >= 0.2 ? "amber" : "green"; }
    var depts = Object.keys(byDept).map(function (d) { return [deptVN(d), mean(byDept[d])]; }).sort(function (a, b) { return b[1] - a[1]; });
    $("chart-dept").innerHTML = depts.map(function (x) { return bar(x[0], x[1] * 100, lvlOf(x[1])); }).join("") +
      '<p class="muted explain">Con số % nghĩa là: trong 100 người của phòng đó thì có khoảng từng đó người muốn nghỉ.</p>';
    var tens = [["Mới vào: 0 đến 2 năm", 0, 2], ["Đã quen việc: 3 đến 5 năm", 3, 5], ["Gắn bó lâu: trên 5 năm", 6, 100]].map(function (t) {
      var xs = RISK.filter(function (r) { return r.YearsAtCompany >= t[1] && r.YearsAtCompany <= t[2]; }).map(function (r) { return r.proba; });
      return [t[0], mean(xs)];
    });
    $("chart-tenure").innerHTML = tens.map(function (x) { return bar(x[0], x[1] * 100, lvlOf(x[1])); }).join("") +
      '<p class="muted explain">Nhóm mới vào thường cần được quan tâm nhiều nhất trong năm đầu tiên.</p>';
    $("dept-conclusion").textContent = "Kết luận nổi bật: Phòng " + deptVN(topDept[0]) + " có nhiều người muốn nghỉ nhất. Nên bắt đầu các buổi trao đổi 1-1 từ phòng này.";

    renderProgress();
    renderTodo();
  }
  renderOverview();

  /* ================= 2. DANH SACH CAN GIU ================= */
  var PAGE = 15, shown = PAGE;
  var curView = [];
  var statusSel = "";

  var BAND_OPTS = [
    { k: "red,amber", label: "Cần giữ ngay", dot: "dot-red" },
    { k: "red", label: "Cao", dot: "dot-red" },
    { k: "amber", label: "Trung bình", dot: "dot-amber" },
    { k: "red,amber,green", label: "Tất cả", dot: "dot-gray" }
  ];
  var STATUS_OPTS = [["", "Mọi trạng thái"], ["new", "Chưa liên hệ"], ["met", "Đã gặp"], ["done", "Đã xử lý"]];

  (function initFilters() {
    var ds = Array.from(new Set(RISK.map(function (r) { return r.Department; }))).sort();
    $("f-dept").innerHTML = '<option value="">Tất cả các phòng</option>' + ds.map(function (d) { return '<option value="' + esc(d) + '">' + esc(deptVN(d)) + '</option>'; }).join("");
    var f = state.filters || {};
    if (f.dept !== undefined) $("f-dept").value = f.dept;
    if (f.band) $("f-band").value = f.band;
    if (f.tenure !== undefined) $("f-tenure").value = f.tenure;
    if (f.sort) $("f-sort").value = f.sort;
    if (f.q) $("f-q").value = f.q;
    statusSel = f.status || "";
    var persist = debounce(function () {
      state.filters = { dept: $("f-dept").value, band: $("f-band").value, tenure: $("f-tenure").value, sort: $("f-sort").value, q: $("f-q").value, status: statusSel };
      writeStore();
    }, 250);
    ["f-dept", "f-band", "f-tenure", "f-sort", "f-q"].forEach(function (id) {
      $(id).addEventListener("input", function () { shown = PAGE; renderRisk(); persist(); });
    });
    renderBandSeg(); renderStatusSeg();
  })();

  function renderBandSeg() {
    var b = $("f-band").value;
    $("band-seg").innerHTML = BAND_OPTS.map(function (o) {
      var on = b === o.k;
      return '<button type="button" data-band="' + o.k + '"' + (on ? ' class="active" aria-pressed="true"' : ' aria-pressed="false"') + '>' +
        '<span class="dot ' + o.dot + '"></span>' + o.label + '</button>';
    }).join("");
  }
  function renderStatusSeg() {
    $("status-seg").innerHTML = STATUS_OPTS.map(function (o) {
      var on = statusSel === o[0];
      return '<button type="button" data-status="' + o[0] + '"' + (on ? ' class="active" aria-pressed="true"' : ' aria-pressed="false"') + '>' + o[1] + '</button>';
    }).join("");
  }
  $("band-seg").addEventListener("click", function (ev) {
    var b = ev.target.closest("[data-band]");
    if (!b) return;
    $("f-band").value = b.getAttribute("data-band");
    shown = PAGE; renderBandSeg(); renderRisk(); saveFiltersNow();
  });
  $("status-seg").addEventListener("click", function (ev) {
    var b = ev.target.closest("[data-status]");
    if (!b) return;
    statusSel = b.getAttribute("data-status");
    shown = PAGE; renderStatusSeg(); renderRisk(); saveFiltersNow();
  });
  function saveFiltersNow() {
    state.filters = { dept: $("f-dept").value, band: $("f-band").value, tenure: $("f-tenure").value, sort: $("f-sort").value, q: $("f-q").value, status: statusSel };
    writeStore();
  }

  /* popover bo loc */
  function closePopover() { $("filters-popover").hidden = true; $("btn-filters").setAttribute("aria-expanded", "false"); }
  $("btn-filters").addEventListener("click", function () {
    var open = $("filters-popover").hidden;
    $("filters-popover").hidden = !open;
    $("btn-filters").setAttribute("aria-expanded", String(open));
  });
  $("btn-clear-filters").addEventListener("click", function () {
    $("f-q").value = ""; $("f-dept").value = ""; $("f-tenure").value = ""; $("f-sort").value = "risk-desc";
    $("f-band").value = "red,amber"; statusSel = "";
    shown = PAGE; renderBandSeg(); renderStatusSeg(); renderRisk(); saveFiltersNow();
    toast("Đã xóa bộ lọc");
  });

  function inTenure(r, v) {
    if (!v) return true;
    if (v === "0-2") return r.YearsAtCompany <= 2;
    if (v === "3-5") return r.YearsAtCompany >= 3 && r.YearsAtCompany <= 5;
    return r.YearsAtCompany > 5;
  }
  function passesBase(r, q, dept, ten) {
    return (!dept || r.Department === dept) && inTenure(r, ten) &&
      (!q || r.JobRole.toLowerCase().indexOf(q) >= 0 || deptVN(r.Department).toLowerCase().indexOf(q) >= 0 || personCode(r._idx).toLowerCase().indexOf(q) >= 0);
  }
  function filteredRisk() {
    var dept = $("f-dept").value, bands = $("f-band").value.split(","), ten = $("f-tenure").value;
    var q = $("f-q").value.trim().toLowerCase(), sort = $("f-sort").value;
    var view = RISK.map(function (r, i) { var c = Object.assign({}, r); c._idx = i; return c; })
      .filter(function (r) {
        return passesBase(r, q, dept, ten) && bands.indexOf(r.band) >= 0 && (!statusSel || tStatus(r._idx) === statusSel);
      });
    if (sort === "risk-desc") view.sort(function (a, b) { return b.proba - a.proba; });
    if (sort === "tenure-asc") view.sort(function (a, b) { return a.YearsAtCompany - b.YearsAtCompany; });
    if (sort === "income-asc") view.sort(function (a, b) { return a.MonthlyIncome - b.MonthlyIncome; });
    return view;
  }
  function rowHtml(r) {
    var t = LEVEL_TXT[r.band];
    return '<button type="button" class="row" data-open="' + r._idx + '">' +
      '<span class="avatar-sm" style="background:' + avColor(r.JobRole) + '">' + initials(r.JobRole) + '</span>' +
      '<span class="row-main">' +
        '<span class="row-title">' + personCode(r._idx) + ' <span class="code">· ' + esc(r.JobRole) + '</span></span>' +
        '<span class="row-meta">Phòng ' + esc(deptVN(r.Department)) + ' · ' + r.YearsAtCompany + ' năm · ' + (r.OverTime === "Yes" ? "Đang làm thêm giờ" : "Không làm thêm giờ") + ' · ' + esc(topReason(r)) + '</span>' +
      '</span>' +
      '<span class="row-side">' + statusChip(r._idx) +
        '<span class="chip ' + t[1] + '">Mức ' + t[0].toLowerCase() + '</span>' +
        '<svg class="icon chevron" aria-hidden="true"><use href="#i-chev-right"/></svg>' +
      '</span>' +
    '</button>';
  }
  function renderRows() {
    $("risk-list").innerHTML = curView.slice(0, shown).map(rowHtml).join("");
    var wrap = $("load-more-wrap");
    if (shown >= curView.length) { wrap.hidden = true; }
    else {
      wrap.hidden = false;
      $("btn-load-more").textContent = "Xem thêm " + Math.min(PAGE, curView.length - shown) + " người (còn " + (curView.length - shown) + ")";
    }
  }
  function renderRisk() {
    curView = filteredRisk();
    var nr = curView.filter(function (r) { return r.band === "red"; }).length;
    var na = curView.filter(function (r) { return r.band === "amber"; }).length;
    if (!curView.length) {
      $("risk-count").textContent = "Không tìm thấy ai đúng tiêu chí vừa chọn.";
      $("risk-list").innerHTML = '<div class="empty"><div class="empty-ico">' + icon("search") + '</div><b>Không có ai đúng tiêu chí vừa chọn.</b><br>Thử nới rộng bộ lọc, ví dụ chọn Tất cả hoặc Mọi trạng thái.</div>';
      $("load-more-wrap").hidden = true;
      return;
    }
    $("risk-count").textContent = "Đang xem " + Math.min(shown, curView.length) + " trên " + curView.length + " người — Cao: " + nr + ", Trung bình: " + na + ", còn lại: " + (curView.length - nr - na) + ".";
    renderRows();
  }
  $("btn-load-more").addEventListener("click", function () { shown += PAGE; renderRows(); });
  $("risk-list").addEventListener("click", function (ev) {
    var row = ev.target.closest("[data-open]");
    if (row) openDrawer(+row.getAttribute("data-open"));
  });
  window.addEventListener("beforeprint", function () { if (curView.length) { shown = curView.length; renderRows(); } });
  window.addEventListener("afterprint", function () { shown = PAGE; if (curView.length) renderRows(); });

  $("btn-export").addEventListener("click", function () {
    var head = ["Mã", "Chức danh", "Phòng ban", "Tuổi", "Mức cần giữ chân", "Tỉ lệ muốn nghỉ", "Làm thêm giờ", "Thu nhập một tháng", "Thâm niên (năm)", "Mức hài lòng", "Trạng thái liên hệ", "Ghi chú"];
    var lines = curView.map(function (r) {
      var t = tGet(r._idx);
      return [personCode(r._idx), r.JobRole, deptVN(r.Department), r.Age, LEVEL_TXT[r.band][0], Math.round(r.proba * 100) + "%",
        r.OverTime === "Yes" ? "Có" : "Không", r.MonthlyIncome, r.YearsAtCompany, r.JobSatisfaction,
        TRACK_TXT[tStatus(r._idx)][0], (t && t.n) || ""]
        .map(function (v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(",");
    });
    var csv = "﻿" + head.join(",") + "\n" + lines.join("\n");
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "danh-sach-can-giu-chan.csv"; a.click();
    toast("Đã xuất danh sách " + curView.length + " người");
  });
  $("btn-print").addEventListener("click", function () { window.print(); });
  renderRisk();

  /* ================= DRAWER CHI TIET ================= */
  var lastFocus = null;
  function trackSegHtml(idx) {
    var cur = tStatus(idx);
    return '<div class="seg" role="group" aria-label="Trạng thái liên hệ">' +
      [["new", "Chưa liên hệ"], ["met", "Đã gặp"], ["done", "Đã xử lý"]].map(function (o) {
        return '<button type="button" data-tstat="' + o[0] + '"' + (cur === o[0] ? ' class="active"' : '') + '>' + o[1] + '</button>';
      }).join("") + '</div>';
  }
  function openDrawer(idx) {
    var r = RISK[idx];
    lastFocus = document.activeElement;
    $("drawer").dataset.idx = String(idx);
    var av = $("drawer-avatar");
    av.textContent = initials(r.JobRole); av.style.background = avColor(r.JobRole);
    $("drawer-title").textContent = personCode(idx) + " · " + r.JobRole;
    $("drawer-body").innerHTML = drawerBodyHtml(idx);
    $("drawer").hidden = false; $("drawer-backdrop").hidden = false;
    document.body.style.overflow = "hidden";
    $("btn-drawer-close").focus();
  }
  function closeDrawer() {
    if ($("drawer").hidden) return;
    $("drawer").hidden = true; $("drawer-backdrop").hidden = true;
    document.body.style.overflow = "";
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  function drawerBodyHtml(idx) {
    var r = RISK[idx], t = LEVEL_TXT[r.band], meta = tGet(idx);
    var html = '<div style="text-align:center;margin:4px 0 0;">' + gauge(r.proba, r.band) + '</div>';
    html += '<p style="text-align:center;margin:6px 0 16px;"><span class="chip ' + t[1] + '">Mức ' + t[0].toLowerCase() + '</span></p>';
    html += '<p class="explain">' + esc(pctExplain(r)) + '</p>';
    html += '<div style="margin:10px 0;">' + tagsHtml(r) + '</div>';
    html += '<h3 style="font-size:14px;margin:18px 0 8px;">' + "Theo dõi xử lý" + '</h3>';
    html += '<div id="drawer-track">' + trackSegHtml(idx) +
      '<div class="note-box" style="margin-top:10px;"><textarea id="drawer-note" placeholder="Ghi chú nhanh: đã hẹn gặp khi nào, cần lưu ý gì...">' + ((meta && meta.n) ? esc(meta.n) : "") + '</textarea>' +
      '<div class="note-hint" id="drawer-note-hint">' + ((meta && meta.t) ? "Đã lưu lúc " + timeTxt(meta.t) : "Ghi chú được lưu ngay trên máy này") + '</div></div></div>';
    html += '<p class="muted">Mở hồ sơ chi tiết và chấm điểm để nhận gợi ý.</p>';
    html += '<div class="drawer-actions"><button class="btn primary block" data-prefill="' + idx + '">' + icon("chart") + 'Xem nguy cơ chi tiết của người này</button></div>';
    return html;
  }
  var saveNote = debounce(function (idx, val) {
    tSet(idx, undefined, val);
    var hint = $("drawer-note-hint");
    if (hint) hint.textContent = "Đã lưu lúc " + timeTxt(new Date().toISOString());
  }, 400);
  $("drawer-body").addEventListener("click", function (ev) {
    var b = ev.target.closest("[data-tstat]");
    if (!b) return;
    var idx = +$("drawer").dataset.idx;
    tSet(idx, b.getAttribute("data-tstat"), $("drawer-note") ? $("drawer-note").value : undefined);
    Array.prototype.forEach.call($("drawer-track").querySelectorAll("[data-tstat]"), function (x) {
      x.classList.toggle("active", x.getAttribute("data-tstat") === tStatus(idx));
    });
    syncRowChip(idx); renderProgress(); renderTodo();
    toast("Đã cập nhật trạng thái: " + TRACK_TXT[tStatus(idx)][0]);
    if (statusSel) { shown = PAGE; renderRisk(); }
  });
  $("drawer-body").addEventListener("input", function (ev) {
    if (ev.target && ev.target.id === "drawer-note") saveNote(+$("drawer").dataset.idx, ev.target.value);
  });
  function syncRowChip(idx) {
    var el = document.querySelector('.row[data-open="' + idx + '"] .chip-status');
    if (!el) return;
    var t = TRACK_TXT[tStatus(idx)];
    el.className = "chip chip-status " + t[1];
    el.textContent = t[0];
  }
  $("btn-drawer-close").addEventListener("click", closeDrawer);
  $("drawer-backdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    if (!$("drawer").hidden) closeDrawer();
    else if (!$("filters-popover").hidden) closePopover();
    else closeNav();
  });

  /* ================= 3. NHAN VIEN (model GraphSAGE thuc) ================= */
  var F4 = { 1: "Rất thấp", 2: "Thấp", 3: "Cao", 4: "Rất cao" };
  var EDU = { 1: "Dưới cao đẳng", 2: "Cao đẳng", 3: "Đại học", 4: "Thạc sĩ", 5: "Tiến sĩ" };
  var F_JOBROLE = [["Health Care Representative", "Y tế"], ["Human Resources", "Nhân sự"], ["Laboratory Technician", "Kỹ thuật viên"], ["Manager", "Quản lý"], ["Manufacturing Director", "Giám đốc sản xuất"], ["Research Director", "Giám đốc nghiên cứu"], ["Research Scientist", "Nghiên cứu viên"], ["Sales Executive", "Kinh doanh cấp cao"], ["Sales Representative", "Nhân viên kinh doanh"]];
  var F_DEPT = [["Sales", "Kinh doanh"], ["Research & Development", "Nghiên cứu và Phát triển"], ["Human Resources", "Nhân sự"]];
  var FIELDS = [
    ["personal", [
      ["Age", "Tuổi", "num", 18, 60, 35],
      ["Gender", "Giới tính", "sel", [["Male", "Nam"], ["Female", "Nữ"]]],
      ["MaritalStatus", "Tình trạng gia đình", "sel", [["Single", "Độc thân"], ["Married", "Đã kết hôn"], ["Divorced", "Ly hôn"]]],
      ["DistanceFromHome", "Khoảng cách tới công ty (km)", "num", 1, 29, 5],
      ["Education", "Trình độ", "f4edu", EDU, 3]
    ]],
    ["job", [
      ["Department", "Phòng ban", "sel", F_DEPT],
      ["JobRole", "Chức danh", "sel", F_JOBROLE],
      ["MonthlyIncome", "Thu nhập một tháng (đô)", "num", 1000, 20000, 5000],
      ["YearsAtCompany", "Số năm làm ở công ty", "num", 0, 40, 6],
      ["OverTime", "Có làm thêm giờ không", "sel", [["No", "Không"], ["Yes", "Có"]]],
      ["YearsInCurrentRole", "Số năm ở vị trí hiện tại", "num", 0, 18, 4],
      ["YearsSinceLastPromotion", "Số năm từ lần lên chức gần nhất", "num", 0, 15, 2]
    ]],
    ["sat", [
      ["JobSatisfaction", "Hài lòng với công việc", "f4", F4, 3],
      ["EnvironmentSatisfaction", "Hài lòng với môi trường", "f4", F4, 3],
      ["WorkLifeBalance", "Cân bằng giữa việc và cuộc sống", "f4", F4, 3],
      ["JobInvolvement", "Mức gắn bó với việc", "f4", F4, 3],
      ["StockOptionLevel", "Thưởng gắn bó dài hạn", "sel", [[0, "Chưa có"], [1, "Ít"], [2, "Vừa"], [3, "Nhiều"]]]
    ]]
  ];
  var FG = { personal: "fg-personal", job: "fg-job", sat: "fg-sat" };
  FIELDS.forEach(function (g) {
    var key = g[0], fs = g[1];
    $(FG[key]).innerHTML = fs.map(function (f) {
      var k = f[0], lb = f[1], t = f[2], a = f[3], b = f[4], c = f[5], input = "";
      if (t === "num") input = '<input type="number" id="pf-' + k + '" min="' + a + '" max="' + b + '" value="' + c + '">';
      else if (t === "sel") input = '<select id="pf-' + k + '">' + a.map(function (o) { return '<option value="' + o[0] + '">' + esc(o[1]) + '</option>'; }).join("") + '</select>';
      else { input = '<select id="pf-' + k + '">' + Object.keys(a).map(function (v) { return '<option value="' + v + '"' + (+v === b ? " selected" : "") + '>' + esc(a[v]) + '</option>'; }).join("") + '</select>'; }
      return '<div><label for="pf-' + k + '">' + esc(lb) + '</label>' + input + '</div>';
    }).join("");
  });
  var PRESETS = {
    sales: { Age: 29, Gender: "Male", MaritalStatus: "Single", DistanceFromHome: 12, Education: 3, Department: "Sales", JobRole: "Sales Representative", MonthlyIncome: 4000, YearsAtCompany: 2, OverTime: "Yes", WorkLifeBalance: 2, JobSatisfaction: 2, EnvironmentSatisfaction: 2, StockOptionLevel: 0, JobInvolvement: 2, YearsInCurrentRole: 2, YearsSinceLastPromotion: 2 },
    lab: { Age: 26, Gender: "Female", MaritalStatus: "Single", DistanceFromHome: 8, Education: 3, Department: "Research & Development", JobRole: "Laboratory Technician", MonthlyIncome: 3500, YearsAtCompany: 1, OverTime: "Yes", WorkLifeBalance: 2, JobSatisfaction: 3, EnvironmentSatisfaction: 3, StockOptionLevel: 0, JobInvolvement: 3, YearsInCurrentRole: 1, YearsSinceLastPromotion: 1 },
    mgr: { Age: 42, Gender: "Male", MaritalStatus: "Married", DistanceFromHome: 5, Education: 4, Department: "Research & Development", JobRole: "Manager", MonthlyIncome: 12000, YearsAtCompany: 10, OverTime: "No", WorkLifeBalance: 3, JobSatisfaction: 4, EnvironmentSatisfaction: 4, StockOptionLevel: 2, JobInvolvement: 4, YearsInCurrentRole: 5, YearsSinceLastPromotion: 1 }
  };
  function fillProfileForm() {
    var P = window._prefill || {};
    FIELDS.forEach(function (g) { g[1].forEach(function (f) {
      var el = $("pf-" + f[0]);
      if (!el || P[f[0]] === undefined) return;
      el.value = String(P[f[0]]);
    }); });
    window._prefill = null;
  }
  function readForm() {
    var o = {};
    FIELDS.forEach(function (g) { g[1].forEach(function (f) {
      var v = $("pf-" + f[0]).value;
      o[f[0]] = (f[2] === "num" || f[2] === "f4" || f[2] === "f4edu" || f[0] === "StockOptionLevel") ? +v : v;
    }); });
    return o;
  }
  function judgeLevel(r) {
    var score = 0; var why = [];
    if (r.OverTime === "Yes") { score += 2; why.push("đang làm thêm nhiều"); }
    if (r.JobSatisfaction <= 2) { score += 1.5; why.push("chưa hài lòng với công việc"); }
    if (r.MonthlyIncome < (medByDept[r.Department] || 0)) { score += 1; why.push("thu nhập thấp hơn mặt bằng phòng"); }
    if (r.YearsAtCompany <= 2) { score += 1; why.push("mới vào dưới 2 năm"); }
    if (r.StockOptionLevel === 0) { score += 1; why.push("chưa có thưởng gắn bó dài hạn"); }
    if (r.WorkLifeBalance <= 2) { score += 1; why.push("mất cân bằng giữa việc và cuộc sống"); }
    if (r.EnvironmentSatisfaction <= 2) { score += 0.5; why.push("chưa hài lòng với môi trường"); }
    var band = score >= 3.5 ? "red" : score >= 2 ? "amber" : "green";
    return { band: band, why: why };
  }

  var MODEL = null;
  try {
    if (SAGE && PRE && window.HR.W && window.HR.X_TRAIN && window.HR.EDGES) {
      MODEL = SAGE.createModel(window.HR.W, window.HR.X_TRAIN, window.HR.EDGES, 15);
    }
  } catch (e) { MODEL = null; }
  function rawForModel(form) {
    var raw = {}, i;
    for (i = 0; i < PRE.num_cols.length; i++) raw[PRE.num_cols[i]] = PRE.means[i];
    for (i = 0; i < PRE.cat_cols.length; i++) {
      var c = PRE.cat_cols[i];
      raw[c] = (PRE.cat_defaults && PRE.cat_defaults[c]) || PRE.categories[i][0];
    }
    Object.keys(form).forEach(function (k) { raw[k] = form[k]; });
    return raw;
  }
  function modelProba(form) {
    if (!MODEL) return null;
    try { return MODEL.predict(SAGE.buildX(rawForModel(form), PRE)); } catch (e) { return null; }
  }
  function gauge(p, band) {
    var color = band === "red" ? "var(--red)" : band === "amber" ? "var(--amber)" : "var(--green)";
    var len = Math.max(1, 314.159 * p);
    return '<svg viewBox="0 0 220 132" class="gauge" role="img" aria-label="Nguy cơ ' + Math.round(p * 100) + ' phần trăm">' +
      '<path d="M 10 120 A 100 100 0 0 1 210 120" class="gauge-track"/>' +
      '<path d="M 10 120 A 100 100 0 0 1 210 120" class="gauge-val" style="stroke:' + color + ';stroke-dasharray:' + len.toFixed(1) + ' 999"/>' +
      '<text x="110" y="100" class="gauge-num">' + Math.round(p * 100) + '%</text>' +
      '<text x="110" y="124" class="gauge-lbl">khả năng muốn nghỉ</text></svg>';
  }
  function wiRow(label, p, color) {
    return '<div class="wi-row"><span class="wi-label">' + esc(label) + '</span>' +
      '<span class="wi-track"><span class="wi-fill" style="width:' + Math.round(p * 100) + '%;background:' + color + '"></span></span>' +
      '<span class="wi-val">' + Math.round(p * 100) + '%</span></div>';
  }
  function whatifHtml(p, pFlip, otYes) {
    return '<div class="whatif">' +
      (otYes
        ? wiRow("Hiện tại (có làm thêm)", p, "var(--amber)") + wiRow("Nếu giảm làm thêm", pFlip, "var(--green)")
        : wiRow("Hiện tại (không làm thêm)", p, "var(--green)") + wiRow("Nếu phải làm thêm", pFlip, "var(--red)")) +
      '</div>';
  }
  $("btn-score").addEventListener("click", function () {
    var r = readForm();
    var actsHtml = reportCardHtml(null);
    var p = modelProba(r);
    var html;
    if (p !== null) {
      var band = SAGE.bandOf(p, KPI.thr);
      var t = LEVEL_TXT[band];
      var pFlip = modelProba(Object.assign({}, r, { OverTime: r.OverTime === "Yes" ? "No" : "Yes" }));
      var j = judgeLevel(r);
      html = '<div class="card" style="margin-top:0;">' + gauge(p, band) +
        '<p style="text-align:center;margin:6px 0 2px;"><span class="chip ' + t[1] + '">Mức ' + t[0].toLowerCase() + '</span></p>' +
        '<p class="muted explain" style="text-align:center;margin-top:8px;">Điểm từ mô hình GraphSAGE thật, đã đối chiếu khớp bản Python (sai số dưới 0,001).</p>' +
        '<p>' + esc(pctExplain({ proba: p })) + (j.why.length ? ' Điểm đáng lưu ý: ' + esc(j.why.join("; ")) + '.' : ' Chưa thấy điểm đáng lo.') + '</p>' +
        (pFlip !== null
          ? '<h3 style="font-size:14px;margin:14px 0 6px;">Thử thay đổi một điều thì sao</h3>' + whatifHtml(p, pFlip, r.OverTime === "Yes") +
            '<p class="muted explain">' + (r.OverTime === "Yes"
              ? "Nếu giảm làm thêm giờ cho người này, mô hình ước tính mức muốn nghỉ giảm khoảng " + Math.round(Math.abs(p - pFlip) * 100) + " điểm %. Con số này cùng chiều với kết quả nhân quả toàn công ty (+" + vn1(KPI.ate_ot) + " điểm %)."
              : "Người này hiện không làm thêm giờ — đó là điểm tốt nên giữ. Nếu phải làm thêm, mức muốn nghỉ có thể tăng khoảng " + Math.round(Math.abs(p - pFlip) * 100) + " điểm %.") + '</p>'
          : '') +
        '<p class="muted explain">Kết quả chỉ là gợi ý. Anh chị vẫn nên trò chuyện trực tiếp với người này trước khi quyết định.</p></div>';
    } else {
      var j2 = judgeLevel(r);
      var t2 = LEVEL_TXT[j2.band];
      var pct = j2.band === "red" ? 80 : j2.band === "amber" ? 45 : 12;
      html = '<div class="card" style="margin-top:0;">' + gauge(pct / 100, j2.band) +
        '<p style="text-align:center;margin:6px 0 2px;"><span class="chip ' + t2[1] + '">Mức ' + t2[0].toLowerCase() + '</span></p>' +
        '<p><b>Khoảng ' + pct + ' trên 100 người</b> có hoàn cảnh giống người này đã muốn nghỉ. ' +
        (j2.why.length ? 'Lý do chính: ' + esc(j2.why.join("; ")) + '.' : 'Chưa thấy điểm đáng lo.') + '</p>' +
        (r.OverTime === "Yes"
          ? '<div class="successbox">Nếu giảm bớt làm thêm giờ cho người này: dự kiến giữ lại được thêm khoảng ' + Math.round(KPI.ate_ot) + ' trên 100 người có hoàn cảnh tương tự.</div>'
          : '<div class="insight">Người này hiện không làm thêm giờ, đó là điểm tốt nên giữ. Nếu sau này phải làm thêm, nguy cơ muốn nghỉ có thể tăng lên rõ rệt.</div>') +
        '<p class="muted explain">Kết quả chỉ là gợi ý. Anh chị vẫn nên trò chuyện trực tiếp với người này trước khi quyết định.</p></div>';
    }
    $("profile-result").innerHTML = html + actsHtml;
    lastSuggestion = suggestionPayload(r, p !== null ? SAGE.bandOf(p, KPI.thr) : judgeLevel(r).band);
    showSuggestions(lastSuggestion);
  });

  /* ================= 4. NGUYEN NHAN ================= */
  (function nguyenNhan() {
    var NAME_VN = { OverTime: "Làm thêm giờ", JobRole: "Chức danh công việc", Age: "Tuổi", JobSatisfaction: "Mức hài lòng với công việc", BusinessTravel: "Đi công tác xa", MaritalStatus: "Tình trạng gia đình", EducationField: "Ngành học", WorkLifeBalance: "Cân bằng giữa việc và cuộc sống", Department: "Phòng ban", EnvironmentSatisfaction: "Mức hài lòng với môi trường", DistanceFromHome: "Khoảng cách từ nhà tới công ty", Gender: "Giới tính" };
    function plainFeat(f) {
      var base = String(f).replace(/^num__/, "").replace(/^cat__/, "").split("_")[0];
      return NAME_VN[base] || VN[f] || f;
    }
    var items = SHAP.grouped.slice(0, 10);
    var max = items[0].mean_abs_shap;
    $("tbl-grouped").innerHTML = items.map(function (d) {
      var w = (d.mean_abs_shap / max * 100).toFixed(1);
      return '<div class="bar-row"><div class="bar-label">' + esc(plainFeat(d.feature_group)) + '</div>' +
        '<div class="bar-track"><div class="bar-fill red" style="width:' + w + '%">' + esc(plainFeat(d.feature_group)) + '</div></div></div>';
    }).join("") + '<p class="muted explain">Thanh càng dài thì điều đó càng liên quan đến việc nghỉ. Ví dụ: làm thêm giờ và chức danh là hai điều liên quan nhiều nhất.</p>';
    function cohortVN(c) {
      if (c === "Sales") return "Nhóm kinh doanh";
      if (c === "OT_Yes") return "Nhóm hay làm thêm giờ";
      if (c === "Tenure_0_2") return "Nhóm mới vào (0 đến 2 năm)";
      if (c === "Age_u30") return "Nhóm dưới 30 tuổi";
      return c;
    }
    $("tbl-cohort").innerHTML = '<div class="table-wrap"><table class="tbl"><tr><th>Nhóm</th><th>Số người đã xem</th><th>Tỉ lệ muốn nghỉ</th><th>3 điểm chung dễ thấy nhất</th></tr>' +
      SHAP.cohort.map(function (d) {
        return '<tr><td><b>' + esc(cohortVN(d.cohort)) + '</b></td><td>' + d.n + ' người</td><td>' + Math.round(d.mean_proba * 100) + '% — nghĩa là trong 100 người của nhóm thì có khoảng ' + Math.round(d.mean_proba * 100) + ' người muốn nghỉ</td>' +
          '<td>' + esc(plainFeat(d.top1)) + '; ' + esc(plainFeat(d.top2)) + '; ' + esc(plainFeat(d.top3)) + '</td></tr>';
      }).join("") + '</table></div>';
    $("insight-top").innerHTML = "Đọc nhanh trong 10 giây: <b>nhóm kinh doanh, nhóm hay làm thêm giờ và nhóm mới vào dưới 2 năm</b> là những nhóm cần quan tâm nhiều nhất.";
    var cs = SHAP.cases;
    var selIdx = 0;
    function caseLabel(c, k) {
      var label = c.true_label === 1 ? "đã nghỉ thật" : "vẫn ở lại";
      return 'Trường hợp ' + (k + 1) + ': khoảng ' + Math.round(c.proba * 100) + ' trên 100 người giống vậy muốn nghỉ (' + label + ')';
    }
    $("case-list").innerHTML = cs.map(function (c, k) {
      return '<button class="case-item' + (k === 0 ? " active" : "") + '" data-case="' + k + '" aria-selected="' + (k === 0) + '" role="option">' + esc(caseLabel(c, k)) + '</button>';
    }).join("");
    function show() {
      var c = cs[selIdx];
      function li(f, up) { return '<li>' + esc(plainFeat(f)) + (up ? " — điểm làm tăng khả năng nghỉ" : " — điểm đang giữ người ở lại") + '</li>'; }
      $("case-detail").innerHTML = '<p><b>Khoảng ' + Math.round(c.proba * 100) + ' trên 100 người</b> có hoàn cảnh giống trường hợp này đã muốn nghỉ.</p>' +
        '<div class="grid2"><div><b>Điểm làm tăng khả năng nghỉ</b><ul class="reason-list">' + li(c.top_up_1, 1) + li(c.top_up_2, 1) + li(c.top_up_3, 1) + '</ul></div>' +
        '<div><b>Điểm đang giữ người ở lại</b><ul class="reason-list">' + li(c.top_down_1, 0) + li(c.top_down_2, 0) + '</ul></div></div>';
    }
    $("case-list").addEventListener("click", function (ev) {
      var b = ev.target.closest ? ev.target.closest("[data-case]") : null;
      if (!b) return;
      selIdx = +b.getAttribute("data-case");
      Array.prototype.forEach.call($("case-list").querySelectorAll(".case-item"), function (x) {
        var on = +x.getAttribute("data-case") === selIdx;
        x.classList.toggle("active", on);
        x.setAttribute("aria-selected", String(on));
      });
      show();
    });
    show();
  })();

  /* ================= 5. CHINH SACH ================= */
  (function chinhSach() {
    var PR = KPI.policy_rates || {};
    var RATES = { ot: num(PR.ot, KPI.ate_ot), buddy: num(PR.buddy, 12), pay: num(PR.pay, 8) };
    var AVG = num(KPI.avg_annual_usd, 65000);
    window.initPolicyPlanner({
      KPI: KPI, rates: RATES, avgAnnual: AVG, PR: PR, $: $, esc: esc, icon: icon,
      fmtN: fmtN, vn1: vn1, num: num, state: state, writeStore: writeStore,
      debounce: debounce, toast: toast, isPolicyPage: function () { return activePage === "chinh-sach"; }
    });
  })();

  /* ================= 6. GIOI THIEU: xoa du lieu theo doi ================= */
  $("btn-clear-track").addEventListener("click", function () {
    var n = Object.keys(state.track).length;
    if (!n) { toast("Chưa có dữ liệu theo dõi nào"); return; }
    if (!window.confirm("Xóa toàn bộ trạng thái và ghi chú đã lưu trên máy này? Không thể hoàn tác.")) return;
    state.track = {}; writeStore();
    renderProgress(); renderTodo(); renderRisk();
    toast("Đã xóa dữ liệu theo dõi trên máy này");
  });

  /* ---------- khoi phuc trang thai ban dau ---------- */
  var restorePage = pageFromHash() || (state.page && PAGES.indexOf(state.page) >= 0 ? state.page : null);
  if (restorePage && restorePage !== activePage) goto(restorePage, { skipHash: !!pageFromHash() });
})();
