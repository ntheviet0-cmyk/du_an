(function () {
  "use strict";

  window.initPolicyPlanner = function (ctx) {
    var KPI = ctx.KPI, RATES = ctx.rates, AVG = ctx.avgAnnual;
    var $ = ctx.$, esc = ctx.esc, icon = ctx.icon, fmtN = ctx.fmtN, vn1 = ctx.vn1;
    var state = ctx.state, writeStore = ctx.writeStore, debounce = ctx.debounce, toast = ctx.toast;
    var PR = KPI.policy_rates || {};
    var POLICIES = {
      ot: {
        title: "Giảm làm thêm giờ",
        formSub: "Ước tính số người thực sự có thể chuyển khỏi tình trạng làm thêm.",
        note: "Đây là phương án duy nhất hiện có bằng chứng tác động. Kết quả vẫn cần được kiểm chứng lại bằng dữ liệu thực tế của công ty.",
        desc: "Dữ liệu mẫu cho thấy giảm làm thêm giờ gắn với mức giảm nghỉ việc " + vn1(RATES.ot) + " điểm phần trăm sau khi đã hiệu chỉnh các khác biệt chính."
      },
      buddy: {
        title: "Thí điểm hỗ trợ nhân viên mới",
        formSub: "Xác định quy mô mà đội ngũ hiện tại có thể hỗ trợ tốt.",
        note: "Chênh lệch nghỉ việc của nhóm mới là tín hiệu để ưu tiên, không phải hiệu quả chắc chắn của chương trình buddy.",
        desc: "Nhóm mới vào có tỷ lệ nghỉ cao hơn " + vn1(RATES.buddy) + " điểm phần trăm trong dữ liệu mẫu " + fmtN(ctx.num(PR.n_buddy_group, 0)) + " người. Cần thí điểm trước khi mở rộng."
      },
      pay: {
        title: "Thí điểm rà soát thu nhập",
        formSub: "Khoanh vùng số hồ sơ có thể rà soát công bằng trong một đợt.",
        note: "Chênh lệch quan sát không chứng minh điều chỉnh thu nhập sẽ làm giảm nghỉ việc. Phương án này cần tiêu chí và phê duyệt rõ ràng.",
        desc: "Nhóm thấp hơn mặt bằng phòng có tỷ lệ nghỉ cao hơn " + vn1(RATES.pay) + " điểm phần trăm trong dữ liệu mẫu " + fmtN(ctx.num(PR.n_pay_group, 0)) + " người. Đây là tín hiệu rà soát, không phải cam kết tăng lương."
      }
    };
    var st0 = state.policy || {};
    var cur = RATES[st0.cur] !== undefined ? st0.cur : "ot";
    var step = st0.step >= 1 && st0.step <= 3 ? st0.step : 1;

    function pick(key) {
      document.querySelectorAll("#policy-pick .pick").forEach(function (button) {
        var selected = button.dataset.policy === key;
        button.classList.toggle("active", selected);
        button.setAttribute("aria-pressed", String(selected));
      });
    }

    var savePolicy = debounce(function () {
      var values = {};
      document.querySelectorAll("#page-chinh-sach input, #page-chinh-sach select").forEach(function (el) { values[el.id] = el.value; });
      state.policy = { cur: cur, step: step, values: values };
      writeStore();
    }, 250);

    function showStep(next) {
      step = next;
      [1, 2, 3].forEach(function (n) {
        $("policy-step-" + n).hidden = n !== step;
        var marker = document.querySelector('#policy-progress [data-policy-step="' + n + '"]');
        marker.classList.toggle("active", n === step);
        marker.classList.toggle("done", n < step);
      });
      savePolicy();
      if (ctx.isPolicyPage()) window.scrollTo(0, 0);
    }

    function selectPolicy(key) {
      if (RATES[key] === undefined) return;
      cur = key;
      pick(cur);
      $("policy-desc").innerHTML = icon("info") + "<span>" + esc(POLICIES[cur].desc) + "</span>";
      $("policy-form-title").textContent = POLICIES[cur].title;
      $("policy-form-sub").textContent = POLICIES[cur].formSub;
      $("policy-form-note").textContent = POLICIES[cur].note;
      document.querySelectorAll("[data-policy-form]").forEach(function (form) { form.hidden = form.dataset.policyForm !== cur; });
      savePolicy();
    }

    function moneyTxt(money) {
      if (money < 0) return "-" + moneyTxt(Math.abs(money));
      if (money >= 1000000) return (money / 1000000).toFixed(1).replace(".", ",") + " triệu đô";
      return fmtN(money) + " đô";
    }

    function val(id, lo, hi) {
      var el = $(id), value = Number(el.value);
      if (!isFinite(value)) value = lo;
      value = Math.max(lo, Math.min(hi, value));
      el.value = value;
      return value;
    }

    function stat(value, label, hint) {
      return '<div class="result-stat"><span class="value">' + esc(value) + '</span><span class="label">' + esc(label) + '</span><span class="hint">' + esc(hint) + '</span></div>';
    }
    function list(items) { return items.map(function (item) { return "<li>" + esc(item) + "</li>"; }).join(""); }
    function oneDecimal(value) { return vn1(value.toFixed(1)); }

    function renderResult() {
      var replace = val("p-replace", 0, 500);
      var budget = val("p-budget", 0, 100000000);
      var stats = [], assumptions = [], actions = [], caution = "", summary = "";
      var badge = $("policy-result-badge");
      $("policy-result-title").textContent = POLICIES[cur].title;

      if (cur === "ot") {
        var total = val("p-ot-total", 1, 100000);
        var otRate = val("p-ot-rate", 0, 100);
        var reduce = val("p-ot-reduce", 0, 100);
        var months = val("p-ot-months", 1, 24);
        var overtime = total * otRate / 100;
        var shifted = overtime * reduce / 100;
        var avoided = shifted * RATES.ot / 100;
        var gross = avoided * AVG * replace / 100;
        var net = gross - budget;
        badge.className = "chip green";
        badge.textContent = "Có bằng chứng tác động";
        stats = [
          stat(fmtN(total), " Nhân sự trong phạm vi", " Quy mô dùng để lập kế hoạch"),
          stat(fmtN(shifted), " Người dự kiến giảm OT", fmtN(overtime) + " người hiện đang làm thêm"),
          stat(oneDecimal(avoided), " Ca nghỉ có thể tránh", " uớc tính trung tâm, không phải cam kết"),
          stat(moneyTxt(net), " Giá trị ròng ước tính", " sau ngân sách triển khai " + moneyTxt(budget))
        ];
        assumptions = [
          otRate + "% nhân sự hiện đang làm thêm giờ.",
          "Có thể giảm làm thêm cho " + reduce + "% trong nhóm này trong " + months + " tháng.",
          "Tác động tham chiếu là " + vn1(RATES.ot) + " điểm phần trăm từ dữ liệu mẫu.",
          "Chi phí thay thế bằng " + replace + "% lương năm trung bình."
        ];
        actions = [
          "Xác nhận số giờ làm thêm thực tế và năng lực phân bổ lại công việc.",
          "Chọn nhóm triển khai, quản lý phụ trách và ngày bắt đầu.",
          "Đo lại giờ làm thêm, mức hài lòng và nghỉ việc sau 1 và 3 tháng."
        ];
        caution = "Ước tính dựa trên dữ liệu mẫu IBM, không phải dữ liệu HRIS hiện tại của công ty. Cần đối chiếu dữ liệu nội bộ trước khi dùng để phê duyệt ngân sách.";
        summary = "PHƯƠNG ÁN GIẢM LÀM THÊM GIỜ\nPhạm vi: " + fmtN(total) + " nhân sự; " + otRate + "% đang làm thêm; dự kiến giảm cho " + fmtN(shifted) + " người trong " + months + " tháng.\nTác động tham chiếu: khoảng " + oneDecimal(avoided) + " ca nghỉ có thể tránh. Giá trị ròng ước tính: " + moneyTxt(net) + ".\nĐiều kiện: xác nhận giờ làm thêm thực tế, nguồn lực thay thế và đo lại sau 1 và 3 tháng.\nLưu ý: số liệu tham chiếu từ dữ liệu mẫu, không phải cam kết kết quả.";
      } else if (cur === "buddy") {
        var eligibleBuddy = val("p-buddy-eligible", 1, 100000);
        var capacityBuddy = val("p-buddy-capacity", 1, 100000);
        var pilotBuddy = Math.min(eligibleBuddy, capacityBuddy);
        var monthsBuddy = val("p-buddy-months", 1, 12);
        var checkin = val("p-buddy-checkin", 1, 12);
        badge.className = "chip amber";
        badge.textContent = "Cần thí điểm và đo lại";
        stats = [
          stat(fmtN(eligibleBuddy), "Nhân viên đủ điều kiện", "Nhóm mới cần hỗ trợ"),
          stat(fmtN(pilotBuddy), "Quy mô thí điểm", "Theo năng lực buddy hiện có"),
          stat(monthsBuddy + " tháng", "Thời gian thí điểm", checkin + " lần check-in mỗi tháng"),
          stat(moneyTxt(budget), "Ngân sách dự kiến", "Chưa tính lợi ích giữ chân")
        ];
        assumptions = [
          "Có " + eligibleBuddy + " nhân viên mới đủ điều kiện.",
          "Đội ngũ có thể hỗ trợ tối đa " + capacityBuddy + " người trong đợt này.",
          "Mỗi cặp thực hiện " + checkin + " lần check-in mỗi tháng trong " + monthsBuddy + " tháng.",
          "Chênh lệch nghỉ việc quan sát được là " + vn1(RATES.buddy) + " điểm phần trăm; chưa phải tác động của buddy."
        ];
        actions = [
          "Chọn buddy, xác nhận tải công việc và tiêu chí ghép cặp.",
          "Ghi nhận mức tham gia, phản hồi và dấu hiệu nghỉ việc trước khi bắt đầu.",
          "So sánh kết quả cuối kỳ với nhóm chưa tham gia trước khi mở rộng."
        ];
        caution = "Không thể dùng chênh lệch hiện tại để tuyên bố chương trình sẽ giữ lại một số người cụ thể. Kết quả phù hợp ở giai đoạn này là một kế hoạch thí điểm có đo lường.";
        summary = "KẾ HOẠCH THÍ ĐIỂM BUDDY\nĐối tượng: " + fmtN(eligibleBuddy) + " nhân viên mới; quy mô thí điểm " + fmtN(pilotBuddy) + " người trong " + monthsBuddy + " tháng; " + checkin + " lần check-in/tháng.\nNgân sách dự kiến: " + moneyTxt(budget) + ".\nĐánh giá: mức tham gia, phản hồi và tỷ lệ nghỉ việc so với nhóm chưa tham gia.\nLưu ý: chưa có bằng chứng để cam kết số người giữ lại.";
      } else {
        var eligiblePay = val("p-pay-eligible", 1, 100000);
        var capacityPay = val("p-pay-capacity", 1, 100000);
        var reviewPay = Math.min(eligiblePay, capacityPay);
        var monthsPay = val("p-pay-months", 1, 12);
        var approval = $("p-pay-approval").value;
        badge.className = "chip amber";
        badge.textContent = "Cần thí điểm và đo lại";
        stats = [
          stat(fmtN(eligiblePay), "Hồ sơ cần rà soát", "Theo mặt bằng từng phòng ban"),
          stat(fmtN(reviewPay), "Hồ sơ xử lý đợt này", "Theo năng lực và ngân sách"),
          stat(monthsPay + " tháng", "Thời gian rà soát", "Phê duyệt bởi " + approval),
          stat(moneyTxt(budget), "Ngân sách dự kiến", "Không mặc định là ngân sách tăng lương")
        ];
        assumptions = [
          "Có " + eligiblePay + " hồ sơ cần rà soát và xử lý " + reviewPay + " hồ sơ trong đợt này.",
          "Quyết định cần " + approval + " phê duyệt.",
          "Chênh lệch nghỉ việc quan sát được là " + vn1(RATES.pay) + " điểm phần trăm; chưa phải tác động của điều chỉnh thu nhập.",
          "Mọi quyết định dựa trên vai trò, cấp bậc và tiêu chí công bằng đã thống nhất."
        ];
        actions = [
          "Chốt tiêu chí rà soát và loại bỏ thuộc tính nhạy cảm khỏi quyết định.",
          "Kiểm tra tính nhất quán giữa các phòng ban và cấp bậc tương đương.",
          "Theo dõi kết quả sau " + monthsPay + " tháng trước khi mở rộng."
        ];
        caution = "Rà soát không đồng nghĩa với cam kết tăng lương. Dữ liệu hiện tại chỉ giúp xác định nhóm cần xem xét, không chứng minh thay đổi thu nhập sẽ làm giảm nghỉ việc.";
        summary = "KẾ HOẠCH RÀ SOÁT THU NHẬP\nPhạm vi: " + fmtN(eligiblePay) + " hồ sơ; xử lý " + fmtN(reviewPay) + " hồ sơ trong " + monthsPay + " tháng.\nPhê duyệt: " + approval + ". Ngân sách dự kiến: " + moneyTxt(budget) + ".\nĐiều kiện: tiêu chí công bằng theo vai trò/cấp bậc và đo lại kết quả sau triển khai.\nLưu ý: chưa có bằng chứng để cam kết số người giữ lại hoặc mức tăng lương.";
      }

      $("policy-result-grid").innerHTML = stats.join("");
      $("policy-assumptions").innerHTML = list(assumptions);
      $("policy-next-actions").innerHTML = list(actions);
      $("policy-caution").textContent = caution;
      $("sim-code").textContent = summary;
      $("sim-code").hidden = true;
      savePolicy();
    }

    document.querySelectorAll("#policy-pick .pick").forEach(function (button) {
      button.addEventListener("click", function () { selectPolicy(button.dataset.policy); });
    });
    $("btn-policy-setup").addEventListener("click", function () { selectPolicy(cur); showStep(2); });
    $("btn-policy-back-1").addEventListener("click", function () { showStep(1); });
    $("btn-policy-result").addEventListener("click", function () { renderResult(); showStep(3); });
    $("btn-policy-back-2").addEventListener("click", function () { showStep(2); });
    $("btn-policy-restart").addEventListener("click", function () { showStep(1); });
    document.querySelectorAll("#page-chinh-sach input, #page-chinh-sach select").forEach(function (el) { el.addEventListener("change", savePolicy); });
    $("btn-copy").addEventListener("click", function () {
      var el = $("sim-code");
      el.hidden = false;
      if (navigator.clipboard) {
        navigator.clipboard.writeText(el.textContent).then(function () { toast("Đã sao chép bản trình duyệt"); }, function () { toast("Bản trình duyệt đã hiện bên dưới để sao chép"); });
      } else {
        toast("Bản trình duyệt đã hiện bên dưới để sao chép");
      }
    });

    if (st0.values && typeof st0.values === "object") {
      Object.keys(st0.values).forEach(function (id) { if ($(id)) $(id).value = st0.values[id]; });
    }
    selectPolicy(cur);
    if (step === 3) renderResult();
    showStep(step);
    window._openPolicy = function (key) { selectPolicy(key); showStep(1); };
  };
})();
