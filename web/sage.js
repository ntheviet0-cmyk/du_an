/* sage.js - GraphSAGE 51->64->32->logit chay trong trinh duyet (khong thu vien).
   Tai hien dung ngu nghia PyG SAGEConv (aggr mean, root_weight, bias):
     h  = relu( W1l . mean(neighbors) + b1 + W1r . x )
     emb=      W2l . mean(neighbors h) + b2 + W2r . h
     p  = sigmoid( Wlin . relu(emb) + blin )
   Gan node moi vao graph train qua top-k cosine (k=15) dung nhu graph_shap_lib.py.
   UMD: window.HRSage (browser) / module.exports (Node de test parity). */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.HRSage = factory();
}(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var EPS = 1e-9;

  function toF64(a) { return a instanceof Float64Array ? a : Float64Array.from(a); }

  function toF64flat(a) {
    if (Array.isArray(a) && a.length && Array.isArray(a[0])) {
      var rows = a.length, cols = a[0].length, out = new Float64Array(rows * cols), r, c;
      for (r = 0; r < rows; r++) for (c = 0; c < cols; c++) out[r * cols + c] = a[r][c];
      return out;
    }
    return toF64(a);
  }

  function matvecInto(W, x, out) {
    var ni = x.length, o = out.length, k = 0;
    for (var oi = 0; oi < o; oi++) {
      var s = 0;
      for (var ii = 0; ii < ni; ii++) s += W[k++] * x[ii];
      out[oi] = s;
    }
    return out;
  }

  /* Dung vector 51 chieu tu du lieu form (raw) theo PRE.feat_names. */
  function buildX(raw, PRE) {
    var feats = PRE.feat_names;
    var x = new Float64Array(feats.length);
    var numIdx = {};
    var i;
    for (i = 0; i < PRE.num_cols.length; i++) numIdx[PRE.num_cols[i]] = i;
    var catCols = PRE.cat_cols;
    for (var k = 0; k < feats.length; k++) {
      var f = feats[k];
      if (f.indexOf("num__") === 0) {
        var nc = f.slice(5);
        var ni = numIdx[nc];
        x[k] = (Number(raw[nc]) - PRE.means[ni]) / PRE.scales[ni];
      } else if (f.indexOf("cat__") === 0) {
        var rest = f.slice(5), col = null, val = null;
        for (var c = 0; c < catCols.length; c++) {
          if (rest.indexOf(catCols[c] + "_") === 0) { col = catCols[c]; val = rest.slice(col.length + 1); break; }
        }
        x[k] = (col !== null && String(raw[col]) === val) ? 1 : 0;
      }
    }
    return x;
  }

  function bandOf(p, thr) {
    return p >= thr ? "red" : (p >= thr * 0.6 ? "amber" : "green");
  }

  /* W: window.HR.W, X_TRAIN: mang 2 chieu, EDGES: {rows, cols, n}, k: so hang xom. */
  function createModel(W, X_TRAIN, EDGES, k) {
    k = k || 15;
    var n = EDGES.n, d = X_TRAIN[0].length;
    var flat = new Float64Array(n * d);
    var i, j, e;
    for (i = 0; i < n; i++) for (j = 0; j < d; j++) flat[i * d + j] = X_TRAIN[i][j];

    var norms = new Float64Array(n);
    for (i = 0; i < n; i++) {
      var sq = 0;
      var base = i * d;
      for (j = 0; j < d; j++) sq += flat[base + j] * flat[base + j];
      norms[i] = Math.sqrt(sq);
    }

    /* Tong va so canh vao (incoming) tung node - de tinh mean(neighbors) nhanh. */
    var inSum = new Float64Array(n * d);
    var inCnt = new Int32Array(n);
    var rows = EDGES.rows, cols = EDGES.cols;
    for (e = 0; e < rows.length; e++) {
      var u = rows[e], v = cols[e];
      inCnt[v]++;
      var ub = u * d, vb = v * d;
      for (j = 0; j < d; j++) inSum[vb + j] += flat[ub + j];
    }

    var W1l = toF64flat(W["c1.lin_l.weight"]), B1 = toF64(W["c1.lin_l.bias"]), W1r = toF64flat(W["c1.lin_r.weight"]);
    var W2l = toF64flat(W["c2.lin_l.weight"]), B2 = toF64(W["c2.lin_l.bias"]), W2r = toF64flat(W["c2.lin_r.weight"]);
    var WL = toF64flat(W["lin.weight"]), BL = toF64(W["lin.bias"]);
    var h1dim = B1.length, h2dim = B2.length;

    var agg = new Float64Array(d);
    var t1 = new Float64Array(h1dim), t2 = new Float64Array(h1dim);
    var agg2 = new Float64Array(h1dim), t3 = new Float64Array(h2dim), t4 = new Float64Array(h2dim);
    var xbuf = new Float64Array(d);
    var h1new = new Float64Array(h1dim);
    var h1mat = new Float64Array(k * h1dim);
    var embBuf = new Float64Array(h2dim);
    var logitBuf = new Float64Array(1);
    var sims = new Float64Array(n);
    var order = new Int32Array(n);
    var top = new Int32Array(k);

    function h1FromAgg(aggVec, selfVec, out) {
      matvecInto(W1l, aggVec, t1);
      matvecInto(W1r, selfVec, t2);
      for (var o = 0; o < h1dim; o++) {
        var z = t1[o] + B1[o] + t2[o];
        out[o] = z > 0 ? z : 0;
      }
      return out;
    }

    function predict(xin) {
      var x = toF64(xin);
      var nx = 0;
      for (j = 0; j < d; j++) nx += x[j] * x[j];
      nx = Math.sqrt(nx);
      for (i = 0; i < n; i++) {
        var base = i * d, s = 0;
        for (j = 0; j < d; j++) s += flat[base + j] * x[j];
        sims[i] = s / (norms[i] * (nx + EPS));
        order[i] = i;
      }
      var ord = Array.prototype.slice.call(order);
      ord.sort(function (a, b) { return sims[b] - sims[a] || a - b; });
      for (var t = 0; t < k; t++) top[t] = ord[t];

      /* h1 cua node moi: mean x cua k hang xom gan nhat */
      for (j = 0; j < d; j++) agg[j] = 0;
      for (t = 0; t < k; t++) {
        var nb = top[t] * d;
        for (j = 0; j < d; j++) agg[j] += flat[nb + j];
      }
      for (j = 0; j < d; j++) agg[j] /= k;
      h1FromAgg(agg, x, h1new);

      /* h1 cua tung hang xom (co them node moi trong danh sach canh vao) */
      for (t = 0; t < k; t++) {
        var node = top[t], nb2 = node * d;
        for (j = 0; j < d; j++) agg[j] = (inSum[nb2 + j] + x[j]) / (inCnt[node] + 1);
        h1FromAgg(agg, flat.subarray(nb2, nb2 + d), h1mat.subarray(t * h1dim, (t + 1) * h1dim));
      }

      /* h2 cua node moi: mean h1 cua k hang xom */
      for (var o = 0; o < h1dim; o++) agg2[o] = 0;
      for (t = 0; t < k; t++) {
        var hb = t * h1dim;
        for (o = 0; o < h1dim; o++) agg2[o] += h1mat[hb + o];
      }
      for (o = 0; o < h1dim; o++) agg2[o] /= k;
      matvecInto(W2l, agg2, t3);
      matvecInto(W2r, h1new, t4);
      for (o = 0; o < h2dim; o++) embBuf[o] = t3[o] + B2[o] + t4[o];

      for (o = 0; o < h2dim; o++) embBuf[o] = embBuf[o] > 0 ? embBuf[o] : 0;
      matvecInto(WL, embBuf, logitBuf);
      var logit = logitBuf[0] + BL[0];
      return 1 / (1 + Math.exp(-logit));
    }

    return { predict: predict, bandOf: bandOf, dim: d, n: n };
  }

  return { buildX: buildX, createModel: createModel, bandOf: bandOf };
}));
