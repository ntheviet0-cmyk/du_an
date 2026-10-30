/* test_sage_parity.js - Kiem tra sage.js khop voi Python (parity.js sinh boi export_web_data.py).
   Chay: node web/test_sage_parity.js
   1) buildX(raw) so voi PARITY.Xt (dau ra that cua preprocessor.joblib)
   2) predict(Xt) so voi PARITY.proba (graph_predict_proba cua PyTorch) */
"use strict";
const fs = require("fs");
const path = require("path");

const HR = {};
function load(rel) {
  const code = fs.readFileSync(path.join(__dirname, rel), "utf8");
  new Function("window", code)(HR);
}
load("data/weights.js");
load("data/xtrain.js");
load("data/edges.js");
load("data/preproc.js");
load("data/parity.js");

const sage = require("./sage.js");
const { W, X_TRAIN, EDGES, PRE, PARITY } = HR.HR;
const model = sage.createModel(W, X_TRAIN, EDGES, 15);

let maxX = 0, maxP = 0, maxPraw = 0;
PARITY.Xt.forEach(function (target, i) {
  const x = sage.buildX(PARITY.raw[i], PRE);
  for (let j = 0; j < target.length; j++) {
    const e = Math.abs(x[j] - target[j]);
    if (e > maxX) maxX = e;
  }
  const p = model.predict(target);
  const pr = model.predict(x);
  maxP = Math.max(maxP, Math.abs(p - PARITY.proba[i]));
  maxPraw = Math.max(maxPraw, Math.abs(pr - PARITY.proba[i]));
});

console.log("max |buildX - PARITY.Xt|      =", maxX.toExponential(3));
console.log("max |predict(Xt) - proba_py|  =", maxP.toExponential(3));
console.log("max |predict(buildX) - proba| =", maxPraw.toExponential(3));

const X_TOL = 1e-4, P_TOL = 2e-3;
const bad = (v) => Number.isNaN(v) || v > 0;
if (bad(maxX - X_TOL) || bad(maxP - P_TOL) || bad(maxPraw - P_TOL)) {
  console.error("PARITY_FAIL (X_TOL=" + X_TOL + ", P_TOL=" + P_TOL + ")");
  process.exit(1);
}
console.log("PARITY_OK");
