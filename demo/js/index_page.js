/* Overview page: results table filled from demo_data.js (never hard-coded numbers). */
(function () {
  "use strict";
  var data = Demo.requireData();
  if (!data) return;
  var summary = data.summary, names = Object.keys(summary), baseline = summary[names[0]];
  var low = data.target_sil_db[0], high = data.target_sil_db[1];
  document.getElementById("target").textContent = low + "–" + high + " dB";

  var rows = names.map(function (name) {
    var s = summary[name];
    return [s.label, s.head_sil_db, s.head_dba, s.head_dba - baseline.head_dba, s.added_mass_kg_m2];
  });
  if (data.audio) {                                   // ANC variants come from the auralisation list
    data.audio.cases.forEach(function (c) {
      if (/_anc$/.test(c.key)) {
        var base = summary[c.key.replace(/_anc$/, "")];
        rows.push([c.label, c.sil_db, c.dba, c.dba - baseline.head_dba, base.added_mass_kg_m2]);
      }
    });
  }
  var html = "<tr><th>Solution</th><th>SIL (dB)</th><th>dB(A)</th><th>Δ dB(A) vs baseline</th><th>Added mass (kg/m²)</th></tr>";
  rows.forEach(function (r, k) {
    html += "<tr><td>" + r[0] + "</td><td>" + r[1].toFixed(1) + (r[1] <= high ? " ✓" : "") + "</td><td>" + r[2].toFixed(1) +
      "</td><td>" + (k === 0 ? "–" : r[3].toFixed(1)) + "</td><td>" + r[4].toFixed(2) + "</td></tr>";
  });
  document.getElementById("results").innerHTML = html;
})();
