/* Quiet-zone maps: horizontal slice through the head, panel along the bottom edge. */
(function () {
  "use strict";
  var data = Demo.requireData();
  if (!data) return;
  var Q = data.quiet_zone;
  if (!Q) { document.querySelector(".page").insertAdjacentHTML("beforeend", "<p>No solution with an active trim panel.</p>"); return; }
  var LEVEL_RANGE = [-25, 10], ATT_RANGE = 20, HEAD_RADIUS = 0.09;
  var n = Q.x_m.length, x0 = Q.x_m[0], x1 = Q.x_m[n - 1], z0 = Q.z_m[0], z1 = Q.z_m[n - 1];
  var freqInput = document.getElementById("freq");
  freqInput.max = Q.frequencies_hz.length - 1;
  document.getElementById("headDistance").textContent = Q.head_m[2].toFixed(2);
  document.getElementById("realisations").textContent = Q.realisations;
  var current = 0;

  // map value grid -> canvas: x to the right, z (into the cabin) upwards
  function px(x, canvas) { return (x - x0) / (x1 - x0) * canvas.width; }
  function py(z, canvas) { return canvas.height - (z - z0) / (z1 - z0) * canvas.height; }

  function drawMap(canvasId, values, colour) {
    var canvas = document.getElementById(canvasId), ctx = canvas.getContext("2d");
    var image = new ImageData(n, n);
    for (var i = 0; i < n; i++) for (var k = 0; k < n; k++) {
      var rgb = colour(values[i][k]), o = ((n - 1 - k) * n + i) * 4;
      image.data[o] = rgb[0]; image.data[o + 1] = rgb[1]; image.data[o + 2] = rgb[2]; image.data[o + 3] = 255;
    }
    var buffer = document.createElement("canvas");
    buffer.width = n; buffer.height = n;
    buffer.getContext("2d").putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(buffer, 0, 0, canvas.width, canvas.height);
    // overlays: panel edge, head, microphones (projected onto the slice)
    ctx.strokeStyle = Demo.INK; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(px(Math.max(x0, 0), canvas), canvas.height - 2);
    ctx.lineTo(px(Math.min(x1, Q.panel_size_m[0]), canvas), canvas.height - 2); ctx.stroke();
    ctx.lineWidth = 2; ctx.strokeStyle = Demo.INK;
    ctx.beginPath();
    ctx.arc(px(Q.head_m[0], canvas), py(Q.head_m[2], canvas), HEAD_RADIUS / (x1 - x0) * canvas.width, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.fillStyle = "#eb6834"; ctx.strokeStyle = "#fcfcfb";
    Q.mics_m.forEach(function (m) {
      ctx.beginPath(); ctx.arc(px(m[0], canvas), py(m[2], canvas), 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    });
    return canvas;
  }

  function levelColour(v) { return Demo.sequential((v - LEVEL_RANGE[0]) / (LEVEL_RANGE[1] - LEVEL_RANGE[0])); }
  function attColour(v) { return Demo.diverging(0.5 - 0.5 * Math.max(-1, Math.min(1, v / ATT_RANGE))); }

  function show(index) {
    current = index;
    freqInput.value = index;
    var f = Q.frequencies_hz[index], off = Q.off_db[index], on = Q.on_db[index];
    var att = off.map(function (row, i) { return row.map(function (v, k) { return v - on[i][k]; }); });
    drawMap("off", off, levelColour);
    drawMap("on", on, levelColour);
    var canvas = drawMap("att", att, attColour), ctx = canvas.getContext("2d");
    // outline of the >= 10 dB region
    ctx.fillStyle = Demo.INK;
    var cw = canvas.width / n;
    for (var i = 0; i < n; i++) for (var k = 0; k < n; k++) {
      var inside = att[i][k] >= 10;
      var edge = inside && ((i > 0 && att[i - 1][k] < 10) || (i < n - 1 && att[i + 1][k] < 10) ||
                            (k > 0 && att[i][k - 1] < 10) || (k < n - 1 && att[i][k + 1] < 10));
      if (edge) ctx.fillRect(i * cw + cw / 2 - 1.5, canvas.height - (k + 0.5) * cw - 1.5, 3, 3);
    }
    var isTone = Q.tones_hz.some(function (t) { return Math.abs(t - f) < 0.5; });
    document.getElementById("freqValue").textContent = f.toFixed(0) + " Hz" + (isTone ? " (engine tone)" : "");
    document.getElementById("diameter").textContent = Q.zone_diameter_m[index] > 0
      ? (Q.zone_diameter_m[index] * 100).toFixed(0) + " cm" : "none";
    var hi = nearest(Q.x_m, Q.head_m[0]), hk = nearest(Q.z_m, Q.head_m[2]);
    document.getElementById("headAtt").textContent = att[hi][hk].toFixed(1) + " dB";
    document.getElementById("share").textContent = (Q.controlled_share[index] * 100).toFixed(0) + " %";
    drawChart();
  }

  function nearest(list, value) {
    var best = 0;
    list.forEach(function (v, k) { if (Math.abs(v - value) < Math.abs(list[best] - value)) best = k; });
    return best;
  }

  // ---- chart: equivalent diameter and lambda/10 vs frequency (log x)
  var chart = document.getElementById("chart"), cctx = chart.getContext("2d");
  var fs = Q.frequencies_hz, fMin = fs[0], fMax = fs[fs.length - 1];
  var yMax = Math.max(Math.max.apply(null, Q.zone_diameter_m), Math.max.apply(null, Q.wavelength_over_10_m)) * 1.1;
  var pad = { l: 52, r: 16, t: 22, b: 30 };
  function cx(f) { return pad.l + Math.log(f / fMin) / Math.log(fMax / fMin) * (chart.width - pad.l - pad.r); }
  function cy(v) { return chart.height - pad.b - v / yMax * (chart.height - pad.t - pad.b); }
  // the two lines meet at high frequency, so identity comes from the legend below the chart
  function series(values, colour) {
    cctx.strokeStyle = colour; cctx.lineWidth = 2; cctx.lineJoin = "round"; cctx.beginPath();
    values.forEach(function (v, k) { if (k === 0) cctx.moveTo(cx(fs[k]), cy(v)); else cctx.lineTo(cx(fs[k]), cy(v)); });
    cctx.stroke();
  }
  function drawChart() {
    cctx.clearRect(0, 0, chart.width, chart.height);
    cctx.font = "12px system-ui, sans-serif"; cctx.strokeStyle = "#e1e0d9"; cctx.lineWidth = 1; cctx.fillStyle = Demo.INK2;
    for (var v = 0; v <= yMax; v += 0.1) {
      cctx.beginPath(); cctx.moveTo(pad.l, cy(v)); cctx.lineTo(chart.width - pad.r, cy(v)); cctx.stroke();
      cctx.textAlign = "right"; cctx.fillText((v * 100).toFixed(0) + " cm", pad.l - 6, cy(v) + 4);
    }
    [63, 125, 250, 500, 1000].forEach(function (f) {
      if (f < fMin || f > fMax) return;
      cctx.textAlign = "center"; cctx.fillText(f + " Hz", cx(f), chart.height - 10);
    });
    series(Q.wavelength_over_10_m, "#eb6834");
    series(Q.zone_diameter_m, "#2a78d6");
    cctx.fillStyle = "#2a78d6"; cctx.strokeStyle = "#fcfcfb"; cctx.lineWidth = 2;
    cctx.beginPath(); cctx.arc(cx(fs[current]), cy(Q.zone_diameter_m[current]), 5, 0, 2 * Math.PI); cctx.fill(); cctx.stroke();
  }

  // ---- hover readouts
  ["off", "on", "att"].forEach(function (id) {
    Demo.attachTooltip(document.getElementById(id), function (fx, fy) {
      var i = Math.min(n - 1, Math.floor(fx * n)), k = Math.min(n - 1, Math.floor((1 - fy) * n));
      var off = Q.off_db[current][i][k], on = Q.on_db[current][i][k];
      return "x " + (Q.x_m[i] - Q.head_m[0]).toFixed(2) + " m, depth " + Q.z_m[k].toFixed(2) + " m\n" +
             "off " + off.toFixed(1) + " dB, on " + on.toFixed(1) + " dB\nreduction " + (off - on).toFixed(1) + " dB";
    });
  });
  Demo.attachTooltip(chart, function (fx) {
    var k = nearest(fs.map(cx), fx * chart.width);
    return fs[k].toFixed(0) + " Hz\nzone " + (Q.zone_diameter_m[k] * 100).toFixed(0) + " cm\nλ/10 " +
           (Q.wavelength_over_10_m[k] * 100).toFixed(0) + " cm";
  });

  // ---- controls
  var toneBox = document.getElementById("toneButtons");
  Q.tones_hz.forEach(function (t) {
    var b = document.createElement("button");
    b.textContent = t.toFixed(0) + " Hz tone";
    b.style.marginRight = "6px";
    b.addEventListener("click", function () { show(nearest(fs, t)); });
    toneBox.appendChild(b);
  });
  freqInput.addEventListener("input", function () { show(Number(freqInput.value)); });
  Demo.drawColorBar(document.getElementById("barLevel"), Demo.sequential, LEVEL_RANGE[0], LEVEL_RANGE[1],
                    "level (dB re. head level without ANC)");
  Demo.drawColorBar(document.getElementById("barAtt"), function (t) { return Demo.diverging(1 - t); }, -ATT_RANGE, ATT_RANGE,
                    "reduction (dB): blue = quieter, red = louder");
  show(nearest(fs, Q.tones_hz[0]));
})();
