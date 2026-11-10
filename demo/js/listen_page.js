/* Auralisation page: one looping <audio> per case; switching keeps the playback position (A/B). */
(function () {
  "use strict";
  var data = Demo.requireData();
  if (!data) return;
  var A = data.audio;
  if (!A) { document.getElementById("cases").textContent = "No audio: run python -m cabinsim.demo_export without --no-audio."; return; }
  var bands = A.bands_hz;
  var players = {}, selected = A.cases[0].key, playing = false;
  var baseline = A.cases[0];

  document.getElementById("calibration").textContent =
    "In these files a full-scale sine would be " + A.full_scale_sine_spl_db.toFixed(0) + " dB SPL.";

  // ---- one card per listening case
  var box = document.getElementById("cases");
  A.cases.forEach(function (c, k) {
    var audio = new Audio(c.file);
    audio.loop = true;
    audio.preload = "auto";
    players[c.key] = audio;
    var card = document.createElement("button");
    card.className = "card case";
    card.style.textAlign = "left";
    card.innerHTML = "<h2>" + c.label + "</h2>" +
      '<div class="stat"><div>SIL<b>' + c.sil_db.toFixed(1) + ' dB</b></div><div>dB(A)<b>' + c.dba.toFixed(1) +
      "</b></div><div>vs baseline<b>" + (k === 0 ? "–" : (c.dba - baseline.dba).toFixed(1) + " dB(A)") + "</b></div></div>";
    card.addEventListener("click", function () { select(c.key); });
    card.dataset.key = c.key;
    box.appendChild(card);
  });

  function select(key) {
    var previous = players[selected], next = players[key];
    if (key !== selected) {
      next.currentTime = previous.currentTime;
      if (playing) { next.play(); previous.pause(); }
      selected = key;
    }
    Array.prototype.forEach.call(box.children, function (card) { card.classList.toggle("active", card.dataset.key === key); });
    var c = A.cases.filter(function (x) { return x.key === key; })[0];
    document.getElementById("status").textContent = "Now: " + c.label;
    drawChart(c);
  }

  document.getElementById("play").addEventListener("click", function (e) {
    playing = !playing;
    if (playing) { players[selected].play(); e.target.textContent = "❚❚ Pause"; }
    else { players[selected].pause(); e.target.textContent = "▶ Play"; }
  });

  // ---- spectrum chart: baseline vs selected
  var chart = document.getElementById("chart"), ctx = chart.getContext("2d");
  var pad = { l: 56, r: 16, t: 16, b: 34 };
  var all = [];
  A.cases.forEach(function (c) { all = all.concat(c.spectrum_db); });
  var yMax = Math.ceil(Math.max.apply(null, all) / 10) * 10, yMin = Math.max(0, yMax - 70);
  function x(k) { return pad.l + k / (bands.length - 1) * (chart.width - pad.l - pad.r); }
  function y(v) { return chart.height - pad.b - (Math.max(v, yMin) - yMin) / (yMax - yMin) * (chart.height - pad.t - pad.b); }
  function line(values, colour, width) {
    ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.beginPath();
    values.forEach(function (v, k) { if (k === 0) ctx.moveTo(x(k), y(v)); else ctx.lineTo(x(k), y(v)); });
    ctx.stroke();
  }
  function drawChart(c) {
    ctx.clearRect(0, 0, chart.width, chart.height);
    ctx.font = "12px system-ui, sans-serif"; ctx.fillStyle = Demo.INK2; ctx.strokeStyle = "#e1e0d9"; ctx.lineWidth = 1;
    for (var v = yMin; v <= yMax; v += 10) {
      ctx.beginPath(); ctx.moveTo(pad.l, y(v)); ctx.lineTo(chart.width - pad.r, y(v)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(v + " dB", pad.l - 6, y(v) + 4);
    }
    bands.forEach(function (b, k) {
      if ([63, 125, 250, 500, 1000, 2000, 4000].indexOf(b) < 0) return;
      ctx.textAlign = "center"; ctx.fillText(b >= 1000 ? b / 1000 + "k" : String(b), x(k), chart.height - 12);
    });
    var comparing = c.key !== baseline.key;
    var baselineColour = comparing ? "#c3c2b7" : Demo.SERIES[0];      // grey only when it is the reference
    line(baseline.spectrum_db, baselineColour, 2);
    if (comparing) line(c.spectrum_db, Demo.SERIES[0], 2);
    document.getElementById("legend").innerHTML = '<span style="--swatch:' + baselineColour + '">' + baseline.label + "</span>" +
      (comparing ? '<span style="--swatch:' + Demo.SERIES[0] + '">' + c.label + "</span>" : "");
  }
  Demo.attachTooltip(chart, function (fx) {
    var k = Math.round((fx * chart.width - pad.l) / (chart.width - pad.l - pad.r) * (bands.length - 1));
    if (k < 0 || k >= bands.length) return null;
    var c = A.cases.filter(function (x) { return x.key === selected; })[0];
    return bands[k] + " Hz\n" + baseline.label + ": " + baseline.spectrum_db[k].toFixed(1) + " dB" +
      (c.key !== baseline.key ? "\n" + c.label + ": " + c.spectrum_db[k].toFixed(1) + " dB" : "");
  });

  select(selected);
})();
