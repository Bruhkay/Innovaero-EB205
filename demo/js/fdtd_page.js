/* Cross-section FDTD page: two solvers (baseline and chosen solution) side by side. */
(function () {
  "use strict";
  var data = Demo.requireData();
  if (!data) return;
  var F = data.fdtd;
  var F_MIN = 63, F_MAX = 800, DYNAMIC_RANGE_DB = 60, SOURCE_RATE = 1e4;
  var PLATE = [59, 58, 54], BLANKET = [233, 217, 166], FLOOR = [155, 154, 148], MARK = [235, 104, 52];

  document.getElementById("exaggeration").textContent = F.exaggeration;
  var freqInput = document.getElementById("freq"), freqValue = document.getElementById("freqValue");
  var select = document.getElementById("solution"), speed = document.getElementById("speed");
  Object.keys(F.solutions).forEach(function (name) {
    if (name === F.baseline) return;
    var option = document.createElement("option");
    option.value = name; option.textContent = F.solutions[name].label;
    select.appendChild(option);
  });
  select.value = F.compare in F.solutions ? F.compare : select.options[0].value;
  freqInput.value = Math.log(250 / F_MIN) / Math.log(F_MAX / F_MIN);

  function frequency() { return F_MIN * Math.pow(F_MAX / F_MIN, Number(freqInput.value)); }

  // A panel = one solver + its canvas, material tint and head-level average.
  function makePanel(canvasId, titleId, name) {
    var solution = F.solutions[name];
    var model = { nx: F.nx, nz: F.nz, dx: F.dx, sponge_cells: F.sponge_cells, source_cell: F.source_cell,
                  materials: solution.materials, index: Demo.base64ToBytes(solution.index_b64) };
    var canvas = document.getElementById(canvasId);
    canvas.width = F.nx; canvas.height = F.nz;
    document.getElementById(titleId).textContent = solution.label;
    var tint = new Array(model.materials.length);
    model.materials.forEach(function (m, k) {
      tint[k] = m.rigid ? FLOOR : /skin|septum|trim/.test(m.name) ? PLATE : m.name === "blanket" ? BLANKET : null;
    });
    return { name: name, model: model, solver: FdtdCore.createSolver(model), canvas: canvas,
             ctx: canvas.getContext("2d"), image: canvas.getContext("2d").createImageData(F.nx, F.nz),
             tint: tint, headSquare: 0 };
  }

  var left = makePanel("left", "leftTitle", F.baseline);
  var right = makePanel("right", "rightTitle", select.value);
  var playing = true, peak = 1e-12;
  var probe = F.probe_cell[0] * F.nz + F.probe_cell[1];

  function restart() {
    [left, right].forEach(function (panel) { panel.solver.reset(); panel.headSquare = 0; });
    peak = 1e-12;
    freqValue.textContent = frequency().toFixed(0) + " Hz";
    updateModelDelta();
  }

  // Signed log scale so the weak cabin field stays visible next to the strong outside field.
  function draw(panel) {
    var p = panel.solver.p, pixels = panel.image.data, nx = F.nx, nz = F.nz, index = panel.model.index;
    for (var i = 0; i < nx; i++) {
      for (var j = 0; j < nz; j++) {
        var cell = i * nz + j, value = p[cell];
        var magnitude = Math.abs(value) / peak;
        var level = magnitude > 0 ? Math.max(0, 1 + 20 * Math.log10(magnitude) / DYNAMIC_RANGE_DB) : 0;
        var rgb = Demo.diverging(0.5 + 0.5 * (value >= 0 ? level : -level));
        var tint = panel.tint[index[cell]];
        var o = ((nz - 1 - j) * nx + i) * 4;
        if (tint) {
          var w = tint === BLANKET ? 0.45 : 0.85;
          pixels[o] = tint[0] * w + rgb[0] * (1 - w);
          pixels[o + 1] = tint[1] * w + rgb[1] * (1 - w);
          pixels[o + 2] = tint[2] * w + rgb[2] * (1 - w);
        } else {
          pixels[o] = rgb[0]; pixels[o + 1] = rgb[1]; pixels[o + 2] = rgb[2];
        }
        pixels[o + 3] = 255;
      }
    }
    [F.source_cell, F.probe_cell].forEach(function (c) {
      for (var di = -2; di <= 2; di++) for (var dj = -2; dj <= 2; dj++) {
        if (Math.abs(di) + Math.abs(dj) > 3) continue;
        var o = ((nz - 1 - (c[1] + dj)) * nx + c[0] + di) * 4;
        pixels[o] = MARK[0]; pixels[o + 1] = MARK[1]; pixels[o + 2] = MARK[2];
      }
    });
    panel.ctx.putImageData(panel.image, 0, 0);
  }

  // Advance one panel; returns the largest |p| in it (for the shared colour scale).
  function advance(panel, steps, f) {
    var s = panel.solver, alpha = Math.min(1, s.dt * f / 3);       // ~3-period running mean of p^2 at the head
    for (var n = 0; n < steps; n++) {
      s.step(SOURCE_RATE * FdtdCore.toneSource(s.time, f));
      panel.headSquare += alpha * (s.p[probe] * s.p[probe] - panel.headSquare);
    }
    var largest = 0;
    for (var k = 0; k < s.p.length; k++) largest = Math.max(largest, Math.abs(s.p[k]));
    return largest;
  }

  function nearestBand(f) {
    var best = 0;
    F.bands_hz.forEach(function (b, k) { if (Math.abs(Math.log(b / f)) < Math.abs(Math.log(F.bands_hz[best] / f))) best = k; });
    return best;
  }

  function updateModelDelta() {
    var a = F.solutions[left.name].head_band_db, b = F.solutions[right.name].head_band_db;
    var k = nearestBand(frequency());
    document.getElementById("modelDelta").textContent = a && b
      ? (a[k] - b[k]).toFixed(1) + " dB quieter (" + F.bands_hz[k] + " Hz band)" : "–";
  }

  function frame() {
    if (playing) {
      var f = frequency(), steps = Number(speed.value);
      var largest = Math.max(advance(left, steps, f), advance(right, steps, f));
      peak = Math.max(peak * 0.97, largest);                        // slowly following the field maximum
      draw(left); draw(right);
      var arrived = left.solver.time > 0.06 + 6 / f;
      document.getElementById("fdtdDelta").textContent = arrived && right.headSquare > 0
        ? (10 * Math.log10(left.headSquare / right.headSquare)).toFixed(1) + " dB quieter" : "waiting…";
    }
    requestAnimationFrame(frame);
  }

  Demo.drawColorBar(document.getElementById("bar"), Demo.diverging, -1, 1,
                    "pressure, signed log scale", [[-1, "−peak"], [0, "≤ " + DYNAMIC_RANGE_DB + " dB below peak"], [1, "+peak"]]);
  freqInput.addEventListener("input", restart);
  select.addEventListener("change", function () { right = makePanel("right", "rightTitle", select.value); restart(); });
  document.getElementById("restart").addEventListener("click", restart);
  document.getElementById("play").addEventListener("click", function (e) {
    playing = !playing; e.target.textContent = playing ? "Pause" : "Play";
  });
  restart();
  requestAnimationFrame(frame);
})();
