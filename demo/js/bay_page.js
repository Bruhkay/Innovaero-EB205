/* Skin bay animation: w(x, y, t) = Re( sum_m v_m / (j w) phi_m(x, y) e^{j phase} ), short vs shunted. */
(function () {
  "use strict";
  var data = Demo.requireData();
  if (!data) return;
  var B = data.bay;
  if (!B) { document.querySelector(".page").insertAdjacentHTML("beforeend", "<p>No solution with piezo shunts.</p>"); return; }
  var NX = 56, NY = 24, HEIGHT = 0.06, VISUAL_HZ = 0.6;     // grid, max drawn displacement (m), slow-motion rate
  var nModes = B.modes.length;
  var freqInput = document.getElementById("freq");
  freqInput.max = B.frequencies_hz.length - 1;

  // mode shapes on the drawing grid: shapes[m][k], k = ix*(NY+1)+iy
  var shapes = B.modes.map(function (mode) {
    var s = new Float32Array((NX + 1) * (NY + 1));
    for (var ix = 0; ix <= NX; ix++) for (var iy = 0; iy <= NY; iy++) {
      s[ix * (NY + 1) + iy] = Math.sin(mode[0] * Math.PI * ix / NX) * Math.sin(mode[1] * Math.PI * iy / NY);
    }
    return s;
  });

  function makeView(elementId) {
    var container = document.getElementById(elementId);
    var renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(renderer.domElement);
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0xeef1f5);
    scene.add(new THREE.HemisphereLight(0xffffff, 0xb8bcc4, 2.0));
    var light = new THREE.DirectionalLight(0xffffff, 1.3);
    light.position.set(0.3, 1, 0.6);
    scene.add(light);
    var geometry = new THREE.PlaneGeometry(B.a, B.b, NX, NY);
    geometry.rotateX(-Math.PI / 2);                        // bay in the horizontal plane, displacement upwards
    geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(geometry.attributes.position.count * 3), 3));
    var mesh = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    scene.add(mesh);
    B.patches.forEach(function (patch) {                   // patch outlines, slightly above the surface
      var cx = (patch.centre[0] - 0.5) * B.a, cz = -(patch.centre[1] - 0.5) * B.b;
      var hx = patch.size[0] / 2, hz = patch.size[1] / 2;
      var outline = [[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz], [-hx, -hz]].map(function (q) {
        return new THREE.Vector3(cx + q[0], 0.004, cz + q[1]);
      });
      var line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(outline),
                                new THREE.LineBasicMaterial({ color: 0x0b0b0b, depthTest: false }));
      line.renderOrder = 1;                                // drawn on top of the moving surface
      scene.add(line);
    });
    var camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20);
    var view = { renderer: renderer, scene: scene, camera: camera, geometry: geometry, container: container };
    view.render = function () { renderer.render(scene, camera); };
    Demo.orbitControl(camera, renderer.domElement, new THREE.Vector3(0, 0, 0), null).set(1.2, 0.85, 0.6);
    return view;
  }

  var views = [makeView("viewShort"), makeView("viewShunt")];
  var configs = ["short", B.shunt];
  document.getElementById("shuntTitle").textContent = B.shunt + " shunt (tuned)";
  document.getElementById("incidence").textContent = "plane wave, " + B.incidence_deg[0] + "° from the normal, " +
    (B.incidence_deg[1] === 0 ? "travelling along the fuselage axis" : B.incidence_deg[1] + "° from the axis");
  var color = new THREE.Color();

  // complex displacement amplitudes per drawing point for frequency index f
  var amplitudes = [null, null], scale = 1;
  function prepare(index) {
    var omega = 2 * Math.PI * B.frequencies_hz[index], peak = 0;
    configs.forEach(function (config, c) {
      var field = B.fields[config][index];
      var re = new Float32Array(shapes[0].length), im = new Float32Array(shapes[0].length);
      for (var m = 0; m < nModes; m++) {
        var vr = field[m], vi = field[nModes + m];           // w = v / (j w) = (vi - j vr) / w
        var wr = vi / omega, wi = -vr / omega, s = shapes[m];
        for (var k = 0; k < s.length; k++) { re[k] += wr * s[k]; im[k] += wi * s[k]; }
      }
      for (var k = 0; k < re.length; k++) peak = Math.max(peak, Math.hypot(re[k], im[k]));
      amplitudes[c] = [re, im];
    });
    scale = HEIGHT / peak;
    freqInput.value = index;
    document.getElementById("freqValue").textContent = B.frequencies_hz[index].toFixed(0) + " Hz";
    document.getElementById("reduction").textContent = B.velocity_reduction_db[index].toFixed(1) + " dB";
    drawChart(index);
  }

  function animate(time) {
    var phase = 2 * Math.PI * VISUAL_HZ * time / 1000, cos = Math.cos(phase), sin = Math.sin(phase);
    views.forEach(function (view, c) {
      var pos = view.geometry.attributes.position, col = view.geometry.attributes.color;
      var re = amplitudes[c][0], im = amplitudes[c][1];
      for (var ix = 0; ix <= NX; ix++) for (var iy = 0; iy <= NY; iy++) {
        var k = ix * (NY + 1) + iy, vertex = iy * (NX + 1) + ix;       // PlaneGeometry rows run along y
        var w = (re[k] * cos - im[k] * sin) * scale;
        pos.setY(vertex, w);
        var rgb = Demo.diverging(0.5 + 0.5 * w / HEIGHT);
        color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace);
        col.setXYZ(vertex, color.r, color.g, color.b);
      }
      pos.needsUpdate = true; col.needsUpdate = true;
      view.geometry.computeVertexNormals();
      view.render();
    });
    requestAnimationFrame(animate);
  }

  // ---- reduction chart (single series, value labels at the tuned modes)
  var chart = document.getElementById("chart"), ctx = chart.getContext("2d");
  var fMin = B.frequencies_hz[0], fMax = B.frequencies_hz[B.frequencies_hz.length - 1];
  var yMax = Math.max.apply(null, B.velocity_reduction_db) + 1, yMin = Math.min(0, Math.min.apply(null, B.velocity_reduction_db)) - 0.5;
  var pad = { l: 44, r: 12, t: 24, b: 30 };
  function xOf(f) { return pad.l + Math.log(f / fMin) / Math.log(fMax / fMin) * (chart.width - pad.l - pad.r); }
  function yOf(v) { return chart.height - pad.b - (v - yMin) / (yMax - yMin) * (chart.height - pad.t - pad.b); }
  function drawChart(index) {
    ctx.clearRect(0, 0, chart.width, chart.height);
    ctx.font = "12px system-ui, sans-serif"; ctx.fillStyle = Demo.INK2; ctx.strokeStyle = "#e1e0d9"; ctx.lineWidth = 1;
    for (var v = Math.ceil(yMin); v <= yMax; v += 2) {
      ctx.beginPath(); ctx.moveTo(pad.l, yOf(v)); ctx.lineTo(chart.width - pad.r, yOf(v)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(v + " dB", pad.l - 6, yOf(v) + 4);
    }
    [300, 400, 500, 600, 800, 1000].forEach(function (f) {
      if (f < fMin || f > fMax) return;
      ctx.textAlign = "center"; ctx.fillText(f + " Hz", xOf(f), chart.height - 10);
    });
    ctx.strokeStyle = "#c3c2b7";
    B.summary.forEach(function (row) {
      ctx.beginPath(); ctx.moveTo(xOf(row.short_circuit_hz), pad.t); ctx.lineTo(xOf(row.short_circuit_hz), chart.height - pad.b); ctx.stroke();
      ctx.textAlign = "left"; ctx.fillText("(" + row.target_mode.join(",") + ")", xOf(row.short_circuit_hz) + 3, pad.t + 12);
    });
    ctx.strokeStyle = Demo.SERIES[0]; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
    B.frequencies_hz.forEach(function (f, k) {
      var x = xOf(f), y = yOf(B.velocity_reduction_db[k]);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    var cx = xOf(B.frequencies_hz[index]), cy = yOf(B.velocity_reduction_db[index]);
    ctx.fillStyle = Demo.SERIES[0]; ctx.strokeStyle = "#fcfcfb"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }
  Demo.attachTooltip(chart, function (fx) {
    var x = fx * chart.width, best = 0;
    B.frequencies_hz.forEach(function (f, k) { if (Math.abs(xOf(f) - x) < Math.abs(xOf(B.frequencies_hz[best]) - x)) best = k; });
    return B.frequencies_hz[best].toFixed(0) + " Hz: " + B.velocity_reduction_db[best].toFixed(1) + " dB";
  });

  // ---- controls
  var modeBox = document.getElementById("modeButtons");
  B.summary.forEach(function (row) {
    var button = document.createElement("button");
    button.textContent = "mode (" + row.target_mode.join(",") + ")";
    button.style.marginRight = "6px";
    button.addEventListener("click", function () {
      var best = 0;
      B.frequencies_hz.forEach(function (f, k) { if (Math.abs(f - row.short_circuit_hz) < Math.abs(B.frequencies_hz[best] - row.short_circuit_hz)) best = k; });
      prepare(best);
    });
    modeBox.appendChild(button);
  });
  document.getElementById("tuning").textContent = B.summary.map(function (r) { return r.tuned_hz.toFixed(0) + " Hz"; }).join(", ");
  freqInput.addEventListener("input", function () { prepare(Number(freqInput.value)); });
  Demo.drawColorBar(document.getElementById("bar"), Demo.diverging, -1, 1, "skin displacement (same scale on both sides)",
                    [[-1, "inward"], [0, "0"], [1, "outward"]]);

  function resize() {
    views.forEach(function (view) {
      var w = view.container.clientWidth, h = view.container.clientHeight;
      view.renderer.setSize(w, h);
      view.camera.aspect = w / h;
      view.camera.updateProjectionMatrix();
    });
  }
  window.addEventListener("resize", resize);
  resize();
  var start = 0;
  B.frequencies_hz.forEach(function (f, k) { if (Math.abs(f - B.summary[0].short_circuit_hz) < Math.abs(B.frequencies_hz[start] - B.summary[0].short_circuit_hz)) start = k; });
  prepare(start);
  requestAnimationFrame(animate);
})();
