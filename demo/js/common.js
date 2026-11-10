/* Shared helpers for the demo pages: colour scales, colour bars, base64, a small orbit control, tooltips. */
(function (root) {
  "use strict";

  // Sequential (one hue, light -> dark) and diverging (blue <-> grey <-> red) ramps.
  var SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"];
  var DIVERGING = ["#104281", "#256abf", "#5598e7", "#9ec5f4", "#f0efec", "#f4b9b5", "#ec7a74", "#e34948", "#a3282a"];
  var INK = "#0b0b0b", INK2 = "#52514e", SURFACE = "#fcfcfb";

  function hexToRgb(hex) {
    var v = parseInt(hex.slice(1), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }

  function makeScale(stops) {
    var rgb = stops.map(hexToRgb);
    var lut = new Uint8ClampedArray(256 * 3);
    for (var k = 0; k < 256; k++) {
      var t = k / 255 * (rgb.length - 1), i = Math.min(Math.floor(t), rgb.length - 2), f = t - i;
      for (var c = 0; c < 3; c++) lut[k * 3 + c] = rgb[i][c] + (rgb[i + 1][c] - rgb[i][c]) * f;
    }
    // value in [0, 1] -> [r, g, b]
    return function (t) {
      var k = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
      return [lut[k], lut[k + 1], lut[k + 2]];
    };
  }

  var sequential = makeScale(SEQUENTIAL);
  var diverging = makeScale(DIVERGING);

  // Horizontal colour bar with tick labels, drawn into a canvas.
  // ticks: numbers, or [value, text] pairs for custom labels.
  function drawColorBar(canvas, scale, min, max, label, ticks) {
    var ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height, barH = 12, pad = 8;
    ctx.clearRect(0, 0, w, h);
    for (var x = 0; x < w - 2 * pad; x++) {
      var c = scale(x / (w - 2 * pad - 1));
      ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      ctx.fillRect(pad + x, 18, 1, barH);
    }
    ctx.fillStyle = INK2; ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "left";
    ctx.fillText(label, pad, 12);
    (ticks || [min, (min + max) / 2, max]).forEach(function (tick, k, all) {
      var v = Array.isArray(tick) ? tick[0] : tick;
      var text = Array.isArray(tick) ? tick[1] : formatNumber(v);
      var px = pad + (v - min) / (max - min) * (w - 2 * pad);
      ctx.textAlign = k === 0 ? "left" : k === all.length - 1 ? "right" : "center";
      ctx.fillRect(px, 18 + barH, 1, 4);
      ctx.fillText(text, k === 0 ? pad : k === all.length - 1 ? w - pad : px, 18 + barH + 16);
    });
  }

  function formatNumber(v) {
    return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1);
  }

  function base64ToBytes(text) {
    var raw = atob(text), bytes = new Uint8Array(raw.length);
    for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return bytes;
  }

  // Minimal orbit control for a three.js camera: drag to rotate, wheel/pinch to zoom.
  function orbitControl(camera, element, target, onChange) {
    var state = { theta: 0.9, phi: 1.1, radius: camera.position.distanceTo(target), dragging: false, x: 0, y: 0 };
    function apply() {
      var s = Math.sin(state.phi);
      camera.position.set(target.x + state.radius * s * Math.cos(state.theta),
                          target.y + state.radius * Math.cos(state.phi),
                          target.z + state.radius * s * Math.sin(state.theta));
      camera.lookAt(target);
      if (onChange) onChange();
    }
    element.addEventListener("pointerdown", function (e) { state.dragging = true; state.x = e.clientX; state.y = e.clientY; element.setPointerCapture(e.pointerId); });
    element.addEventListener("pointerup", function () { state.dragging = false; });
    element.addEventListener("pointermove", function (e) {
      if (!state.dragging) return;
      state.theta += (e.clientX - state.x) * 0.008;
      state.phi = Math.max(0.15, Math.min(Math.PI - 0.15, state.phi - (e.clientY - state.y) * 0.008));
      state.x = e.clientX; state.y = e.clientY;
      apply();
    });
    element.addEventListener("wheel", function (e) {
      e.preventDefault();
      state.radius = Math.max(2, Math.min(400, state.radius * Math.exp(e.deltaY * 0.001)));
      apply();
    }, { passive: false });
    state.set = function (theta, phi, radius) { state.theta = theta; state.phi = phi; state.radius = radius; apply(); };
    apply();
    return state;
  }

  // Tooltip that follows the pointer over an element; `text(x, y)` returns the content or null.
  function attachTooltip(element, text) {
    var tip = document.createElement("div");
    tip.className = "tooltip";
    tip.hidden = true;
    document.body.appendChild(tip);
    element.addEventListener("pointermove", function (e) {
      var rect = element.getBoundingClientRect();
      var content = text((e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height);
      if (content === null) { tip.hidden = true; return; }
      tip.textContent = content;
      tip.hidden = false;
      tip.style.left = (e.pageX + 14) + "px";
      tip.style.top = (e.pageY + 14) + "px";
    });
    element.addEventListener("pointerleave", function () { tip.hidden = true; });
  }

  function requireData() {
    if (!root.CABINSIM_DATA) {
      document.body.insertAdjacentHTML("afterbegin",
        '<p class="missing">demo/data/demo_data.js not found. Run <code>python -m cabinsim.demo_export</code> first.</p>');
      return null;
    }
    return root.CABINSIM_DATA;
  }

  root.Demo = {
    sequential: sequential, diverging: diverging, makeScale: makeScale, hexToRgb: hexToRgb,
    drawColorBar: drawColorBar, base64ToBytes: base64ToBytes, orbitControl: orbitControl,
    attachTooltip: attachTooltip, requireData: requireData, formatNumber: formatNumber,
    INK: INK, INK2: INK2, SURFACE: SURFACE,
    SERIES: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
  };
})(this);
