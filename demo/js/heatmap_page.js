/* 3D heat map: fuselage mesh coloured by the exterior engine level per vertex.
 * cabinsim axes (x aft, y starboard, z up) map to three.js axes as (x, z, -y). */
(function () {
  "use strict";
  var data = Demo.requireData();
  if (!data) return;
  var H = data.heatmap;
  var container = document.getElementById("view");

  function toThree(x, y, z) { return new THREE.Vector3(x, z, -y); }

  var renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(renderer.domElement);
  var scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeef1f5);
  var camera = new THREE.PerspectiveCamera(35, 1, 0.5, 1000);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb8bcc4, 2.2));
  var sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(-20, 40, 30);
  scene.add(sun);

  // ---- fuselage with per-vertex colours
  var count = H.positions.length / 3;
  var positions = new Float32Array(H.positions.length);
  for (var v = 0; v < count; v++) {
    var p = toThree(H.positions[3 * v], H.positions[3 * v + 1], H.positions[3 * v + 2]);
    positions[3 * v] = p.x; positions[3 * v + 1] = p.y; positions[3 * v + 2] = p.z;
  }
  var geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(H.triangles);
  geometry.computeVertexNormals();
  var colors = new Float32Array(positions.length);
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  var fuselage = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ vertexColors: true }));
  scene.add(fuselage);

  // ---- context: nacelles, a sketched wing, rings at the cabin section ends (cosmetic, not in the calculation)
  var grey = new THREE.MeshLambertMaterial({ color: 0xb9bcc2 });
  H.engines.forEach(function (e) {
    var nacelle = new THREE.Mesh(new THREE.CylinderGeometry(H.fan_diameter / 2 + 0.25, H.fan_diameter / 2, 5, 32, 1, true), grey);
    nacelle.material.side = THREE.DoubleSide;
    nacelle.rotation.z = Math.PI / 2;
    nacelle.position.copy(toThree(e[0] + 2.0, e[1], e[2]));
    scene.add(nacelle);
    var span = Math.sign(e[1]);
    var wing = new THREE.BufferGeometry().setFromPoints([
      toThree(4, span * 2.6, -1.4), toThree(15, span * 2.6, -1.4), toThree(22, span * 30, 0.6), toThree(19.5, span * 30, 0.6)]);
    wing.setIndex([0, 1, 2, 0, 2, 3]);
    wing.computeVertexNormals();
    scene.add(new THREE.Mesh(wing, new THREE.MeshLambertMaterial({ color: 0xc9ccd2, side: THREE.DoubleSide })));
  });
  H.cabin_section_x.forEach(function (x) {
    var ring = [];
    for (var k = 0; k <= 64; k++) {
      var a = 2 * Math.PI * k / 64;
      ring.push(toThree(x, (H.radius + 0.03) * Math.cos(a), (H.radius + 0.03) * Math.sin(a)));
    }
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(ring), new THREE.LineBasicMaterial({ color: 0x0b0b0b })));
  });

  // ---- band selection and colouring
  var bandNames = Object.keys(H.levels);
  var current = "overall", range = [0, 1];
  var buttons = document.getElementById("bands");
  bandNames.forEach(function (name) {
    var b = document.createElement("button");
    b.textContent = name === "overall" ? "Overall" : (Number(name) >= 1000 ? Number(name) / 1000 + " kHz" : name + " Hz");
    b.addEventListener("click", function () { setBand(name); });
    b.dataset.band = name;
    buttons.appendChild(b);
  });

  function setBand(name) {
    current = name;
    var levels = H.levels[name];
    var lo = Infinity, hi = -Infinity;
    levels.forEach(function (l) { lo = Math.min(lo, l); hi = Math.max(hi, l); });
    lo = Math.floor(hi - 30);                                 // show the top 30 dB; quieter areas share the lightest colour
    range = [lo, Math.ceil(hi)];
    var color = new THREE.Color();
    for (var v = 0; v < count; v++) {
      var rgb = Demo.sequential((levels[v] - range[0]) / (range[1] - range[0]));
      color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace);
      colors[3 * v] = color.r; colors[3 * v + 1] = color.g; colors[3 * v + 2] = color.b;
    }
    geometry.attributes.color.needsUpdate = true;
    Array.prototype.forEach.call(buttons.children, function (b) { b.classList.toggle("active", b.dataset.band === name); });
    Demo.drawColorBar(document.getElementById("bar"), Demo.sequential, range[0], range[1], "Lp outside (dB re 20 µPa)");
    var cabinSum = 0, cabinCount = 0, maxIndex = 0;
    for (v = 0; v < count; v++) {
      var x = H.positions[3 * v];
      if (x >= H.cabin_section_x[0] && x <= H.cabin_section_x[1]) { cabinSum += Math.pow(10, levels[v] / 10); cabinCount++; }
      if (levels[v] > levels[maxIndex]) maxIndex = v;
    }
    document.getElementById("maxLevel").textContent = levels[maxIndex].toFixed(1) + " dB at x = " + H.positions[3 * maxIndex].toFixed(1) + " m";
    document.getElementById("cabinLevel").textContent = (10 * Math.log10(cabinSum / cabinCount)).toFixed(1) + " dB";
    render();
  }
  document.getElementById("lw").textContent = H.sound_power_level_db.toFixed(1) + " dB re 1 pW";

  // ---- camera, resize, hover
  function render() { renderer.render(scene, camera); }
  var orbit = Demo.orbitControl(camera, renderer.domElement, new THREE.Vector3(2, 0, 0), render);
  orbit.set(2.2, 1.15, 55);
  function resize() {
    var w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    render();
  }
  window.addEventListener("resize", resize);

  var raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
  Demo.attachTooltip(renderer.domElement, function (fx, fy) {
    pointer.set(fx * 2 - 1, -fy * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    var hit = raycaster.intersectObject(fuselage)[0];
    if (!hit) return null;
    var v = hit.face.a;
    return H.levels[current][v].toFixed(1) + " dB\nx = " + H.positions[3 * v].toFixed(1) + " m";
  });

  setBand(current);
  resize();
})();
