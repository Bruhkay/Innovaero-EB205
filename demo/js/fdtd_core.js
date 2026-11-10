/* 2D acoustic FDTD core - a line-by-line port of cabinsim/fdtd.py (coefficients + FdtdSolver.step).
 * Works in the browser (window.FdtdCore) and in Node (module.exports) so tests can compare it with Python.
 *
 * model = {nx, nz, dx, sponge_cells, source_cell: [i, j],
 *          materials: [{density, bulk_modulus, flow_resistivity, rigid}],
 *          index: Uint8Array of length nx*nz, cell (i, j) at index[i*nz + j]}
 * Grid: p at cell centres (nx*nz), vx on vertical faces ((nx+1)*nz), vz on horizontal faces (nx*(nz+1)).
 */
(function (root) {
  "use strict";
  var CFL_SAFETY = 0.5;

  function soundSpeed(m) { return m.rigid ? 0 : Math.sqrt(m.bulk_modulus / m.density); }

  function timeStep(model) {
    var cMax = 0;
    model.materials.forEach(function (m) { cMax = Math.max(cMax, soundSpeed(m)); });
    return CFL_SAFETY * model.dx / (cMax * Math.SQRT2);
  }

  // gamma = 10 c / (cells dx) * depth^2 inside the edge layer (same as sponge_profile in fdtd.py)
  function spongeProfile(model, speed) {
    var nx = model.nx, nz = model.nz, cells = model.sponge_cells;
    var gamma = new Float64Array(nx * nz);
    if (cells <= 0) return gamma;
    var alongZ = nz > 2 * cells;
    var gMax = 10 * speed / (cells * model.dx);
    for (var i = 0; i < nx; i++) {
      var dX = Math.max(cells - Math.min(i, nx - 1 - i), 0) / cells;
      for (var j = 0; j < nz; j++) {
        var dZ = alongZ ? Math.max(cells - Math.min(j, nz - 1 - j), 0) / cells : 0;
        var depth = Math.max(dX, dZ);
        gamma[i * nz + j] = gMax * depth * depth;
      }
    }
    return gamma;
  }

  function buildCoefficients(model, dt) {
    var nx = model.nx, nz = model.nz, n = nx * nz, dx = model.dx;
    var rho = new Float64Array(n), kappa = new Float64Array(n), sigma = new Float64Array(n), rigid = new Uint8Array(n);
    for (var c = 0; c < n; c++) {
      var m = model.materials[model.index[c]];
      rigid[c] = m.rigid ? 1 : 0;
      rho[c] = m.rigid ? 1.0 : m.density;
      kappa[c] = m.bulk_modulus;
      sigma[c] = m.flow_resistivity;
    }
    var outsideSpeed = soundSpeed(model.materials[0]) || 0;
    if (!outsideSpeed) model.materials.forEach(function (mm) { outsideSpeed = Math.max(outsideSpeed, soundSpeed(mm)); });
    var gamma = spongeProfile(model, outsideSpeed);

    function face(a, b) {                      // a, b: flat indices of the two cells beside a face
      if (rigid[a] || rigid[b]) return [0, 0];
      var r = 0.5 * (rho[a] + rho[b]);
      var s = 0.5 * (sigma[a] + sigma[b]);
      var g = 0.5 * (gamma[a] + gamma[b]);
      var damp = (s / r + g) * dt / 2;
      return [(1 - damp) / (1 + damp), dt / (r * dx) / (1 + damp)];
    }
    var vxKeep = new Float64Array((nx - 1) * nz), vxPush = new Float64Array((nx - 1) * nz);
    for (var i = 0; i < nx - 1; i++) for (var j = 0; j < nz; j++) {
      var f = face(i * nz + j, (i + 1) * nz + j);
      vxKeep[i * nz + j] = f[0]; vxPush[i * nz + j] = f[1];
    }
    var vzKeep = new Float64Array(nx * (nz - 1)), vzPush = new Float64Array(nx * (nz - 1));
    for (i = 0; i < nx; i++) for (j = 0; j < nz - 1; j++) {
      f = face(i * nz + j, i * nz + j + 1);
      vzKeep[i * (nz - 1) + j] = f[0]; vzPush[i * (nz - 1) + j] = f[1];
    }
    var pKeep = new Float64Array(n), pPush = new Float64Array(n);
    for (c = 0; c < n; c++) {
      var b = gamma[c] * dt / 2;
      pKeep[c] = rigid[c] ? 0 : (1 - b) / (1 + b);
      pPush[c] = rigid[c] ? 0 : kappa[c] * dt / dx / (1 + b);
    }
    return { vxKeep: vxKeep, vxPush: vxPush, vzKeep: vzKeep, vzPush: vzPush, pKeep: pKeep, pPush: pPush };
  }

  function createSolver(model) {
    var nx = model.nx, nz = model.nz;
    var dt = timeStep(model);
    var k = buildCoefficients(model, dt);
    var p = new Float64Array(nx * nz);
    var vx = new Float64Array((nx + 1) * nz);      // vx[i*nz + j], i = 0..nx (faces 0 and nx stay 0)
    var vz = new Float64Array(nx * (nz + 1));      // vz[i*(nz+1) + j], j = 0..nz
    var source = model.source_cell[0] * nz + model.source_cell[1];
    var solver = { p: p, dt: dt, time: 0, nx: nx, nz: nz };

    // One time step. sourceRate [Pa/s] is added to dp/dt at the source cell (dt-independent strength).
    solver.step = function (sourceRate) {
      var i, j, a, c;
      for (i = 0; i < nx - 1; i++) {
        for (j = 0; j < nz; j++) {
          a = i * nz + j; c = (i + 1) * nz + j;
          vx[c] = k.vxKeep[a] * vx[c] - k.vxPush[a] * (p[c] - p[a]);
        }
      }
      var nz1 = nz + 1;
      for (i = 0; i < nx; i++) {
        for (j = 0; j < nz - 1; j++) {
          a = i * nz + j;
          var face = i * nz1 + j + 1;
          var coef = i * (nz - 1) + j;
          vz[face] = k.vzKeep[coef] * vz[face] - k.vzPush[coef] * (p[a + 1] - p[a]);
        }
      }
      for (i = 0; i < nx; i++) {
        for (j = 0; j < nz; j++) {
          a = i * nz + j;
          var divergence = (vx[(i + 1) * nz + j] - vx[a]) + (vz[i * nz1 + j + 1] - vz[i * nz1 + j]);
          p[a] = k.pKeep[a] * p[a] - k.pPush[a] * divergence;
        }
      }
      p[source] += (sourceRate || 0) * dt;
      solver.time += dt;
    };

    solver.reset = function () {
      p.fill(0); vx.fill(0); vz.fill(0); solver.time = 0;
    };
    return solver;
  }

  // sin(2 pi f t) with a raised-cosine start over `rampPeriods` periods (same as tone_source in fdtd.py)
  function toneSource(time, frequency, rampPeriods) {
    var ramp = (rampPeriods === undefined ? 3 : rampPeriods) / frequency;
    var envelope = 0.5 * (1 - Math.cos(Math.PI * Math.min(time / ramp, 1)));
    return envelope * Math.sin(2 * Math.PI * frequency * time);
  }

  var api = { createSolver: createSolver, timeStep: timeStep, toneSource: toneSource, soundSpeed: soundSpeed };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FdtdCore = api;
})(this);
