// Used by tests/test_fdtd.py: node js_fdtd_runner.js <core.js> <model.json>
// Runs the browser FDTD core for model.steps steps with a tone source and prints the pressure field.
const fs = require("fs");
const core = require(process.argv[2]);
const model = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
model.index = Uint8Array.from(model.index);
const solver = core.createSolver(model);
for (let n = 0; n < model.steps; n++) {
  solver.step(model.amplitude * core.toneSource(solver.time, model.frequency));
}
process.stdout.write(JSON.stringify({ dt: solver.dt, p: Array.from(solver.p) }));
