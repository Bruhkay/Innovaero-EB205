# cabinsim — EB205 InnovAero cabin noise model

Simulation software of team EB205 (Lufthansa Technik InnovAero 2026): how engine and
boundary-layer noise get through the sidewall of a Boeing 787-9 into a VIP cabin, and how much
each treatment (passive layers, shunted piezo patches, active trim panel) takes away.

The physics core is a plain Python package (`cabinsim`, numpy/scipy) that runs on its own,
from a notebook, or from the command line.

## Install

You need Python 3.11 or newer. All commands below are run from the repository folder.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,app]"
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,app]"
```

(`.[dev,app]` installs the test tools and the Streamlit app as well; the core alone is
`python -m pip install -e .`)

(In `cmd.exe` activate with `.venv\Scripts\activate.bat`. If PowerShell blocks the script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

**Google Colab**

```python
!pip install "git+<repository-url>"
from cabinsim.params import load_params
from cabinsim.pipeline import run
results = run(load_params(), fast=True)
```

A complete example (run, figures, listening, changing an input) is in `notebooks/colab_example.ipynb`.

## Run

After activating the virtual environment the commands are the same on every platform:

| command | what it does |
|---|---|
| `python -m cabinsim.run` | full model (~1 min), writes `out/results.json`, `out/spectra.csv`, `out/summary.csv` |
| `python -m cabinsim.run --fast` | coarser numerics, shorter ANC run |
| `python -m cabinsim.run --plots` | also save every figure as PNG in `out/figures/` |
| `python -m cabinsim.run --config-dir my_config` | run with your own copy of the YAML files |
| `python -m cabinsim.assumptions` | list every `[ASM]` value with its note (`--all` for DOC/LIT too, `--markdown out/assumptions.md` for a table) |
| `python -m streamlit run app/app.py` | team tool in the browser: edit inputs, run, compare solutions, listen, download results and config |
| `python -m cabinsim.demo_export` | refresh the jury demo: `demo/data/demo_data.js` and `demo/audio/*.wav` (uses `out/results.json`) |
| `python -m cabinsim.sensitivity` | one-at-a-time sensitivity of the results to uncertain inputs → `out/sensitivity.md` (~5 min) |
| `python -m cabinsim.coupons` | model predictions for the COMSOL coupon checks → `out/coupons/` (see `docs/validation.md`) |
| `python -m pytest` | run the tests |

What each model leaves out, how far the results can be trusted, and the COMSOL comparison plan are
in [`docs/validation.md`](docs/validation.md). A plain-language tour of the whole project (one
calculation followed from engine to ear, how to change things, likely jury questions) is in
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

## Jury demo (offline)

Open **`demo/index.html`** by double-clicking it. It needs no internet and no server: the data is a
`.js` file (browsers block `fetch()` on `file://`, but not `<script>`), sounds are plain `.wav` files
next to the pages, and the only library is a local copy (`demo/vendor/three.min.js`, Three.js r158,
MIT licence). `python -m http.server -d demo` and opening `http://localhost:8000` also works; it is
not required. Tested in Chrome; any current browser with WebGL should work.

| page | what it shows |
|---|---|
| `index.html` | overview, results table, links to everything below |
| `listen.html` | auralisation: the cabin sound at the head for each solution, instant A/B switching |
| `fdtd.html` | 2D cross-section, engine sound passing through the sidewall, baseline vs a chosen solution (real-time FDTD, qualitative) |
| `quiet_zone.html` | pressure around the head with ANC off/on and quiet-zone size vs frequency |
| `bay.html` | one skin bay vibrating, patches short-circuited vs RL-shunted |
| `heatmap.html` | 3D fuselage coloured by the exterior engine level (any mesh can be used, see `cabinsim/heatmap.py`) |

After changing the configuration, run `python -m cabinsim.run` and then `python -m cabinsim.demo_export`
to refresh the pages. The FDTD in the browser (`demo/js/fdtd_core.js`) is the same scheme as
`cabinsim/fdtd.py`; a test runs both and compares them (needs Node.js, skipped otherwise).

**About the sound files.** All files share one gain, so level differences between solutions are the
model's differences (synthesised 1/3-octave levels within ±0.6 dB of the model). The absolute level on
headphones cannot be guaranteed; set the volume once on the baseline and leave it. Use headphones:
laptop speakers do not reproduce the 50–125 Hz bands that dominate cabin noise.

## Configuration

Every number lives in `src/cabinsim/config/*.yaml` with its unit and a source tag:

- `DOC` — from the team archive (EB205.pdf)
- `LIT` — literature value (reference in the note)
- `ASM` — our assumption; must be checked

To change something, copy the `config` folder, edit it, and pass `--config-dir` (or edit in the
app and download the config zip; edited values come back tagged `ASM [changed in app]`). Solutions are in
`solutions.yaml`; each one `extends` the previous one and lists only what it changes, so new
variants (e.g. a V4) are a few lines.

## Package layout

```
src/cabinsim/
  params.py          YAML loader + source tags
  atmosphere.py      ISA air at cruise and in the cabin
  metrics.py         dB sums, A-weighting, SIL
  source.py          engine sound power, broadband + shaft-order tones
  geometry.py        fuselage panels, point-source projection (r, cos theta)
  materials.py       plates, Miki blanket model, piezo constants
  tbl.py             Goody spectrum, Corcos / Mellen wavenumber spectra
  sidewall/          archive closed forms, transfer matrices, finite-panel radiation, bridge path
  piezo_shunt.py     skin bay modal model with RL / negative-capacitance shunts
  cabin.py           reverberant + direct field at the passenger's head
  active_control/    trim-panel plant, diffuse primary field, FxLMS, simulation
  pipeline.py        runs the whole chain for every solution
  run.py             command line
  assumptions.py     assumption register
  export.py          JSON / CSV / PNG output
  plots.py           all figures (matplotlib), shared by CLI, app and notebooks
  config_writer.py   writes edited parameters back to tagged YAML
  fdtd.py            cross-section FDTD (material map + reference solver)
  heatmap.py         exterior levels on any triangle mesh
  demo_export.py     data bundle and sound files for the demo pages
  auralisation.py    cabin spectrum -> .wav (shaped noise + engine tones)
  sensitivity.py     one-at-a-time sensitivity study
  coupons.py         model predictions for the COMSOL coupons
  active_control/quiet_zone.py   steady-state quiet-zone maps
app/
  app.py             Streamlit page (tabs, downloads)
  controls.py        sidebar input widgets
tests/               pytest
```

## Dependencies

| package | why |
|---|---|
| numpy | arrays and linear algebra for every model |
| scipy | filters, Welch spectra, FFT convolution, grid interpolation |
| pyyaml | reads the configuration files |
| pytest (dev) | tests |
| matplotlib (`.[plots]`, `.[app]`, `.[dev]`) | all figures; one plotting library for CLI, app and notebooks |
| streamlit (`.[app]`) | the team tool in the browser (brings pandas for its tables) |
| Three.js r158 (bundled file, not a pip package) | 3D views in the demo pages, loaded locally so they work offline |
| Node.js (optional, not a pip package) | only for the test that compares the browser FDTD with the Python one |
