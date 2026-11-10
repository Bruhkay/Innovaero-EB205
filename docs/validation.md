# Validation and error sources

What each part of `cabinsim` leaves out, how far each result can be trusted, which inputs move the
answer most, and how the COMSOL coupon analyses will check the physics. Written for the 22 November
checkpoint item *"know where the errors come from"*.

All numbers below come from the default configuration (`src/cabinsim/config/*.yaml`) and can be
regenerated:

```
python -m cabinsim.run            # main results  -> out/results.json
python -m cabinsim.sensitivity    # section 4     -> out/sensitivity.md   (~5 min)
python -m cabinsim.coupons        # section 6     -> out/coupons/*.csv
python -m cabinsim.assumptions    # every [ASM] value with its note
python -m pytest                  # section 3
```

## 1. The calculation chain in one line each

| step | model | main simplification |
|---|---|---|
| Engine source | point monopole per engine, W = 138 W, log-parabola broadband + shaft-order tones | spectrum shape and tone share are assumptions; no installation or flight effects |
| Path to the fuselage | free-field spreading, `cos θ = |û·n̂|` per panel, fixed loss on shadowed panels | no diffraction, no wing reflection, no convection by the M0.85 flow |
| Boundary layer | Goody point spectrum × Mellen (or Corcos) wavenumber spectrum | τ_w and δ from the archive, unverified; one uniform TBL over the section |
| Sidewall | transfer matrices (skin, blanket, septum, air, trim) with curvature, cabin-pressure tension and a finite radiating panel | infinite flat layers locally; no windows, frames only through the bridge path |
| Bridge path | Sharp's point-connection formula, driven by the skin velocity | empirical; trim isolator gain is a guess |
| Piezo shunt | modal model of one bay, RL shunts per mode | one simply supported bay; its ΔTL is applied to every panel and path |
| Cabin | Sabine reverberant field + Lambertian direct field from every panel at the head | absorption is a guess; no ECS noise, no floor/ceiling/window paths |
| Active trim panel | modal trim panel, Rayleigh radiation, FxLMS in the time domain, diffuse remainder field | free field for the actuators; perfect secondary-path model |

## 2. What each model neglects, and what that does to the result

The direction column says whether leaving it out most likely makes our cabin **louder (+)** or
**quieter (−)** than reality, or changes mainly the **difference** between solutions (Δ).

### 2.1 Source and projection

| neglected | likely effect | comment |
|---|---|---|
| Fan size (2.8 m) vs distance (8–19 m) | ± at high frequency | At the BPF (690 Hz) λ = 0.43 m and D²/λ ≈ 18 m, so the panels are in the fan's near field. The point-source idea is better at *low* frequency, not "above 100 Hz" as the archive asks. |
| Flight effects (M0.85) | ± several dB on the engine path | Forward of the fan the sound travels against the flow; convection, refraction in the shear layer and the boundary layer are all missing. This is the biggest unmodelled physics on the engine path — but the engine path is not what sets the cabin level (section 4). |
| Shielding by wing and nacelle, diffraction round the fuselage | ± | Replaced by a constant 15 dB shadow loss (sensitivity in section 4). |
| Spectrum shape, tone share, N1 | ± | All [ASM]. The tones at 115/230 Hz carry most of the tonal energy reaching the cabin. |
| Only cruise | — | Climb, descent and ground (no pressurisation, different N1) are not computed. |

### 2.2 Turbulent boundary layer

| neglected | likely effect | comment |
|---|---|---|
| Source of τ_w = 15.5 Pa, δ = 0.21 m | ± | Plausible for x ≈ 25 m from the nose (c_f ≈ 1.3·10⁻³, flat-plate δ ≈ 0.2 m) but not from a CFD or flight test. |
| Wavenumber model | ± 3 dB on SIL | Corcos over-predicts the low-wavenumber tail; Mellen is lower by about 8 dB there. We report both (`head_tbl_alternative` in results.json). |
| TBL varies along the fuselage, pressure gradients, windows, steps | ± | One uniform TBL is used for the whole section. |
| "TBL transmits 20–30 dB less than sound" | — | No longer an input. The model computes this coupling penalty: 9–23 dB over most bands but **about 0 dB at 400–800 Hz**, where the pressurised skin's free waves travel as fast as the convected TBL pressure (aerodynamic coincidence). A flat 20 dB would under-predict the cabin there. |

### 2.3 Sidewall transmission

| neglected | likely effect | comment |
|---|---|---|
| **Windows** | + (real cabin louder) | A 787 window seat sits next to a large, thin, double-pane window — probably the strongest flanking path at mid and high frequency. Not modelled at all. |
| Frames and stringers as stiffeners | ± | They enter only through the bridge path (Sharp). Bay modes are not in the TMM, only in the shunt model. |
| Curvature and pressurisation | included, approximately | Donnell curvature term and membrane tension N = Δp·R in the plate impedance. Pressurisation moves the first bay mode from 337 Hz (archive formula) to **542 Hz**. |
| Finite panel size | included, approximately | Spatial windowing with one radiating trim panel size; changes the lowest-band TL by several dB but SIL and dB(A) by at most 1 dB (section 4). |
| Blanket model | ± at low frequency | Delany–Bazley–Miki is fitted for 0.01 < f/σ < 1; at 50–100 Hz we are below that range. Limp-frame, no compression of the blanket. |
| Leaks, seals, gaps at the trim panel edges | + | Not modelled. |
| Two-medium critical angle | included | With c_outside = 296 m/s < c_cabin = 344 m/s an infinite panel transmits nothing above 59.5°. The finite-panel radiation removes this artefact (33 % of the incident power arrives above that angle). |

### 2.4 Bridge (structure-borne) path

| neglected | likely effect | comment |
|---|---|---|
| Real attachment dynamics | ± large at high frequency | The bridge path carries more power than the airborne path above 1.6 kHz for the baseline wall and above 630–800 Hz for the treated walls (V1–V3). There the cabin level follows the [ASM] isolator gain (15 dB) and spacing (0.46 m) almost dB for dB. This is the first thing to measure. |

### 2.5 Shunted piezo patches

| neglected | likely effect | comment |
|---|---|---|
| Real bay edge conditions | ± frequency shift | Simply supported assumed; clamped edges raise the modes. The shunt tuning must follow the real (cruise) frequencies. |
| Patch stiffness, bonding layer | − (gain over-estimated) | Only the patch mass is included. |
| Same ΔTL for every angle, panel and path | ± | The bay model is solved for a diffuse field and the band gain is applied everywhere. |
| Circuit | − | Ideal R and L. The required inductance is 0.1–0.2 H, which in practice needs a synthetic (op-amp) inductor and power; capacitance drifts with temperature and detunes the shunt. |
| Ground vs cruise tuning | — | Modelled (`piezo.tuning_condition`). Tuning on the ground (unpressurised) misses the cruise modes by about 180 Hz (mode (1,1): 318 Hz on the ground vs 498 Hz in cruise). |

### 2.6 Cabin

| neglected | likely effect | comment |
|---|---|---|
| ECS (air conditioning) noise | + at mid/high frequency | Real cabins have an air-system floor that often sets the SIL. Not included, so our SIL is optimistic. |
| Absorption values | about ± 1.5 dB | Furnished-cabin guesses; ×0.7…×1.3 moves the head level by −1.0…+1.4 dB (section 4). |
| Floor, ceiling, door and window paths | + | Only the sidewall above the floor radiates. |
| One passenger position | — | Head at the window seat on the engine side; other seats are quieter in the direct field. |

### 2.7 Active trim panel

| neglected | likely effect | comment |
|---|---|---|
| Cabin reflections of the actuator sound | ± | The secondary path is free field; real cabins add reflections. |
| Perfect secondary-path model | − (gain over-estimated) | FxLMS uses the exact plant; real identification errors reduce the gain and margin. |
| DSP latency, sensor noise, actuator limits | − | Not modelled; the predicted drive is about 11 V rms, well below the ~140 V limit of a 0.25 mm patch. |
| Convergence | − | The tonal controller is stable only below μ ≈ 0.003 (the trim panel delays the secondary path by ~0.09 s) and does not fully converge within the 8 s simulation; head-zone values may still move by a few dB. |
| Broadband reference coherence | − | Taken as 0.9 for the controlled panel only. The rest of the field is incoherent, which is why broadband ANC is ineffective (and slightly harmful) in the model. |

## 3. What has been checked

| check | result | where |
|---|---|---|
| TMM single plate, normal incidence = archive TL₀ formula | difference < 10⁻⁶ dB | `tests/test_sidewall.py` |
| Mass-law slope | 6.0 dB/octave | `tests/test_sidewall.py` |
| f₁₁ from the archive formula | 337.4 Hz vs 335 Hz quoted (0.7 %) | `tests/test_sidewall.py` |
| L_w for 138 W | 141.4 dB | `tests/test_source_geometry.py` |
| Finite-panel radiation: large panel → 1/cos θ, small panel → k²A/2π | within 2 % | `tests/test_sidewall.py` |
| TBL wavenumber integral reproduces the diffuse-field TL (all factors of 2 and π) | within 0.1 dB | `tests/test_sidewall.py` |
| Rayleigh modal radiation efficiency → 1 at high frequency | 1.0 ± 0.05 | `tests/test_piezo_shunt.py` |
| Shunt short/open resonances vs analytic | exact / within 0.3 % | `tests/test_piezo_shunt.py` |
| FxLMS stable and converging; steady-state formula = time simulation | > 20 dB reduction; within 0.5 dB | `tests/test_fxlms.py`, `tests/test_visual_data.py` |
| FDTD wave speed in free field | within 2 % of c | `tests/test_fdtd.py` |
| FDTD limp plate vs TL₀ | within 0.2 dB (0.02 dB measured) | `tests/test_fdtd.py` |
| Browser FDTD = Python FDTD | identical to 10⁻⁹ | `tests/test_fdtd.py` |
| Auralisation band levels vs model | within 0.6 dB | `tests/test_auralisation.py` |
| Numerical convergence | TBL integral: 800 vs 2500 wavenumbers < 0.02 dB; diffuse TL: 24 vs 181 angles < 0.1 dB; bridge calibration needs 181 angles (91 vs 181 < 1 dB) | development runs, settings in `numerics.yaml` |

Comparison with the archive's own numbers: the model gives **79.5 dB(A)** at the head for the
baseline, inside the archive's typical-cabin range of 78–82 dB(A). This was not tuned; it is the
first-principles result with the default inputs. Nothing has yet been compared with measured
787 cabin data.

## 4. Which inputs matter most

One input at a time, each moved to the two ends of a plausible range; everything else at default.
Fast numerics, baseline (V0) and passive (V1) solutions. Sorted by effect on the baseline level.

Reference (fast numerics): baseline head SIL 56.0 dB, 79.4 dB(A); passive solution gains 3.0 dB SIL, 5.8 dB(A).

| input | low / high setting | Δ baseline SIL | Δ baseline dB(A) | Δ passive gain, SIL | Δ passive gain, dB(A) |
|---|---|---|---|---|---|
| Bridge isolator gain | 10 dB / 20 dB | +3.7 / -3.0 | +1.3 / -0.5 | -1.1 / +1.5 | -1.8 / +1.2 |
| TBL convection speed U_c/U | 0.6 / 0.8 | -2.8 / +3.7 | -2.1 / +3.8 | +0.5 / -0.3 | +0.7 / -0.3 |
| Skin thickness | 3 mm / 5 mm | +3.5 / -1.1 | +1.5 / -1.4 | -0.1 / +0.8 | -0.2 / +0.4 |
| TBL wall shear stress | -30 % / +30 % | -2.4 / +2.0 | -1.7 / +1.6 | +0.2 / -0.1 | +0.5 / -0.3 |
| Bridge attachment spacing | 0.30 m / 0.60 m | +2.7 / -1.5 | +0.9 / -0.3 | -0.8 / +0.6 | -1.3 / +0.7 |
| Skin tension from pressure | off / on (default) | +4.1 / +0.0 | +3.8 / +0.0 | +0.1 / +0.0 | -1.0 / +0.0 |
| TBL wavenumber model | Mellen / Corcos | +0.0 / +3.3 | +0.0 / +1.8 | +0.0 / +0.4 | +0.0 / +0.3 |
| CFRP Young's modulus | 45 GPa / 70 GPa | +2.4 / -0.1 | +2.4 / -0.5 | -0.1 / +0.7 | -0.3 / +0.8 |
| Cabin absorption | x0.7 / x1.3 | +1.3 / -0.9 | +1.4 / -1.0 | -0.0 / +0.0 | -0.0 / +0.0 |
| Engine acoustic power | -3 dB / +3 dB | -0.4 / +0.6 | -0.9 / +1.4 | -0.1 / +0.2 | -0.3 / +0.4 |
| Blanket flow resistivity | x0.5 / x2 | +1.0 / -0.9 | +0.7 / -1.3 | +0.7 / -0.5 | -0.1 / +0.3 |
| Radiating trim panel size | 0.6 x 0.5 m / 1.6 x 1.2 m | +0.6 / -0.1 | +1.0 / +0.1 | +0.4 / -0.1 | +0.7 / +0.0 |
| Engine tone fraction | 0.15 / 0.45 | +0.1 / -0.1 | +0.2 / -0.2 | +0.0 / -0.1 | +0.1 / -0.2 |
| Shadow loss | 8 dB / 25 dB | +0.0 / -0.0 | +0.2 / -0.0 | +0.0 / -0.0 | +0.0 / -0.0 |
| Head distance from trim | 0.30 m / 0.60 m | +0.1 / -0.1 | +0.0 / -0.0 | +0.0 / -0.0 | -0.0 / +0.0 |
| Engine spanwise position | -1 m / +1 m | -0.0 / -0.0 | +0.1 / -0.1 | +0.0 / +0.0 | +0.2 / -0.2 |

What stands out:

- **The boundary layer and the bridge path set the cabin level, not the engine.** Halving or doubling
  the engine power (±3 dB) moves the baseline by at most 1.4 dB(A) and 0.6 dB SIL. The engine's
  141.4 dB source level, its spectrum and its position are therefore *not* the critical unknowns.
- The largest single effects are the trim **isolator gain** (bridge path, −3…+4 dB SIL), the **TBL
  convection speed** and **wall shear stress** (±2–4 dB), the **skin thickness / stiffness**, and the
  **TBL wavenumber model** (Corcos +3.3 dB SIL over Mellen).
- **Cabin pressure matters**: leaving the skin tension out raises the baseline by 4 dB, because the
  unpressurised skin's aerodynamic coincidence moves into the SIL bands.
- **The benefit of the passive treatment is robust**: across all 32 runs it moves by at most 1.8 dB,
  while the absolute level moves by up to 4 dB.

How to read it: the left two Δ columns say how wrong the **absolute** level could be because of
that input; the right two say how wrong the **benefit of the passive treatment** could be. For most
inputs the benefit moves much less than the absolute level, which is why the summary keeps saying
that differences between solutions are more reliable than absolute numbers.

## 5. How far each result can be trusted

| result | confidence | why |
|---|---|---|
| Closed-form frequencies (c_L, f_r, f_c, f₁₁, f₀) | high | textbook formulas, tested; only the material inputs are uncertain |
| Pressurisation raises bay modes (337 → ~540 Hz) | medium–high | standard prestressed-plate physics; the share of hoop load carried by the skin is assumed |
| TBL coupling penalty vs frequency (shape) | medium | physics is standard; level depends on the TBL model (± 3 dB) |
| Benefit of the passive treatment (V1 − V0) | medium | 3–4 dB SIL, ~6 dB(A); robust across most inputs (section 4) |
| Benefit of the piezo shunt | medium in the bay, low in the cabin | 5–7 dB less bay vibration at the first two tuned modes, ~0 at the third; in the cabin it is cancelled by the patches' added mass shifting the aerodynamic coincidence |
| Tonal ANC around the head | medium | 7–12 dB at 115/230 Hz, ~2 dB at 690 Hz; free-field secondary path, not fully converged |
| Broadband ANC | low (but clearly small) | limited by the incoherent diffuse field; the conclusion "little or no benefit" is robust |
| Absolute head SIL and dB(A) | low | no ECS noise, no windows, assumed source and isolators; ± 5 dB or more |
| Whether the VIP target (48–52 dB SIL) is met | low | depends on the absolute level; the model says none of V0–V3 reaches it |

## 6. COMSOL coupon plan

The archive (10 August notes) proposes coupon-size COMSOL/ANSYS models because of node limits.
Each coupon below isolates one part of `cabinsim` with the same assumptions, so a disagreement
points at one place. `python -m cabinsim.coupons` writes our prediction for each as CSV and every
input value in `out/coupons/coupons_spec.json`. Mesh rule of thumb: at least 6 quadratic elements
per wavelength in air (≈ 3 cm at 2 kHz) and per bending wavelength in the plates.

### C1 — Bare skin bay (modes and TL)

- **Geometry**: CFRP plate 0.533 m × 0.200 m × 4 mm (E = 60 GPa, ρ = 1550 kg/m³, ν = 0.30, η = 0.01).
  Flat first; then curved (R = 2.885 m, curved along the 0.2 m side).
- **Boundary conditions**: simply supported edges (pinned out of plane, free in plane). Repeat with
  clamped edges to see how much the BC matters.
- **Studies**: (a) eigenfrequencies; (b) prestressed eigenfrequencies with in-plane tension
  N_hoop = 165 kN/m, N_axial = 83 kN/m (cabin Δp = 57 kPa); (c) TL for a normally incident plane wave,
  plate in a rigid baffle between two air half-spaces (PML), outside air ρ = 0.38, c = 296.5; cabin air
  ρ = 0.96, c = 344.4.
- **Compare with**: `C1_bay_modes.csv` (flat / curved / curved + pressurised) and `C1_bay_tl_normal.csv`.
- **Accept**: modes within 5 %; TL within 3 dB away from resonances.

### C2 — Bay with one shunted PZT patch

- **Geometry**: C1 flat plate plus one PZT-5A patch 0.16 × 0.10 × 0.25 mm at the centre.
- **Physics**: solid mechanics + piezoelectric + electrical circuit (patch electrodes to a series R–L).
- **Studies**: eigenfrequencies with the electrodes shorted and open → effective coupling
  k² = (f_open² − f_short²)/f_short²; then TL (normal incidence) shorted vs RL (values in the spec file).
- **Compare with**: `C2_patch_frequencies.csv`, `C2_patch_tl_normal.csv`.
- **Accept**: k² within 30 % (our model ignores patch stiffness and bond layer); TL gain at the tuned
  mode within 2 dB.

### C3 — Layered sidewall, infinite (unit cell)

- **Geometry**: one small unit cell (e.g. 5 × 5 cm) of the V0 and V1 build-ups: skin (shell),
  blanket (poroacoustics, Delany–Bazley–Miki with σ from the spec), septum (V1, limp mass), air gap,
  trim (shell with the given m'' and D).
- **Boundary conditions**: Floquet periodicity on the sides (this is exactly the infinite-layer
  assumption of the TMM), port/PML on both ends.
- **Studies**: TL at 0°, 30° and 60° incidence, 50–2000 Hz.
- **Compare with**: `C3_layered_wall_tl.csv`. Above 59.5° COMSOL should also show (almost) no
  transmission, because the cabin air is faster than the outside air.
- **Accept**: within 1–2 dB (same physics; differences mean an input or a model error).

### C4 — Active trim panel, secondary path

- **Geometry**: trim panel 1.066 × 0.80 m (m'' = 1.65 kg/m², D = 400 N·m, η = 0.03) in a rigid baffle,
  one 0.10 m PZT patch at (0.3, 0.3) of the panel, air on the cabin side, point at the head centre
  (0.53, 0.44, 0.45 m in panel coordinates).
- **Studies**: frequency response, 1 V on the patch → pressure at the head point; then a
  pressure load on the back of the panel → pressure at the head point.
- **Compare with**: `C4_active_panel_frf.csv` (magnitude and phase).
- **Accept**: magnitude within 3 dB below 500 Hz. Then repeat inside a small box "cabin" with
  absorbing walls to see how much the reflections (which our model ignores) change the path.

### Order of work

C3 first (cheapest, should agree almost exactly and checks the material inputs), then C1 (bay modes
and the pressurisation effect, which drives the shunt tuning), then C2, then C4.

## 7. Answers to the archive's question and verification lists (pp. 7–8)

| archive item | what the model says now |
|---|---|
| Source of the 141.4 dB engine | Still only `W = η F U∞ = 138 W`; η and F are not documented. But ± 3 dB on W moves the cabin head level by at most 1.4 dB(A) and 0.6 dB SIL (section 4): at cruise the boundary layer dominates, so this value is less critical than it looked. |
| Is the standard impedance really 400? | 400 Pa·s/m is not air's impedance; it is p₀²/I₀ = (20 µPa)²/(10⁻¹² W/m²), which makes L_p and L_I line up. Air at sea level is about 413; at 35 000 ft it is 113, so the term is −5.5 dB. |
| a = 2.885 m based on what? | Half of the 787 fuselage external width, 5.77 m. The real section is slightly taller than wide. |
| ISA Table 1 values | Now computed from the ISA formulas: 10 668 m → T = 218.8 K, p = 23.84 kPa, ρ = 0.380 kg/m³, c = 296.5 m/s. |
| ISO 1683 reference 20 µPa | Yes, p₀ = 20 µPa in air; W₀ = 1 pW. |
| Frame 0.533 m and stringer 0.2 m based on what? | No public 787 value found; marked [ASM]. They set the bay modes and the shunt tuning; the CFRP modulus and thickness have similar leverage (section 4). |
| d = 0.125 m, a 12.5 cm wall? | Used as the skin-to-trim depth. It sets the mass–air–mass resonance (138 Hz, next to the 115 Hz tone). |
| TBL τ_w = 15.5 Pa, δ = 0.21 m: what is this? | Wall shear stress and boundary-layer thickness; they set the level (Goody) and the spread over wavenumbers of the wall pressure. Plausible, still unverified. |
| Is TBL really transmitted 20–30 dB less? | Not across the board: the model gives 9–23 dB, and about 0 dB at 400–800 Hz (aerodynamic coincidence of the pressurised skin). |
| Convection at 0.7 U? | TBL pressure patterns are carried downstream at U_c ≈ 0.6–0.8 U, so their wavenumber is ω/U_c, far above the acoustic one. That mismatch is why the TBL couples weakly — except near coincidence. |
| Why is the point source valid above 100 Hz? | It is the other way round: valid at low frequency, questionable near the BPF (see 2.1). |
| Why does the anechoic assumption under-predict? | Without walls that reflect, only the direct sound is left, and absorption has nothing to act on. In our model the reverberant field is the larger part: at the window seat the head is only 2–4 dB above the reverberant room average (2.3 dB(A), 4 dB SIL), so dropping the reverberant part would lose most of the level. |
| The room is a box but the walls are curved | The cabin model uses the real circular segment above the floor (surface and volume). |
| Sound cannot reach the floor directly; why include it? | The floor does not radiate in the model; only the sidewall and crown above it do. |
| V3: ANC or not? | Now defined: V3 = V2 + active trim panel (PZT actuators on the trim panel, error microphones at the head, FxLMS). Seat-headrest loudspeaker ANC is not part of it. |
| Is the VIP cabin really 48–52 dB SIL? | Target kept [DOC, to verify]. Model: baseline 56 dB, passive 53 dB at the head, so not met without further measures. |
| Is the cabin really 78–82 dB(A)? | The baseline model gives 79.5 dB(A) at the head, inside that range (not tuned). |
| Does a composite aircraft need more treatment than aluminium? | Not computed yet. Swap `CFRP_QI` for `Al2024` in `solutions.yaml` to compare; mind the different coincidence frequency and damping. |

## 8. What to do before the final

In order of how much they can change the answer (section 4):

1. **Boundary layer inputs**: get τ_w, δ and U_c for the cabin station from a CFD run or a published
   cruise TBL study, and decide between the Mellen and Corcos spectra with a literature comparison.
2. **Trim isolators and attachment spacing**: look up or measure the isolator stiffness; it sets the
   bridge path and the high-frequency level.
3. **Skin build-up**: actual 787 skin thickness and laminate modulus at the cabin station (also moves
   the bay modes and therefore the shunt tuning).
4. **One measured or published cruise cabin spectrum** (787 or a similar twin-aisle) to put a number
   on the absolute-level error, band by band.
5. Add the **window** as a parallel path and an **ECS noise floor**; both are missing and both make
   the real cabin louder.
6. Run the four **COMSOL coupons** (section 6); update `config/*.yaml` with what they correct and re-run.
