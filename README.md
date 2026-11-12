# VisionForge — a visual task factory

Generates synthetic images + questions + **answers computed from the scene spec** (no human or model
labelling), across difficulty knobs, so frontier multimodal models can be probed for where they break.

Every task follows five rules: computed answer · short exact-match answer · one difficulty knob `d ∈ [0,1]` ·
no surface shortcuts · easy for people. Ambiguous scenes are rejected in code (e.g. near-tangent circles, a
ball landing on a bin divider, a fruit only partly visible, a door width too close to the cart width).

## Quick start

```bash
pip install pillow numpy scipy matplotlib
python -m vf.samples --n 10 --out samples          # 10 samples per task, d spread 0 -> 1
python -m vf.samples --n 10 --group driving         # one group: puzzle | indoor3d | driving | real2d | real3d
```

Open `samples/index.html` for a gallery (question + answer under each image). Each task folder has
`items.jsonl` (question, answer, answer type, text twin, full scene spec) and `_sheet.png` (contact sheet).

No model API is called anywhere in sample generation. Model adapters live in `vf/models.py` and are only
constructed when explicitly requested.

## Verifying answers

```bash
pip install networkx
python -m vf.verify --samples samples
```

`vf/verify.py` re-derives every stored answer **without the generator's answer code**: from the image pixels
where possible (flood roads, valve states, mirror orientations, switch states, pegs, Gantt bars, crack length,
robot clearance), otherwise with a different algorithm on the stored scene (geometric gear meshing, forward
paper folding, Euler's formula for circle regions, pocket unfolding for billiards, networkx path search, ...),
plus structural checks for the 3D-only tasks (no colliding furniture, pedestrian actually visible). Run it after
any generator change; a mismatch means the image, the scene and the answer disagree.

Bugs this caught in the first build (all fixed): touching pegs, a crack running off the image, box supports
mis-computed by banker's rounding, a robot-path task that was always "yes", colliding furniture creating
unintended anomalies, pedestrians too hidden to see, and mirror manipulations too subtle for people.

## Task catalogue (56 tasks)

| Group | Tasks |
|---|---|
| `puzzle` (2D reasoning, Pillow) | hidden_outline, layer_shuffle, order_frames, missing_piece, circle_regions, layer_stack, gear_train, mirror_laser, fold_punch, string_pegs, one_stroke, ball_drop, shell_game (+ wire_trace, crowd_count, gauge, chart, twist, invoice in `vf/families`) |
| `indoor3d` (ray-traced rooms) | out_of_place, would_fall, two_view_change, robot_path, shadow_check, voxel_count |
| `driving` | obstacle_lane, hidden_pedestrian (3D); right_of_way, stop_in_time, lane_change, wrong_way (diagrams) |
| `real2d` | pcb_defect, crack_measure, cytology_count, tube_tip, caliper_measure, floorplan_path, pid_flow, circuit_leds, transit_route, aerial_change, flood_route, shelf_planogram, liquid_volume, cylinder_reading, billiards_bank, ui_state, spreadsheet_rank, gantt_overlap |
| `real3d` (ray-traced) | assembly_check, pick_order, fruit_count, offside_call, line_call, mirror_check; **cable_match** (Blender, see below) |

Each task records its real-world application, the capability it tests, its knob, and its **puzzle twin**
(the abstract counterpart, e.g. tube_tip ↔ wire_trace, pick_order ↔ layer_stack, billiards_bank ↔ mirror_laser).
Comparing realistic version vs puzzle twin vs text twin separates *realism*, *perception* and *reasoning* failures.

## Layout

```
vf/core.py        Family / Item, scoring, answer parsing, dataset I/O
vf/taskapi.py     @task decorator (one function per task), Reject, helpers
vf/rt.py          tiny NumPy ray tracer: boxes/spheres/cylinders, shadows, mirrors, object-id buffer
vf/tasks/*.py     task generators by group
vf/families/*.py  first six families (with counterfactual edits + pixel re-checks)
vf/runner.py      model evaluation with on-disk response cache
vf/adaptive.py    closed-loop breaking-point (d50) search + knob ablation
vf/models.py      oracle / random / simulated models + Anthropic, OpenAI-compatible, Gemini adapters
vf/samples.py     sample generator + gallery
```

## Adding a task

```python
@task("my_task", application="...", capability="...", knob="...", twin="...", group="real2d")
def my_task(rng, d):
    ...                       # build the scene from rng; raise Reject() if it is ambiguous
    return dict(image=img, question=q, answer=a, atype="int" | "float" | "choice" | "seq",
                choices=[...], tol=0.0, spec={...})
```

## cable_match: physically simulated, Blender-rendered cables

TRACE's task (arXiv 2609.29103: which connector is the other end of a cable?) with WireSeg-style physics.
`vf/ropesim.py` drops cables onto a desk with a position-based-dynamics rope simulation (inextensible, bending
stiffness, gravity, friction, cable-cable collision), so cables rest on each other at crossings.
`vf/blender/cables_scene.py` renders the scene in Blender via BlenderProc (GPU), with a mask per cable.

```bash
pip install blenderproc          # downloads Blender 4.2 on first run (~350 MB)
python -m vf.samples --n 10 --tasks cable_match
```

Knobs (stored per sample, each overridable): `n_cables` (2-5), `wiggle`, `mono` (all white, as in TRACE),
`stiffness`, `extra_ports`, `tilt`; measured: `crossings_on_target`, `min_cross_angle`, `target_covered_frac`.
Human-solvability rules enforced in code: no spot where 3+ cables meet, crossings on the target >= 40 degrees,
no side-by-side runs; harder settings require 1-3+ crossings on the target. The checker reads the rendered
cable mask to confirm which port the target cable reaches. Renders are cached in `vf_cache/` (not in git), so
on a fresh clone re-render before running `vf.verify` on this task.

## Upgrade path: photoreal indoor scenes

The 3D tasks currently use the built-in ray tracer (simple shapes). For photoreal rooms, keep the same scene
spec (object category, position, size, yaw → the answer), and swap the renderer for **Blender + Objaverse
assets scaled to each object's box**, which is how SceneCritic (arXiv 2604.13035) renders LayoutGPT /
Holodeck / LayoutVLM layouts. Layouts can also be taken from 3D-FRONT or generated by Holodeck/ProcTHOR.
Needs a Blender install and asset downloads (check licences per asset).
