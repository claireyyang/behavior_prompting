# DrawAnything-Dots

A 2D drawing environment for studying **one task with several valid manners**, and the counterpart
to [DrawAnything-Sim](drawanything_sim.md).

## Why this exists alongside DrawAnything-Sim

The two environments optimise for opposite things, and the difference is not cosmetic.

**DrawAnything-Sim** maximises *task diversity*: thousands of unique target drawings, evaluated on
unseen shapes. Its `SimpleDrawEnv` variant goes further and makes manner provably invisible in the
goal image, by inking an edge set snapped to the target polyline. Two consequences follow, and both
are fatal for studying manner:

1. **Wrong-place drawing cannot be penalized.** Inked edges are always a subset of the goal edges,
   so `edge_iou = |inked| / |goal| = coverage` — the metric reduces algebraically to *recall*, with
   no precision term at all. A policy that scribbles across the canvas scores exactly the same as
   one that does nothing.
2. **One shape has one correct answer**, so there is no residual freedom for a manner to live in.

**DrawAnything-Dots** inverts both. The task — touch all the dots — is fully specified by the dot
set, leaving *how* you touch them free; and ink is an honest rasterization of wherever the pen
actually went, so drawing in the wrong place leaves a mark and costs precision.

### Terminology

There is **one task** with many **instances** (dot layouts and pen starts). This is not a multi-task
setup, and it matters for reading results: held-out instances test generalization across *layouts*,
not across tasks. Mechanically each instance occupies the replay buffer's `task_names` field,
because that is what the per-instance train/val split keys on.

## The environment

[`env/draw_dot/draw_dot_env.py`](../behavior_prompting/train_network/env/draw_dot/draw_dot_env.py)

- **Scene frame** is the unit square `[0,1]²`, so dots, pen position and action deltas share one
  unit system and canvas resolution is a pure rendering choice.
- **Observation**: `canvas` `(3, 96, 96)` — ch0 ink, ch1 dot discs, ch2 walls (all-zero in v1);
  `dots` `(4, 2)`; `pen_pose` `[x, y, z]`.
- **Action**: `[Δx, Δy, Δz]` — continuous **deltas**, applied to the current pose. Observations are
  absolute, actions relative. Deltas are stored natively in the dataset, so there is no
  absolute↔relative conversion layer anywhere.
- **Contact** at `z <= 0.5`. The pen state *after* a step governs the whole segment swept during it.
- **Executed-action buffer**: the last 8 executed `(Δx, Δy, Δz)`, kept on the env and deliberately
  **not** a policy input — see [Steering](#steering-phase-2).
- Renders in both `rgb_array` and `human` modes.

## The four manners

[`scripts/draw_dot/strategies.py`](../behavior_prompting/train_network/scripts/draw_dot/strategies.py)

| | path | contact |
| --- | --- | --- |
| `CONNECT` | straight line between consecutive dots | pen down throughout |
| `TOUCH` | **the same route as CONNECT** | down only on the dots, lifting between |
| `CURVE` | **bowed** arc between consecutive dots | pen down throughout |
| `PARALLEL` | short parallel dashes through each dot | down per dash, lifting between |

All four touch every dot, so the *task* is held constant while the manner varies. Two pairs carry
the structure the experiment depends on:

- **`CONNECT` vs `TOUCH` — separable only by contact.** They traverse the same waypoints, including
  the same dwell steps at each dot, so their xy streams are *bitwise identical* and only z differs.
- **`CONNECT` vs `CURVE` — separable by kinematics alone.** Both are one continuous pen-down stroke;
  only the path differs.

Both properties hold **by construction, for any dot layout**, which is why dots can be placed
uniformly at random and no instance is ever rejected:

- `CURVE` bows each segment perpendicular by `clip(0.3 · L, 0.06, 0.10)` rather than splining
  through the dots. A spline's deviation depends on how the dots happen to fall and vanishes as they
  approach collinear; a floored bow does not. The ceiling matters too — it keeps the arc provably
  inside the scene, which is what makes a perfect `CURVE` demo score `ink_precision` exactly 1.0.
- `PARALLEL`'s dashes are longer than a dot diameter, so it cannot collapse into `TOUCH`.
- Visit order is fixed left-to-right for all four. Order is not one of the manner axes, so
  randomizing it would inject uncontrolled multimodality on top of the manner variation.

## Metrics

[`utils/dot_strategy_metrics.py`](../behavior_prompting/train_network/utils/dot_strategy_metrics.py)

**Constraint satisfaction** is a precision/recall pair — the thing `SimpleDrawEnv` structurally could
not express:

- `dot_coverage` (recall) — fraction of dots touched pen-down. The headline success number, and what
  `checkpoint.topk.monitor_key` watches.
- `ink_precision` — fraction of drawn ink inside the allowed region, where *allowed* is the union of
  all four manners' ink templates, dilated. It cannot be "dot discs plus straight pairwise segments":
  `PARALLEL`'s dashes and `CURVE`'s arcs lie off those segments, so two perfectly valid manners would
  be scored as stray ink.

**Manner identification** classifies each rollout: `n_strokes` splits `{CONNECT, CURVE}` (one
continuous stroke) from `{TOUCH, PARALLEL}` (one per dot), and ink IoU against each template resolves
within the pair. Because the manners leave visibly different pictures, this works from the final
image alone.

> [!IMPORTANT]
> Metrics are read from **terminal env state**, never from reward aggregation. `DrawRunner` reduces
> per-step rewards with `np.max` over the episode; coverage is monotone so that is harmless, but
> `ink_precision` is not — under `max`, a policy that touched every dot and then scribbled would
> score identically to one that stopped cleanly.

## Generating data

```bash
cd behavior_prompting/train_network/scripts/draw_dot
python generate_dot_demos.py -o ../../datasets/draw_dot/dots_train.zarr \
  --num-instances 1000 --repeats-per-strategy 4
python generate_dot_demos.py -o ../../datasets/draw_dot/dots_eval.zarr \
  --num-instances 100 --repeats-per-strategy 4 --base-seed 10000
```

Every instance is demonstrated in all four manners, so manner varies *within* fixed conditioning.
Instance names embed the base seed (`dots_10000_00042`), which is what keeps train and eval instance
names disjoint. Held-out instances are consumed via `task.eval_dataset_path`, a rollout-only path —
there is no offline dataloader for it. The dataset is small: roughly 0.1 GB per million steps.

## Verification

Both run without a trained policy and should be run after any change to the env or the manners.

```bash
cd behavior_prompting/train_network
python scripts/draw_dot/verify_dot_env.py --n-instances 300
python scripts/draw_dot/audit_dataset.py -d datasets/draw_dot/dots_train.zarr
```

`verify_dot_env.py` checks env mechanics, the manner separability structure, and that a perfect demo
scores 1.0/1.0. **The separability check gates everything downstream** — if it fails, the dataset does
not encode what the experiment needs and no amount of training or steering recovers it.
`audit_dataset.py` reports manner balance, dot-placement spread, and delta-action sanity over a
generated dataset.

## Training

```bash
cd behavior_prompting/train_network
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=draw_dot_policy_dunet \
  task.dataset.dataset_path=datasets/draw_dot/dots_train.zarr \
  task.eval_dataset_path=datasets/draw_dot/dots_eval.zarr \
  exp_name="dots_v1" \
  training.seed=0
```

The defaults (96px canvas, ResNet34, batch 64) are sized to fit a single 24 GB card comfortably. To
scale out, prefix with `accelerate launch --gpu_ids 0,1,2,3 --num_processes=4`.

Two things differ from the DrawAnything-Sim configs, both because actions are deltas:

- The action normalizer is **symmetric with zero offset**, so a zero delta ("hold still") maps to
  exactly 0 rather than to some arbitrary point — a diffusion policy's prior is `N(0, I)`.
- Action padding is `zeros`, not `repeat_last`. Repeating a final *delta* commands the pen to keep
  drifting past the end of the episode.

The vision backbone is a ResNet rather than the `diffusion_unet` default CLIP ViT: the ViT is fixed
at 224×224 and asserts on anything else, and at ~86M parameters it is heavily oversized for a
near-binary 96×96 canvas.

## Evaluation

```bash
cd behavior_prompting/train_network
CUDA_VISIBLE_DEVICES=0 python train.py --config-name=draw_dot_policy_dunet \
  +rollout=draw_dot \
  'rollout.checkpoint_path="PATH/TO/epoch=0059-....ckpt"' \
  task.dataset.dataset_path=datasets/draw_dot/dots_train.zarr \
  task.eval_dataset_path=datasets/draw_dot/dots_eval.zarr
```

> [!TIP]
> Quote the checkpoint path. Topk filenames contain `=`, which Hydra's override grammar otherwise
> rejects with a bare `mismatched input '='`.

> [!TIP]
> This rollout path rebuilds the model from the command-line config rather than from the config
> stored in the checkpoint, so any model-shaping override used during training must be repeated here.

### The multimodality experiment

The task is fixed by the goal specification (the dot set). **Strategy is the residual multimodality
the policy exhibits under that fixed conditioning.** `p(τ | goal)` either has multiple modes or it
does not, and steering is only possible when it does — so the operational definition is just "the
variation the policy exhibits at fixed conditioning", which is directly measurable.

#### 1. The measurement run

```bash
cd behavior_prompting/train_network
CUDA_VISIBLE_DEVICES=0 python train.py --config-name=draw_dot_policy_dunet \
  +rollout=draw_dot \
  'rollout.checkpoint_path="PATH/TO/ckpt"' \
  task.dataset.dataset_path=null \
  task.eval_dataset_path=datasets/draw_dot/dots_eval.zarr \
  task.eval_rollout_max_tasks=25 \
  task.env_runner.n_test=32 \
  task.env_runner.fix_initial_state=true \
  exp_name=dots_v1_multimodality \
  logging.mode=offline
```

| override | why |
| -------- | --- |
| `+rollout=draw_dot` | swaps `_target_` to `RolloutPolicyWorkspace` — load a checkpoint and evaluate, no training |
| `task.dataset.dataset_path=null` | skip the *train*-split instances entirely; multimodality is only interesting on held-out layouts |
| `task.eval_dataset_path=...` | the held-out instances to roll out on (100 available in `dots_eval.zarr`) |
| `task.eval_rollout_max_tasks=25` | use 25 of those 100 instances |
| `task.env_runner.n_test=32` | rollouts per instance — **actually yields 40**, see below |
| `task.env_runner.fix_initial_state=true` | pin the pen start so sampling noise is the *only* thing varying |
| `logging.mode=offline` | `RolloutPolicyWorkspace` always initializes wandb; this keeps it local |

`fix_initial_state` is the load-bearing one. It pins the dot set **and the pen start** across all
envs of an instance, so the only thing varying is the policy's sampling noise. Without it the runner
randomizes the start per seed and the spread conflates the policy's own multimodality with its
sensitivity to initial conditions.

`n_test=32` yields **40** rollouts per instance, not 32: runners built from an eval dataset fold
`n_train` into `n_test` (`load_env.py`, "all environments are test environments for eval datasets"),
so the task config's `n_train: 8` is added on. Budget process count and memory accordingly — that is
40 async worker processes.

Note this run is shaped opposite to a success-rate run: **few instances, many rollouts each**. A
distribution over four modes needs samples per instance; mean `dot_coverage` needs breadth. Run them
separately rather than reading both off one job.

#### What each metric means

Every one is computed from the **terminal** state of a rollout — the final ink image plus the stroke
count — never aggregated over the episode. See `utils/dot_strategy_metrics.py`.

| metric | meaning |
| ------ | ------- |
| `dot_coverage` | **recall.** Fraction of the 4 dots touched *pen-down* (`z ≤ 0.5`, within `DOT_RADIUS`). The task-success number. |
| `ink_precision` | **precision.** Fraction of drawn ink inside the allowed region (union of the four manner templates, dilated). What makes drawing in the wrong place cost something. |
| `n_valid` / `valid_rate` | how many rollouts cleared `dot_coverage ≥ 0.9`. **Every manner metric below is computed only over these.** |
| `manner` | which of CONNECT / TOUCH / CURVE / PARALLEL the ink looks like. Stroke count is a hard prior — `1` → {CONNECT, CURVE}, `n_dots` → {TOUCH, PARALLEL}, anything else → all four — then template IoU resolves within the candidate set. |
| `manner_entropy` | entropy of the 4-way manner histogram **divided by `log 4`**, so it lands in [0, 1]. The headline multimodality number. |
| `contact_entropy` | the same on a 2-way collapse, divided by `log 2`: {CONNECT, CURVE} = *continuous* (never lift) vs {TOUCH, PARALLEL} = *lifting*. Localizes a partial collapse. |
| `manner_frac_<M>` | share of valid rollouts classified as manner `M`. |

**How to read it.** `eval/strategy/test/manner_entropy` is normalized against `log 4`:

- **≈ 0** — the policy collapsed onto one manner, and steering has nothing to select among. This is a
  real result about the policy class, not a bug, and it is what determines whether steering work is
  viable at all.
- **≈ 1** — the multimodality survived training.
- **in between** — `eval/strategy/test/contact_entropy` localizes what collapsed. A policy can easily
  preserve the `CONNECT`/`CURVE` path distinction while collapsing contact to always-drag.

> [!TIP]
> The `eval/` prefix is not optional: `env_rollout` dispatches eval-dataset runners with
> `aggregate_prefix='eval'`, so every key from this run is `eval/strategy/...`. A run driven from a
> *train* dataset emits the unprefixed `strategy/...` instead. Entropies are averaged over instances
> (per-instance first, then meaned), which is the right order — pooling rollouts across instances
> first would let across-instance variation masquerade as within-instance multimodality.

> [!WARNING]
> Every manner metric is gated on task success. A rollout that never touches the dots has a
> meaningless stroke structure, and scoring it anyway would report "no diversity" for what is really
> task failure. Read `eval/strategy/test/n_valid` and `dot_coverage` before reading any manner number.

#### 2. Watching rollouts

The batch job gives the number; `scripts/draw_dot/interactive_rollout.py` shows what the number is
made of. It steps one `DrawingDotEnv` synchronously in a live window, one episode at a time, on
held-out eval instances, and prints per-episode manner plus a running histogram as it accumulates.

```bash
cd behavior_prompting/train_network
python scripts/draw_dot/interactive_rollout.py \
  --checkpoint "runs/.../checkpoints/epoch=0040-eval_test_mean_score=0.924.ckpt" \
  --n-instances 5 --episodes-per-instance 8
```

Keys: `n` next instance · `r` replay this instance with fresh sampling noise · space pause · `q` quit.
`--no-window` runs it headless as a small printing-only eval; `--fps` controls playback speed;
`--instances NAME...` drills into specific layouts.

It reads its config from **inside the checkpoint** rather than from Hydra, so unlike the
`+rollout=draw_dot` path there is no set of model-shaping overrides to keep in sync. It also defaults
to `--fix-initial-state` (opposite of `config/task/draw_dot.yaml`, which is shaped for success-rate
runs) because manner is only meaningful at fixed conditioning; `--random-pen-start` flips it.

Its per-episode line carries what the batch aggregate throws away — the stroke count and all four
template IoUs — which is what makes it the tool for deciding whether a reported manner is a real
behavior or a classifier default:

```
dots_10000_00023 ep02  cov 1.000  prec 1.000  strokes  4  manner PARALLEL [CONNECT:0.15 TOUCH:0.10 CURVE:0.24 PARALLEL:0.65]
```

#### 3. Diagnostics on a collapsed instance

Two follow-ups worth running whenever an instance reports `manner_entropy == 0`. Both are cheap.

```bash
# (a) Is the collapse real, or is one template just winning by default?
python scripts/draw_dot/interactive_rollout.py --no-window \
  --instances dots_10000_00023 dots_10000_00007 dots_10000_00006 \
  --episodes-per-instance 5 --checkpoint "PATH/TO/ckpt"

# (b) Is it a property of the layout, or of the pinned initial condition?
python scripts/draw_dot/interactive_rollout.py --no-window --random-pen-start \
  --instances dots_10000_00023 dots_10000_00007 dots_10000_00006 \
  --episodes-per-instance 8 --checkpoint "PATH/TO/ckpt"
```

For (a), read `strokes` against the IoU spread. A stroke count equal to `n_dots` with the winning
template several times clear of the runner-up is a real behavior. A stroke count matching *neither*
1 nor `n_dots` puts all four manners in contention and lets IoU alone decide, which is where a
low-ink rollout can get binned into whichever template has the smallest footprint. For (b), entropy
returning under a resampled start means the collapse was about the initial condition rather than the
dot arrangement.

### Results — `dots_v1`, 2026-08-07

Checkpoint `epoch=0040-eval_test_mean_score=0.924.ckpt`, 25 held-out instances × 40 rollouts = 1000
rollouts at fixed conditioning. Run dir
`runs/2026.08.07/15.13.05_dots_v1_multimodality_rollout_policy_draw_dot_diffusion_unet`.

```
eval/mean_scores/test/all              0.909  (stderr 0.018)
eval/strategy/test/ink_precision       0.969
eval/strategy/test/manner_entropy      0.565        <- normalized against log 4
eval/strategy/test/contact_entropy     0.595
eval/strategy/test/contact_frac_continuous  0.451
manner fractions   CONNECT 0.190   TOUCH 0.130   CURVE 0.261   PARALLEL 0.420
mean valid_rate                        0.725
```

**Multimodality survived training, but unevenly.** The mean of 0.565 does not describe a typical
instance — the 25 per-instance entropies are bimodal, with almost no mass in the middle:

| | instances | mean H | mean coverage | mean valid_rate |
| --- | --- | --- | --- | --- |
| multimodal (3–4 modes) | 18 / 25 | 0.785 | 0.914 | 0.749 |
| collapsed (H = 0) | 7 / 25 | 0.000 | 0.895 | 0.663 |

10 of 25 instances retained all four manners; 17 retained at least three. The collapsed instances are
**not** the failures — their coverage is within noise of the multimodal ones. This is a per-layout
property, and it is the operative finding for steering: steering has a mode to select among on most
instances and nothing to select among on roughly a quarter of them.

**Every collapsed instance collapsed onto PARALLEL**, never a different manner. Five of the seven are
solidly evidenced (28–40 valid rollouts, all PARALLEL); `dots_10000_00020` and `_00009` had only 10
and 4 valid rollouts, so their `H = 0` is thin.

`contact_entropy` 0.595 with `contact_frac_continuous` 0.451 says the lift-vs-drag axis is close to a
coin flip, so contact is *not* the axis that collapsed.

#### What was ruled out

- **Not a classifier default.** Re-running the collapsed instances gives `strokes == 4` (= `n_dots`,
  so the candidate set is {TOUCH, PARALLEL}) with PARALLEL IoU 0.33–0.65 against TOUCH's 0.06–0.11 —
  a 4–6× margin — at precision 0.98–1.00. The policy really is lifting between dots and laying a
  short dash across each one.
- **Not a data imbalance.** `dots_train.zarr` is exactly balanced: 16000 episodes over 1000
  instances, 25.0% per strategy (4 demos per strategy per instance). `dots_eval.zarr` likewise.
- **Not explained by layout geometry.** Comparing the 7 collapsed against 9 multimodal instances on
  polyline length, longest/shortest segment, minimum pairwise dot separation and spatial spread shows
  no separation — every range overlaps (e.g. min pairwise separation 0.242 vs 0.223, polyline length
  1.176 vs 1.244). Nothing about the dot arrangement predicts which instances collapse.

#### Open questions

- **Why PARALLEL specifically?** It is the only manner whose ink does not depend on the dot
  arrangement — `dash_segments()` emits a fixed length at a fixed angle centred on each dot, while
  CONNECT / CURVE / TOUCH all require tracing a route determined by dot ordering and position. That
  makes PARALLEL the simplest function of the observation and the cheapest mode to reconstruct under
  BC's mode-averaging pressure, which would explain the *global* skew (0.420 vs TOUCH's 0.130) on top
  of any per-instance collapse. Untested.
- **Is manner decided in the first action chunk?** On a blank canvas nothing in the observation
  specifies the manner — that is the point. But once ink exists the canvas *does* disambiguate, so
  the policy reads its own committed manner back off ch0. If so, manner is drawn from sampling noise
  in the first ~8 steps and self-reinforced thereafter, and collapse means the first chunk reliably
  lands in the PARALLEL basin. Weak supporting hint: collapsed instances have the pen starting closer
  to the first dot (0.384 vs 0.520 mean distance), small n. Diagnostic (b) above tests this.
- **Is TOUCH under-detected?** Its IoU sits at 0.03–0.11 across all instances, collapsed or not — its
  template is pen-radius blobs, the smallest footprint of the four and structurally the hardest to
  match. It is still picked 13% of the time overall so it is not broken, but if TOUCH rollouts are
  being read as PARALLEL then part of the "collapse" is two lifting modes being merged.

## Walls (Phase 2)

Not implemented. Walls are line segments that ink cannot cross while `z <= 0.5`; lifting over
(`z > 0.5`) is legal, and a pen-down crossing is a violation counted separately from stray ink.
Channel 2 of `canvas` is already reserved for the wall mask and
[`layout.py`](../behavior_prompting/train_network/env/draw_dot/layout.py) already owns geometry
sampling.

What walls unlock: a wall between two dots forces a lift, so `CONNECT` becomes genuinely *infeasible*
on some layouts, and the question becomes whether steering finds the nearest feasible manner
(`TOUCH`, which shares CONNECT's route but lifts) or breaks the constraint. That requires a
**feasibility oracle** per (instance, manner) — deciding whether a demonstrated manner is achievable
at all and what the nearest achievable one is. That oracle, not the wall geometry, is the substantive
work.

## Steering (Phase 2)

Not implemented. The [multimodality results](#results--dots_v1-2026-08-07) are the prerequisite and
they come back **conditionally green**: `p(τ | goal)` retains 3–4 modes on 18 of 25 held-out
instances, so there is something to select among — but 7 instances pin to `PARALLEL` with zero
residual entropy, and any steering evaluation has to report those separately rather than averaging
them in. A controller cannot select a mode the policy does not have.

Steering is why the env keeps an executed-action buffer: motion axes are functions
of actions, so the buffer is what a steering controller reads. It is deliberately **not** a policy
input — giving the policy its own action history would let it infer the manner from its own past
rather than from the steering signal, which is the opposite of what the experiment measures.

The planned interactive collector is an **ungrounded** demonstration window: a mouse drag gives
`z < 0.5` (inking), plain movement gives `z >= 0.5` (pen up), and the recorded `[x, y, z]` path
converts to the same delta action stream the env consumes. "Ungrounded" is load-bearing — the
demonstration is a *manner*, not a solution to an instance. It has no dots and cannot be replayed or
scored; it exists only to be reduced to motion axes the steering controller conditions on. This is a
different collector from `scripts/draw/demo_draw.py`, which records grounded demos into a replay
buffer. The env supports `render_mode='human'` from the start specifically so this is not blocked.
