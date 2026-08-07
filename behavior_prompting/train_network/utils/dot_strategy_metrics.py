"""
Scoring for `DrawingDotEnv` rollouts: did it do the task, and which manner did it use.

Two families, kept as separate numbers rather than blended into one score.

**Constraint satisfaction** is a precision/recall pair, which is the specific thing `SimpleDrawEnv`
could not express. There, inked edges were always a subset of the goal edges, so its IoU reduced
algebraically to recall and wrong-place drawing cost nothing. Here:

    dot_coverage   (recall)    -- fraction of dots touched pen-down
    ink_precision  (precision) -- fraction of drawn ink inside the allowed region

**Manner identification** asks which of the four valid ways the rollout used. Because the manners
leave visibly different ink, this works from the final image plus the stroke count, with no need to
reconstruct the trajectory.

⚠️ Every manner metric is gated on task success. A rollout that never touches the dots has a
meaningless stroke structure, and scoring it anyway reports "no diversity" for what is really task
failure. `manner_distribution` drops those and reports `n_valid`; read `dot_coverage` first.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from behavior_prompting.train_network.env.draw_dot.layout import (
    CONTINUOUS_MANNERS,
    MANNERS,
    DotLayout,
    ink_templates,
    mask_iou,
)

# A rollout must essentially finish the drawing before its manner means anything.
DEFAULT_COVERAGE_THRESHOLD = 0.9


@dataclass
class RolloutResult:
    """What one rollout achieved."""
    dot_coverage: float
    ink_precision: float
    n_strokes: int
    manner: Optional[str] = None
    manner_ious: Dict[str, float] = field(default_factory=dict)

    @property
    def contact_axis(self) -> Optional[str]:
        """'continuous' vs 'lifting' -- the 2-way sub-axis, reported separately."""
        if self.manner is None:
            return None
        return 'continuous' if self.manner in CONTINUOUS_MANNERS else 'lifting'

    def is_valid(self, coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD) -> bool:
        return self.dot_coverage >= coverage_threshold


def classify_manner(ink: np.ndarray, n_strokes: int, layout: DotLayout,
                    size: Optional[int] = None) -> (str, Dict[str, float]):
    """
    Which manner does this ink look like?

    Stroke count is used as a hard prior rather than a soft feature: it cleanly splits
    {CONNECT, CURVE} (one continuous stroke) from {TOUCH, PARALLEL} (one per dot), and IoU then
    resolves within the pair -- polyline vs bowed arc, blobs vs dashes. When the stroke count
    matches neither expectation the rollout is doing something unlike any demonstrated manner, so
    all four stay in contention and IoU alone decides.
    """
    size = size or ink.shape[-1]
    templates = ink_templates(layout, size)
    ious = {m: mask_iou(ink, t) for m, t in templates.items()}

    if n_strokes == 1:
        candidates = list(CONTINUOUS_MANNERS)
    elif n_strokes == layout.n_dots:
        candidates = [m for m in MANNERS if m not in CONTINUOUS_MANNERS]
    else:
        candidates = list(MANNERS)
    return max(candidates, key=lambda m: ious[m]), ious


def evaluate_state(ink: np.ndarray, dots_visited: np.ndarray, n_strokes: int,
                   layout: DotLayout, allowed: Optional[np.ndarray] = None) -> RolloutResult:
    """
    Score one rollout from its TERMINAL state.

    Terminal, not aggregated: `DrawRunner` reduces per-step rewards with `np.max` over the episode.
    Coverage is monotone so that would be harmless, but precision is not -- under `max`, a policy
    that touched every dot and then scribbled across the canvas would score identically to one that
    stopped cleanly.

    Takes raw arrays rather than an env so it also works from a vectorized runner, where the envs
    live in worker processes and only their attributes come back over the pipe.
    """
    ink = np.asarray(ink)
    size = ink.shape[-1]
    if allowed is None:
        from behavior_prompting.train_network.env.draw_dot.layout import allowed_ink_mask
        allowed = allowed_ink_mask(layout, size)

    total = int(ink.sum())
    precision = 1.0 if total == 0 else float(
        1.0 - (ink.astype(bool) & ~allowed.astype(bool)).sum() / total)
    coverage = float(np.asarray(dots_visited).mean()) if len(dots_visited) else 0.0

    manner, ious = classify_manner(ink, n_strokes, layout, size)
    return RolloutResult(dot_coverage=coverage, ink_precision=precision,
                         n_strokes=int(n_strokes), manner=manner, manner_ious=ious)


def evaluate_env(env, layout: Optional[DotLayout] = None) -> RolloutResult:
    """Convenience wrapper around `evaluate_state` for a local (non-vectorized) env."""
    layout = layout if layout is not None else env.layout
    return evaluate_state(env.ink, env.dots_visited, env.n_strokes(), layout, env.allowed)


def _normalized_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float('nan')
    k = len(counts)
    if k <= 1:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum() / math.log(k))


def manner_distribution(results: Sequence[RolloutResult],
                        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
                        ) -> Dict[str, float]:
    """
    The multimodality measurement: how does the policy's manner distribute at fixed conditioning?

    This is the operational definition of strategy -- the variation the policy exhibits when the
    goal and the initial state are held constant. `p(tau | goal)` either has multiple modes or it
    does not, and steering is only possible when it does.

    Read `manner_entropy` against `log 4`:
      ~0     -- collapsed onto one manner; steering has nothing to select among. A real result about
                the policy class, not a bug.
      ~1     -- multimodality survived training.
      between -- consult `contact_entropy` to localize which axis collapsed. A policy can easily
                preserve the CONNECT/CURVE path distinction while collapsing contact to always-drag.
    """
    out: Dict[str, float] = {}
    if not results:
        return out

    out['dot_coverage'] = float(np.mean([r.dot_coverage for r in results]))
    out['ink_precision'] = float(np.mean([r.ink_precision for r in results]))
    out['n_strokes'] = float(np.mean([r.n_strokes for r in results]))

    valid = [r for r in results if r.is_valid(coverage_threshold)]
    out['n_valid'] = float(len(valid))
    out['valid_rate'] = float(len(valid) / len(results))
    if not valid:
        return out

    counts = np.array([sum(r.manner == m for r in valid) for m in MANNERS], dtype=np.float64)
    out['manner_entropy'] = _normalized_entropy(counts)
    out['manner_n_distinct'] = float(int((counts > 0).sum()))
    for m, c in zip(MANNERS, counts):
        out[f'manner_frac_{m}'] = float(c / counts.sum())

    contact = np.array([
        sum(r.contact_axis == 'continuous' for r in valid),
        sum(r.contact_axis == 'lifting' for r in valid)], dtype=np.float64)
    out['contact_entropy'] = _normalized_entropy(contact)
    out['contact_frac_continuous'] = float(contact[0] / contact.sum())
    return out


def summarize(results: Sequence[RolloutResult],
              coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD) -> str:
    """One-line human summary, for logs and the audit script."""
    d = manner_distribution(results, coverage_threshold)
    if not d:
        return 'no rollouts'
    parts = [f"cov {d['dot_coverage']:.3f}", f"prec {d['ink_precision']:.3f}",
             f"valid {d['valid_rate']:.2f}"]
    if 'manner_entropy' in d:
        parts.append(f"H(manner) {d['manner_entropy']:.3f}")
        parts.append(f"modes {int(d['manner_n_distinct'])}/4")
        parts.append('[' + ' '.join(f"{m}:{d[f'manner_frac_{m}']:.2f}" for m in MANNERS) + ']')
    return '  '.join(parts)
