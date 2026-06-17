# Decision Memo Template: Compassion Modulation Week

## One-Line Decision
- `continue with linear/composed controls`
- or `move to targeted SAE work on late-think layers`

## Executive Result
- Did we find a compassion modulation that:
  - softens sharp-source outputs,
  - preserves source identity,
  - preserves competence,
  - and respects social gating?

## Pre-Registered Gates

### Day 2 gate
- `compassion` on `think_mean @ L39`:
  - held-out `R²`:
  - held-out `pearson`:
  - top/bottom quartile AUC:
- Gate passed:
  - `yes / no`

### Day 3 gate
- Best composed direction:
- Mean reverse-pair `target_prob` lift:
- Mean compassion-axis lift:
- Mean irony delta:
- Mean severity delta:
- Null/scrambled control deltas:
- Gate passed:
  - `yes / no`

### Day 5 gate
- Vulnerable bucket compassion lift:
- Arrogant bucket edge retention:
- Neutral bucket compassion lift:
- Gating passed:
  - `yes / no`

### Day 6 gate
- Format-ok rate:
- Accuracy drop on structured reasoning:
- Accuracy drop on repair-after-challenge:
- Sham control result:
- Competence preservation passed:
  - `yes / no`

## Falsification Checks

### Norm / basin sanity
- Are compassionate targets materially lower-norm in common space?
- Are compassionate targets materially lower-margin in common space?
- Does reverse failure look more like:
  - `norm mismatch`
  - `basin depth`
  - `timing error`
  - `two-site intervention need`
  - `genuine entangled control`

### Timing
- Best timing regime:
- Does timing rescue reverse-direction compassion?
- Is timing effect specific to reverse compassion or generic across forward pairs?

### Two-site question
- Did prompt-side plus late-`think` scoring suggest a likely two-site intervention?
- If not tested live, is it now justified?

## Best Working Direction
- Formula:
- Layer / region:
- Patch timing:
- Patch token limit:
- Best source -> target case:
- Best representative output path:

## Failure Modes
- Most common reverse-direction failure:
- Most common competence failure:
- Most common gating failure:
- Most common formatting failure:

## Interpretation
- What moved reliably?
- What stayed sticky?
- Is compassion still entangled with:
  - irony
  - severity
  - punitive framing
  - target vulnerability sensitivity

## Decision Logic

### Continue with linear/composed controls if:
- Day 2 gate passed
- Day 3 gate passed
- Day 5 gate passed
- Day 6 gate passed or only narrowly failed
- and remaining errors look like:
  - timing
  - composition
  - or context gating problems

### Move to targeted SAE work if:
- Day 2 gate passed
- but Day 3-6 repeatedly fail despite:
  - orthogonalized directions
  - timing sweeps
  - sham controls
  - gating tests
- and the same reverse-direction stickiness remains

## If SAE Is Triggered
- Target layers:
- Target region:
- Dataset slice:
- Primary hypothesis:
- What simpler method failed first:

## Concrete Next Step
- next artifact:
- next script:
- next compute allocation:
