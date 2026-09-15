# Entropy-Gated Per-Sample Trust Regions: Routing Gradients from Confidently-Correct Samples into No-Decrease Constraints

## Abstract

Standard empirical risk minimization treats every training sample as an *objective*: each
example contributes a gradient term to a single scalar loss, and the optimizer follows the
sum. We propose an alternative decomposition in which a sample's role in optimization is
determined by its own predictive state. Samples that are misclassified, or correct but
uncertain, act as objectives and supply the descent direction. Samples that are both correct
and *confident* — operationalized by an information-theoretic gate, e.g. predictive entropy
\(H (x) \le H_{\mathrm{mem}} = 0.5\) nats — are removed from the objective and instead
instantiate **inequality constraints** on the aggregate update: the step must not decrease
their margin (equivalently, must not increase their loss or entropy) to first order. The
resulting per-step problem is a small quadratic program whose solution is a projection of the
raw objective direction onto a polyhedral *no-decrease cone*, with dual multipliers that admit
natural reading as the shadow price of each stabilized sample. We call the method **EG-PTGP** (Entropy-Gated Per-Sample
Trust-Region Gradient Projection). We give the
formulation, closed-form and approximate solvers, a spectrum of **region policies** — from
one aggregate no-decrease region, through one region per class or per gradient-space cluster,
to one region per sample — together with a fidelity bound that makes clustered regions the
principled middle, a first-order stability analysis, an honest
placement relative to large-margin losses, trust-region policy optimization, GEM/A-GEM,
orthogonal gradient descent, loss flooding, and confidence penalties, and a pre-registered set
of falsifiable predictions together with the failure modes we expect (chiefly *over-frozen
correctness* and *gradient dead zones*). The central claim is not that constrained descent is
new, but that **certainty-gated, within-task, per-sample constraint routing** is a distinct and
testable training regime with a principled stopping criterion for memorization.

---

## 1. Introduction

In canonical supervised training we minimize

$$
\mathcal{L} (\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell_i (\theta),
\qquad
\theta_{t+1} = \theta_t - \eta\, \nabla_\theta \mathcal{L} (\theta_t),
$$

which assigns every sample the same *structural role*. A sample that the model already
classifies with 0.999 confidence still contributes a gradient that pushes parameters, and that
push interacts — constructively or destructively — with the pushes coming from samples the
model gets wrong. Two well-documented pathologies follow. First, **gradient interference**: the
aggregate step can degrade examples that were previously correct, producing prediction churn,
boundary oscillation, and (across tasks) catastrophic forgetting. Second, **effort
misallocation**: with cross-entropy, easy examples never stop asking for more confidence, so a
non-trivial fraction of the gradient budget is spent driving already-solved examples from
\(10^{-2}\) to \(10^{-6}\) loss rather than repairing the frontier.

The mechanism we propose is a change of *role*, not a change of loss shape:

> If the model already gets a sample right, and is confident about it, that sample should stop
> being a thing we optimize and start being a thing we must not break.

Concretely, we partition each batch into an **objective set** \(E\) and a **constraint set**
\(M\) ("memorized"), and we solve, at each step,

$$
d^\star = \arg\min_{d}\ \tfrac{1}{2}\lVert d - d_{\mathrm{obj}}\rVert^2
\quad \text{s.t.}\quad \langle v_i, d\rangle \le \varepsilon_i \ \ \forall i \in M,
$$

where \(d_{\mathrm{obj}}\) is the ordinary mini-batch gradient restricted to \(E\), \(v_i\) is
the gradient of sample \(i\)'s margin, and \(\varepsilon_i \ge 0\) is a certainty-dependent
slack. Membership in \(M\) is gated by an entropy threshold, which supplies a *principled
criterion for when a sample is done being learned*: not "zero loss", but "entropy below
\(H_{\mathrm{mem}}\) nats".

Two design commitments distinguish this from nearby methods and deserve stating up front.

1. **The constraint is on the aggregate step, not on the sample's own gradient.** A subtlety
   that is easy to get wrong (and which we derive in Appendix A) is that for cross-entropy, a
   correctly-classified sample's *own* negative gradient already increases its own margin. A
   literal per-sample projection of \(g_i\) onto \(\{g : \langle g, v_i\rangle \le 0\}\) is
   therefore almost always vacuous. The constraint only *binds* under aggregation: it is the
   sum over the batch — dominated by error gradients — that can erode sample \(i\)'s margin.
   EG-PTGP is therefore a per-sample- *constraint*, whole-step- *projection* method.
2. **The gate is information-theoretic and two-sided.** Correctness alone is a brittle gate (a sample can be correct at
   0.34/0.33/0.33). Entropy adds the missing axis: the gate fires
   only for correct *and* low-entropy samples, and the same threshold simultaneously acts as a **memorization cap** — we
   do not ask the objective to push entropy below
   \(H_{\mathrm{mem}}\), we only ask it not to rise back above it.
3. **Constraints are organized by a region policy, not necessarily one per sample.** The
   name says "per-sample", and one region per memorized sample is the exact end of the
   spectrum, but the method is defined for *any* aggregation of \(M\) into \(k\) no-decrease
   regions: one global region, one per class, one per (class, runner-up) pair, or one per
   gradient-space cluster (§3.7). Clustering is not merely a cost reduction. A region's
   constraint protects the region's *mean* margin gradient, and how well that protects each
   member is governed by how aligned the members' gradients are (Proposition 2, §4.5). The
   partition is therefore part of the method, and we argue that clustered regions — not the
   two extremes — are where fidelity and plasticity balance.

---

## 2. Preliminaries and notation

Let \(f_\theta : \mathcal{X} \to \mathbb{R}^K\) produce logits \(z (x) = f_\theta (x)\) and
probabilities \(p_j (x) = \mathrm{softmax}_j (z (x))\). For a labelled sample \( (x_i, y_i)\):

- per-sample loss \(\ell_i (\theta) = -\log p_{y_i} (x_i)\), gradient \(g_i = \nabla_\theta \ell_i\);
- predictive entropy \(H_i = -\sum_j p_j (x_i)\log p_j (x_i)\) (nats), gradient \(h_i = \nabla_\theta H_i\);
- logit margin \(m_i (\theta) = z_{y_i} (x_i) - \max_{j\ne y_i} z_j (x_i)\), gradient \(v_i = \nabla_\theta m_i\);
- correctness indicator \(c_i = \mathbb{1}[m_i > 0]\).

We write the update as \(\theta_{t+1} = \theta_t - \eta\, d\) for a search direction \(d\). To
first order in \(\eta\),

$$
\Delta m_i \approx -\eta \langle v_i, d\rangle,
\qquad
\Delta \ell_i \approx -\eta \langle g_i, d\rangle,
\qquad
\Delta H_i \approx -\eta \langle h_i, d\rangle .
$$

Hence the three equivalent renderings of "do not degrade sample \(i\)":

$$
\textbf{margin:}\ \langle v_i, d\rangle \le 0,
\qquad
\textbf{loss:}\ \langle g_i, d\rangle \ge 0,
\qquad
\textbf{entropy:}\ \langle h_i, d\rangle \ge 0 .
$$

The margin form is scale-stable and directly geometric; the loss form is the GEM-style
constraint; the entropy form is the one that matches our gate. They are not identical — margin
can grow while entropy also grows (if a third class rises) — and which to use is an empirical
question we treat as an ablation (§7.4). Unless stated otherwise we use the margin form with
an entropy-form auxiliary.

---

## 3. Method: EG-PTGP

### 3.1 The certainty gate

Fix a memorization entropy \(H_{\mathrm{mem}}\) (default \(0.5\) nats) and a soft band width
\(\tau\). Define the **stability weight**

$$
s_i \;=\; c_i \cdot \sigma\!\big (-\beta\, (H_i - H_{\mathrm{mem}})\big) \in [0,1],
\qquad \beta = 1/\tau ,
$$

so \(s_i \to 1\) for confidently-correct samples and \(s_i \to 0\) for misclassified or
high-entropy ones. The complementary **objective weight** is \(\alpha_i = 1 - s_i\). Three
regimes emerge, matching the intended semantics:

| regime              | condition                                 | role                                   |
|---------------------|-------------------------------------------|----------------------------------------|
| error               | \(c_i = 0\)                               | pure objective, full gradient budget   |
| fragile correctness | \(c_i = 1\), \(H_i > H_{\mathrm{mem}}\)   | mostly objective, weak constraint      |
| memorized           | \(c_i = 1\), \(H_i \le H_{\mathrm{mem}}\) | pure constraint, zero objective weight |

The hard-gated variant uses \(s_i = c_i \cdot \mathbb{1}[H_i \le H_{\mathrm{mem}}]\).

### 3.2 Objective direction

$$
d_{\mathrm{obj}} \;=\; \frac{1}{\sum_i \alpha_i}\sum_{i \in \mathcal{B}} \alpha_i\, g_i .
$$

Note what this alone does: it is an **entropy hinge**, or "flooding in entropy space". Once a
sample crosses below \(H_{\mathrm{mem}}\) its objective pull vanishes, so cross-entropy stops
sharpening it. This half of the method is close in spirit to known techniques (hinge losses,
confidence penalties, loss flooding, self-adaptive training) and we expect it to account for
part — but not all — of the observed effect; the ablation in §7.4 is designed to separate the
two halves.

### 3.3 Constraint set and the no-decrease cone

Let \(M = \{ i : s_i > s_{\min}\}\), optionally augmented with a reservoir buffer of previously
memorized samples (§3.6). Define per-sample slack

$$
\varepsilon_i \;=\; \kappa\, (1 - s_i)\,\lVert v_i\rVert ,
$$

so that fully-memorized samples get a hard constraint (\(\varepsilon_i = 0\)) and borderline
samples are allowed to erode a little margin in exchange for the model's freedom to improve
elsewhere. The feasible set is the (shifted) polyhedral cone

$$
\mathcal{C} \;=\; \{\, d : \langle v_i, d\rangle \le \varepsilon_i,\ \forall i \in M \,\}.
$$

### 3.4 The step: projection onto \(\mathcal{C}\)

$$
\boxed{\;
d^\star = \Pi_{\mathcal{C}}\big (d_{\mathrm{obj}}\big)
= \arg\min_{d \in \mathcal{C}} \tfrac12 \lVert d - d_{\mathrm{obj}}\rVert^2 ,
\qquad \theta_{t+1} = \theta_t - \eta\, d^\star .}
$$

The Lagrangian dual is a non-negative least-squares problem of dimension \(|M|\):

$$
d^\star = d_{\mathrm{obj}} - \sum_{i\in M} \lambda_i v_i ,
\qquad
\lambda = \arg\min_{\lambda \ge 0} \tfrac12 \lambda^\top G \lambda - \lambda^\top (b - \varepsilon),
$$

with Gram matrix \(G_{ij} = \langle v_i, v_j\rangle\) and \(b_i = \langle v_i, d_{\mathrm{obj}}\rangle\).
Complementary slackness \(\lambda_i \, (\langle v_i, d^\star\rangle - \varepsilon_i) = 0\) means
only the samples *actually about to be damaged* pay anything. The multipliers \(\lambda_i\) are
the rigorous version of the "stability credit" intuition: \(\lambda_i\) is the marginal amount
of objective progress being surrendered to keep sample \(i\) intact.

**Single-constraint closed form** (the A-GEM-style approximation, using the mean
\(\bar v = \frac{1}{|M|}\sum_{i\in M} v_i\)):

$$
d^\star = d_{\mathrm{obj}} - \frac{\max\big (0,\ \langle \bar v, d_{\mathrm{obj}}\rangle - \bar\varepsilon\big)}{\lVert \bar v\rVert^2}\,\bar v .
$$

This costs one extra gradient-sized buffer and one inner product, and is the variant we
recommend as the default baseline implementation.

### 3.5 Soft / Lagrangian variant

Instead of solving the QP each step, maintain persistent multipliers and do dual ascent:

$$
d = d_{\mathrm{obj}} - \sum_{i \in M}\lambda_i v_i,
\qquad
\lambda_i \leftarrow \big[\lambda_i + \eta_\lambda (\langle v_i, d\rangle - \varepsilon_i)\big]_+ .
$$

This is the KKT-flavoured branch: cheaper, smoother, no exact feasibility guarantee. The hard
projection is the TRPO-flavoured branch: exact first-order feasibility, higher cost, more
brittle when \(|M|\) is large. We treat hard-vs-soft as a first-class experimental axis because
we do not know which wins, and the notes that motivated this work explicitly left it open.

### 3.6 Making it affordable

Per-sample gradients \(v_i\) over all parameters are the obvious cost centre. Four reductions,
in increasing order of aggressiveness:

1. **Last-layer / head-only projection.** Constrain only the final linear layer (and optionally
   the last block). For \(m_i\), \(v_i\) restricted to the head is available analytically from
   the penultimate features, so the Gram matrix is cheap. Empirically most decision-boundary
   geometry lives here.
2. **Logit-space (Gauss–Newton/NTK) surrogate.** With \(J_i = \partial z (x_i)/\partial\theta\),
   \(\Delta z_i \approx -\eta J_i d\); constraints on \(\Delta m_i\) are then constraints on a
   single row-combination of \(J_i\). This lets one express "don't move this sample's logits in
   the bad direction" without materializing full parameter gradients.
3. **Region aggregation.** Partition \(M\) into \(k\) regions and constrain region means.
   \(k = 1\) recovers §3.4's closed form; \(k = |M|\) is the exact per-sample cone. The
   choice of partition is consequential enough to deserve its own section (§3.7).
4. **Reservoir subsampling.** Sample \(k \ll |M|\) constraints per step from a memory buffer of
   memorized examples, so the *population* of constraints is respected in expectation while
   per-step cost is \(O (k)\).

### 3.7 Region policy: how many no-decrease regions, and who shares one

The constraint channel admits a one-parameter family of policies indexed by how the memorized
set is aggregated into constraints. The two extremes are easy to state; the interesting
territory is in between.

**Extreme 1 — one region (\(k = 1\)).** A single constraint on the \(s\)-weighted mean margin
gradient: *the average margin (or, in loss form, the average loss) of already-learned samples
must not degrade.* This is the closed form of §3.4 and the A-GEM analogue. It is the cheapest
policy and can never produce a dead zone (a single half-space removes at most one direction),
but it protects only the average. A step may sacrifice individual memorized samples as long as
others gain, and — worse — opposed margin gradients *cancel* in the mean, so the constraint is
closest to vacuous precisely when the memorized set is internally conflicted, which is when
protection is most needed.

**Extreme 2 — one region per memorized sample (\(k = |M|\)).** Exact first-order protection
of every memorized sample. Costs \(O (|M|)\) gradients, an \(O (|M|^2 P)\) Gram matrix and an
NNLS solve; the feasible set shrinks fastest with training, so the dead zone of H8 arrives
earliest. Viable only head-only with a small reservoir.

**Middle — one region per cluster.** Partition \(M\) into \(k\) regions
\(\{\mathcal{R} _c\}_{c=1}^k\) and let each contribute one constraint

$$
\bar v_c = \frac{\sum_{i\in\mathcal{R}_c} s_i v_i}{\sum_{i\in\mathcal{R}_c} s_i},
\qquad
\bar\varepsilon_c = \frac{\sum_{i\in\mathcal{R}_c} s_i \varepsilon_i}{\sum_{i\in\mathcal{R}_c} s_i}
\;-\; \zeta\,\sigma_c\,\lVert d_{\mathrm{obj}}\rVert ,
\qquad
\langle \bar v_c, d\rangle \le \bar\varepsilon_c ,
$$

where \(\sigma_c\) is a within-region dispersion radius (§4.5) and \(\zeta \ge 0\) tightens
the slack to make the region constraint conservative for its members (\(\zeta = 0\) is the
plain mean constraint). Proposition 2 shows that the first-order damage a region constraint
can leak onto an individual member is bounded by that member's distance from the region
centroid, so *the partition should minimize within-region gradient dispersion*. That
criterion ranks the candidate clustering strategies:

(a) **By class** (\(k = K\)). "No class the model already knows may get worse this step" —
the per-category average-loss region of the original notes. Cheap and interpretable (and
it doubles as a fairness constraint over classes), but a weak geometric grouping: samples
of one class with different runner-up classes have only partially aligned margin
gradients, and within-class feature diversity is large.

(b) **By (class, runner-up) pair.** In the head-only parametrization the margin gradient
factorizes as \(v_i = (e_{y_i} - e_{r_i}) \otimes \phi_i\) with \(\phi_i\) the
penultimate features, so
\(\langle v_i, v_j\rangle = \langle e_{y_i} - e_{r_i},\, e_{y_j} - e_{r_j}\rangle\,\langle \phi_i, \phi_j\rangle\).
Same pair gives \(2\langle\phi_i,\phi_j\rangle\); same class, different runner-up gives
\(\langle\phi_i,\phi_j\rangle\); and if \(i\)'s class is \(j\)'s runner-up the class-space
factor is *negative* — the cancellation case. Grouping by pair therefore fixes the
class-space direction exactly and leaves only feature dispersion inside a region. The
number of *populated* pairs is far below \(K (K-1)\): it is the set of confusable class
pairs, i.e. one region per live decision boundary. This is our recommended default for
classification at moderate \(K\).

(c) **By feature-space cluster.** \(k\)-means on \(\phi_i\), which is already computed.
Approximates (b) when \(K\) is large enough that pairs are too many, since features of a
class tend to be locally coherent and runner-ups are locally consistent.

(d) **By gradient-space cluster.** Spherical \(k\)-means (cosine) on the \(v_i\) themselves.
This *directly* minimizes the leakage bound of Proposition 2 and rules out cancellation by
construction, because opposed gradients cannot share a region. It looks expensive, but in
the head-only form the Gram over \(M\) factorizes,
\(G = (A A^\top) \odot (\Phi \Phi^\top)\) with rows \(a_i = e_{y_i} - e_{r_i}\), so kernel
\(k\)-means on \(G\) never materializes \(v_i\). This is the strategy we consider most
promising and the one the ablations should be built around.

(e) **Random partition.** The control. It gets the variance reduction of averaging but no
alignment. If (d) does not beat (e) at matched \(k\), clustering is not load-bearing and
regions are pure cost machinery.

(f) **Adaptive / hierarchical.** Start at \(k = 1\); *split* a region when its measured
post-step violation rate (instrumentation §7.1 (g)) or dispersion \(\sigma_c\) exceeds a
threshold; *merge* two regions whose multipliers have both been zero for \(T\) steps or
whose centroids are within an angle \(\vartheta\). This ties the region policy to the
dead-zone controller: growing \(k\) is the "tighten" action and merging is the "relax"
action, so a single controller targeting
\(\lVert d^\star\rVert/\lVert d_{\mathrm{obj}}\rVert\) can drive both.

(g) **Subspace (OGD-style).** Not a partition: take the top-\(r\) right singular vectors of
\(V\) and project \(d\) onto their orthogonal complement. Stronger than needed (equality
rather than inequality, sign of the margin change ignored, slack semantics lost); we keep
it as a strong-freeze baseline rather than a candidate.

**Online maintenance.** \(M\) changes every step, so centroids are maintained on the
reservoir \(R\) rather than the batch: re-cluster \(R\) every \(T_c\) steps (kernel
\(k\)-means on the factorized Gram, \(O (|R|^2)\)), assign batch members to the nearest
centroid at \(O (|M| k)\) inner products, route newly memorized samples to their nearest
centroid, and merge any region whose population falls below \(n_{\min}\).

**Choosing \(k\).** Larger \(k\) means higher fidelity (Prop. 2) and a smaller feasible set (the union of fine
constraints implies the coarse mean constraint, not conversely), hence
earlier dead zones. We predict an interior optimum (H9), and that the right scale for \(k\) is
the number of *confusable class pairs* rather than \(|M|\) or \(K\).

**Summary of the spectrum.**

| policy                      | \(k\)             | protects          | cancellation risk    | dead-zone risk |
|-----------------------------|-------------------|-------------------|----------------------|----------------|
| one region (mean)           | 1                 | average only      | high                 | none           |
| per class                   | \(K\)             | class averages    | medium               | low            |
| per (class, runner-up) pair | #confusable pairs | boundary averages | low                  | moderate       |
| gradient-space clusters     | tunable           | aligned groups    | none by construction | moderate       |
| random partition            | tunable           | nothing specific  | high                 | moderate       |
| per sample                  | \(                | M                 | \)                   | every sample   | none | high |

### 3.8 Algorithm

~~~
Algorithm 1: EG-PTGP (one step)
inputs: batch B, params θ, lr η, H_mem, β, κ, warmup T0, buffer R
 1: forward pass -> logits z_i, probs p_i for i in B
 2: H_i  <- -Σ_j p_ij log p_ij
 3: c_i  <- 1[argmax_j z_ij == y_i]
 4: s_i  <- c_i * sigmoid(-β (H_i - H_mem));   α_i <- 1 - s_i
 5: if t < T0:  s_i <- 0  (warmup: plain ERM, entropy estimates untrustworthy)
 6: d_obj <- Σ_i α_i g_i / Σ_i α_i                 # objective channel
 7: M <- {i in B : s_i > s_min}  ∪  sample(R, n_R) # constraint channel
 8: compute v_i (margin grads, head-only or via NTK surrogate) for i in M
 9: ε_i <- κ (1 - s_i) ||v_i||
10: c(i) <- region of i, i in M   (§3.7: class / (y,r) pair / nearest gradient-space centroid)
11: v̄_c <- Σ_{c(i)=c} s_i v_i / Σ s_i ;  ε̄_c <- Σ_{c(i)=c} s_i ε_i / Σ s_i - ζ σ_c ||d_obj||
12: d*  <- Π_C(d_obj),  C = {d : <v̄_c, d> ≤ ε̄_c ∀c}   via NNLS dual  (k=1: closed form)
13: if ||d*|| < δ ||d_obj||:  relax (increase κ, merge regions, or drop oldest)  # dead-zone guard
14: θ <- optimizer_step(θ, d*)                    # d* may be fed to SGD/Adam as the "gradient"
15: update R with newly memorized samples (reservoir sampling); every T_c steps refresh centroids
~~~

Line 14 matters: \(d^\star\) is substituted for the gradient *before* the adaptive optimizer, so
Adam's preconditioning applies to the already-projected direction. Projecting after
preconditioning breaks the first-order feasibility guarantee (the constraint geometry is
defined in gradient space, not in Adam-rescaled space); this is a real implementation trap and
an ablation worth running.

---

## 4. Analysis

### 4.1 First-order stability

**Proposition 1 (no-decrease to first order).** If \(d^\star\) is feasible and
\(\theta_{t+1} = \theta_t - \eta d^\star\), then for every \(i \in M\),
\(m_i (\theta_{t+1}) \ge m_i (\theta_t) - \eta\varepsilon_i + O (\eta^2 L_i)\), where \(L_i\)
bounds the curvature of \(m_i\) along the step. With \(\varepsilon_i = 0\) the memorized set is
margin-non-decreasing up to \(O (\eta^2)\).

The \(O (\eta^2)\) term is not cosmetic. All guarantees here are first-order; with large
learning rates, momentum, or sharp curvature the constraint can be violated in practice. Three
mitigations: (i) shrink \(\eta\) when the active set is large; (ii) add a curvature margin by
requiring \(\langle v_i,d\rangle \le -\gamma\eta\lVert d\rVert^2\); (iii) re-check feasibility
post-hoc on a held-out slice of \(M\) and backtrack. We recommend (iii) as a diagnostic even
when not used as a control.

### 4.2 Fixed points and what "converged" means

A point \(\theta\) is a fixed point of EG-PTGP when either \(E = \emptyset\) (everything is
correct and below \(H_{\mathrm{mem}}\)) or \(d^\star = 0\), i.e. \(d_{\mathrm{obj}}\) lies in the
normal cone \(\mathrm{cone}\{v_i\}_{i\in M}\). The second case is the interesting one: it is a
KKT point of the *constrained* problem

$$
\min_\theta \ \sum_i \alpha_i \ell_i (\theta)
\quad \text{s.t.}\quad m_i (\theta) \ge m_i^{\,0}\ \ \forall i \in M,
$$

and it says something legible: **training halts not when the loss is zero but when every
remaining error is unfixable without breaking something already known.** That is a genuinely
different stopping semantics from ERM, and it is directly measurable (report the active-set
size and \(\lVert d^\star\rVert/\lVert d_{\mathrm{obj}}\rVert\) over training).

### 4.3 Why the entropy cap is not just early stopping

A memorized sample contributes zero objective gradient but a live constraint. So the model is
free to keep improving *elsewhere* while that sample's confidence is pinned in a band around
\(H_{\mathrm{mem}}\) rather than driven to zero entropy. The resulting predictive distributions
are, by construction, not saturated — which predicts better calibration (lower ECE/NLL) and
more headroom for later revision, at the cost of slightly worse training loss. Early stopping,
by contrast, freezes everything at once and cannot allocate plasticity selectively.

### 4.4 The gradient-economy reading, made precise

The metaphor that motivated this work — misclassified samples "spend" gradient, correct samples
"accumulate stability credit" — turns out to be the dual problem:

- \(d_{\mathrm{obj}}\) is demand for parameter movement from unsolved examples;
- each \(v_i\) is a *right of way* held by a solved example;
- \(\lambda_i\) is the shadow price paid to pass through, nonzero only for binding rights;
- \(\lVert d_{\mathrm{obj}} - d^\star\rVert\) is the total price of the step, i.e. how much
  learning was forgone to preserve what is known.

This makes the economy instrumentable. Per-sample cumulative \(\sum_t \lambda_i^{ (t)}\) is a *stability price index*
identifying which examples are structurally expensive — a plausible
detector for mislabeled data, prototypes, and conflicting supervision.

### 4.5 Region fidelity: what a clustered constraint actually guarantees

A region constraint protects a mean. The question is how much of that protection reaches the
members.

**Proposition 2 (leakage bound).** Let region \(\mathcal{R}_c\) have \(s\)-weighted mean
\(\bar v_c\) and suppose \(d\) satisfies \(\langle \bar v_c, d\rangle \le \bar\varepsilon_c\).
Then for every \(i \in \mathcal{R}_c\),

$$
\langle v_i, d\rangle
= \langle \bar v_c, d\rangle + \langle v_i - \bar v_c, d\rangle
\le \bar\varepsilon_c + \lVert v_i - \bar v_c\rVert\,\lVert d\rVert ,
$$

so the first-order margin loss of an individual member is bounded by its distance from the
region centroid times the step length:
\(\Delta m_i \ge -\eta\big (\bar\varepsilon_c + \lVert v_i - \bar v_c\rVert\,\lVert d\rVert\big) + O (\eta^2)\).
Because \(0 \in \mathcal{C}\) and projection onto a convex set containing the origin is
norm-non-increasing, \(\lVert d^\star\rVert \le \lVert d_{\mathrm{obj}}\rVert\), which is why
the tightened slack \(\bar\varepsilon_c - \zeta\sigma_c\lVert d_{\mathrm{obj}}\rVert\) in §3.7
with \(\sigma_c = \max_{i\in\mathcal{R}_c}\lVert v_i - \bar v_c\rVert\) and \(\zeta = 1\)
makes the region constraint *imply* every member's individual constraint (at the cost of a
smaller feasible set). \(\square\)

**Corollary (the partition is the \(k\)-means objective).** Summing the squared bound radii
over a partition gives \(\sum_c \sum_{i\in\mathcal{R}_c} \lVert v_i - \bar v_c\rVert^2\), which
is exactly the \(k\)-means objective on the margin gradients. Gradient-space \(k\)-means at
fixed \(k\) is therefore the partition that minimizes the aggregate worst-case leakage, and
class-, pair- and feature-based partitions are good to the extent that they approximate it.

**Corollary (cancellation).** If a region contains \(v_i, v_j\) with \(\langle v_i, v_j\rangle < 0\),
then \(\lVert \bar v_c\rVert\) can be small while the radii are large: the constraint
approaches vacuous and both members are individually exposed. In the head-only
factorization of §3.7 (b) the sign of \(\langle v_i, v_j\rangle\) is governed by whether the (class, runner-up) pairs
agree, so the one-region policy is structurally exposed to
cancellation between samples on opposite sides of the same decision boundary — the
\(y_i = r_j\) case — which is precisely the frontier where boundary refinement happens.
Spherical \(k\)-means avoids this by clustering on direction rather than magnitude.

**Fidelity–plasticity trade-off.** Refining a partition can only shrink \(\mathcal{C}\): if
every member constraint holds, the mean constraint holds, but not conversely. So \(k\) trades
leakage (small \(k\)) against dead zones (large \(k\)), and neither extreme is expected to be
optimal. The instrumentation in §7.1 (g) measures leakage directly — post-step violations of *individual* memorized
samples — so this trade-off is observable, not just argued.

### 4.6 Complexity

| variant                                | extra memory                          | extra compute / step                                                  |
|----------------------------------------|---------------------------------------|-----------------------------------------------------------------------|
| single mean constraint (§3.4)          | 1 gradient buffer                     | 1 backward over \(M\), 1 inner product                                |
| \(k\) region constraints (§3.7)        | \(k\) buffers                         | 1 backward over \(M\) + \(k\) weighted sums, \(O(k^2 P)\) Gram + NNLS |
| + gradient-space clustering, head-only | \(k \cdot P_{\text{head}}\) centroids | \(O(                                                                  |R|^2)\) factorized Gram every \(T_c\) steps, \(O(|M|k)\) assignment |
| head-only, \(k\) constraints           | \(k \cdot P_{\text{head}}\)           | negligible                                                            |
| full per-sample, \(                    | M                                     | \) large                                                              | prohibitive | prohibitive |

The practical recommendation is head-only with (class, runner-up) pair or gradient-space
regions and \(k \in [4, 32]\); the single-mean constraint is the baseline, not the default.

---

## 5. Relation to prior work

We want the novelty claim to be narrow and defensible, so we state what is *not* new.

**Constrained/projected updates that avoid degrading stored examples.** Gradient Episodic
Memory (GEM) and A-GEM solve essentially the projection of §3.4 using loss-form constraints
\(\langle g_i, d\rangle \ge 0\) on episodic memories; Orthogonal Gradient Descent projects onto
the orthogonal complement of old-task gradients. EG-PTGP's projection machinery is the same
family. The differences are: (i) constraints are generated *within* a task, online, by a
certainty criterion rather than by task boundaries; (ii) constraints are margin/entropy-form
rather than loss-form; (iii) the same gate also removes those samples from the objective. (iv) The constraint set is
organized by an explicit region policy (§3.7). Seen this way,
GEM's one-constraint-per-past-task is a partition by task identity and A-GEM's single mean
constraint is the \(k = 1\) policy; EG-PTGP asks which partition of the protected set is *right* and answers with a
fidelity bound (Prop. 2) rather than with the task boundary.

**Trust regions.** TRPO/PPO constrain a global divergence between successive policies.
EG-PTGP's cone is per-sample and anchored to individual decision margins, so the trust region
is *shaped* rather than spherical — but the projection logic is inherited.

**Large-margin objectives.** SVM hinge, L-Softmax, ArcFace/CosFace reshape the loss to reward
margin. They remain objectives: a large-margin sample still pulls. EG-PTGP converts margin from
a reward into a *floor*.

**Saturation control on the objective side.** Loss flooding, confidence penalties, label
smoothing, and self-adaptive/abstention losses all stop or reverse the drive toward zero loss.
Our entropy hinge (§3.2) belongs here; the \(H_{\mathrm{mem}} = 0.5\) nats gate is a
particularly interpretable instance ("memorize to no more than half a nat"), but we do not
claim the mechanism is new in kind.

**Example weighting.** Curriculum learning, hard-example mining, focal loss, and selective
backprop reweight or drop easy samples. Dropping an easy sample and *constraining* on it are
materially different: the first surrenders its geometry, the second defends it.

**Therefore the claimed contribution is:** the composition — an information-theoretic,
two-sided certainty gate that simultaneously (a) retires a sample from the objective and (b) enlists it as a first-class
inequality constraint on the aggregate step, with slack and
dual prices scaling in the sample's own uncertainty. To our knowledge this specific composition,
applied within-task and online, has not been studied as a training regime.

---

## 6. Predictions and pre-registered hypotheses

Stated so they can fail.

**H1 — Reduced prediction churn.** The per-epoch fraction of test predictions that flip will
drop substantially (target: ≥30% relative reduction) versus matched ERM at equal accuracy. *Falsified if* churn is
unchanged or worse once learning rate schedules are matched.

**H2 — Better calibration.** ECE and NLL improve, driven by the entropy cap; confidence
histograms should show mass accumulating near \(e^{-H_{\mathrm{mem}}}\)-ish rather than at 1.0. *Falsified if*
accuracy-matched ECE does not improve, or if the improvement is fully explained
by the §3.2 ablation (in which case the constraint channel adds nothing calibration-wise).

**H3 — Faster reduction of error on hard examples.** With easy samples retired, the effective
gradient concentrates on the frontier; expect faster fall in training error on the hardest
decile at matched step count. *Falsified if* frontier error falls no faster, or if the retired
samples' margins decay anyway.

**H4 — Emergent stability manifold.** Margin distributions become bimodal: a tight
high-confidence mode pinned near the gate, and a diffuse frontier mode. Samples should cross
\(H_{\mathrm{mem}}\) mostly once, with low recidivism (target: <5% of memorized samples ever
returning to \(E\)). *Falsified if* the entropy trajectory shows heavy oscillation across the
gate — that would indicate the constraints are not doing their job or the gate has hysteresis
problems (add a Schmitt-trigger band if so).

**H5 — Robustness to label noise.** Under symmetric label noise, EG-PTGP should resist
memorizing noisy labels *later* than ERM, because noisy samples are rarely simultaneously
correct and low-entropy early, and because the entropy cap removes the mechanism by which
cross-entropy drives noisy examples to zero loss. Additionally, cumulative \(\lambda_i\) should
be diagnostic of noisy labels (AUC > 0.8). *Falsified if* noise memorization matches ERM.

**H6 — Reduced forgetting in continual settings.** Because the constraint set is generated
automatically and continuously, split-task benchmarks should show reduced forgetting even *without* task boundaries
being supplied. *Falsified if* EG-PTGP underperforms A-GEM given the
same memory budget.

**H7 — Non-monotone dependence on \(H_{\mathrm{mem}}\).** Very small \(H_{\mathrm{mem}}\) (e.g. 0.01 nats) degenerates
toward ERM; very large (e.g. 2 nats in a 10-class problem)
over-freezes and underfits. We predict an interior optimum, plausibly 0.2–0.7 nats for 10-class
problems, scaling roughly with \(\log K\). *Falsified if* performance is monotone in
\(H_{\mathrm{mem}}\) over a wide range — which would mean the gate is not the operative variable.

**H8 — Gradient dead zones are real.** As \(|M|\) grows late in training,
\(\lVert d^\star\rVert / \lVert d_{\mathrm{obj}}\rVert\) will collapse toward zero and progress
will stall unless slack \(\kappa\) is annealed upward or constraints are subsampled. We predict
this is the dominant practical failure mode, not instability. **H9 — The region policy is load-bearing.** At matched
\(k\) and compute, gradient-space (or (class, runner-up) pair) regions will show (a) a lower measured post-step
*individual*
violation rate among memorized samples than random or per-class regions, and (b) a smaller
price ratio \(\lVert d_{\mathrm{obj}} - d^\star\rVert / \lVert d_{\mathrm{obj}}\rVert\) than
the per-sample policy at equal violation rate. Performance will be non-monotone in \(k\):
\(k = 1\) under-protects (individual violations comparable to ERM's churn on the same
samples, driven by cancellation), \(k = |M|\) dead-zones. *Falsified if* random partitions
match gradient-space clustering at every \(k\), or if \(k = 1\) already achieves H4's
recidivism target — either result would mean regions are only cost machinery and the
spectrum of §3.7 collapses to a single knob.

---

## 7. Experimental protocol

### 7.1 Instrumentation (the actually interesting part)

Beyond accuracy, log per epoch: (a) \(|E|,|M|\) and the flow between them; (b) entropy
histograms and the mass crossing \(H_{\mathrm{mem}}\); (c) margin percentiles for \(M\) and
\(E\) separately; (d) active-set size and the \(\lambda\) spectrum; (e)
\(\lVert d^\star\rVert/\lVert d_{\mathrm{obj}}\rVert\) (the "price of stability"); (f) test
prediction churn; (g) measured post-step constraint violations, to quantify the \(O (\eta^2)\)
gap.

### 7.2 Benchmarks

MNIST/Fashion-MNIST (mechanism visibility, small MLP/CNN), CIFAR-10/100 (ResNet-18,
ViT-small), CIFAR-10N/100N and synthetic symmetric/asymmetric noise (H5), split-CIFAR-100 and
split-TinyImageNet (H6), and a small-scale LLM fine-tuning probe where the gate is applied
per-token (entropy over the vocabulary) rather than per-sequence.

### 7.3 Baselines

ERM; ERM + label smoothing; ERM + confidence penalty; loss flooding; focal loss; selective
backprop; A-GEM with equal memory; L-Softmax/ArcFace; early stopping tuned per-sample-free.
Matched compute and matched tuning budget throughout; report at equal *steps* and equal *wall-clock*, since projection
is not free.

### 7.4 Ablations (these decide whether the idea has content)

1. **Objective gate only** (\(\alpha_i\) as in §3.2, no constraints). Isolates the entropy-hinge
   effect against the literature.
2. **Constraints only** (all samples keep full objective weight, memorized samples add
   constraints). Isolates the projection effect.
3. **Correctness gate vs entropy gate** — i.e. is the information-theoretic axis load-bearing?
4. **Constraint form**: margin vs loss vs entropy vs margin+entropy.
5. **Hard projection vs Lagrangian dual ascent** (§3.5).
6. **Region policy** (§3.7, H9): \(k = 1\) vs per-class vs per- (class, runner-up) pair vs
   feature-space vs gradient-space vs random vs adaptive split/merge vs per-sample, at
   \(k \in \{1, 4, 16, 64, |M|\}\) where applicable; slack tightening \(\zeta \in \{0, 0.5, 1\}\);
   centroid refresh period \(T_c\); head-only vs full-network. Report individual violation
   rate and price ratio jointly, since either alone can be gamed by \(k\).
7. **Slack schedule** \(\kappa\): constant vs annealed (dead-zone mitigation, H8).
8. **Warmup length** \(T_0\): entropy is untrustworthy early; does gating before convergence of
   the feature extractor hurt?
9. **Projection before vs after adaptive preconditioning** (§3.8, line 14).
10. **Interaction with BatchNorm**: BN's batch statistics couple samples, so a "per-sample"
    constraint is not truly per-sample. Compare BN vs GroupNorm vs LayerNorm; we expect the
    method to be cleaner under sample-independent normalization, and this is a concrete,
    testable prediction about where the mechanism is well-posed.

---

## 8. Limitations, uncertainties, and expected failure modes

**Over-frozen correctness.** The most likely way this fails: constraints from a large
low-entropy population lock the geometry so tightly that nuanced boundary refinement stops,
yielding *underfitting* of the frontier. Mitigations: slack annealing, constraint subsampling,
capping \(|M|\) per step, or admitting only a random \(\rho\)-fraction of memorized samples as
constraints.

**Gradient dead zones / magnitude collapse.** Feasible sets can shrink toward \(\{0\}\). Needs
an explicit monitor (line 13 of Algorithm 1) and probably a controller that targets a fixed
\(\lVert d^\star\rVert/\lVert d_{\mathrm{obj}}\rVert\) ratio; region merging (§3.7 (f)) is the
natural relaxation action alongside slack annealing.

**Region staleness and mis-assignment.** Centroids are refreshed every \(T_c\) steps on the
reservoir, but the feature map — and hence the margin gradients — drifts continuously. A stale
partition can silently reintroduce cancellation (two now-opposed gradients still sharing a
region), and the leakage bound of Prop. 2 is only as good as the dispersion actually realized
at step \(t\). The measured individual violation rate is the guard; if it drifts upward
between refreshes, \(T_c\) is too long. There is also a chicken-and-egg problem early in
training: the runner-up class \(r_i\), which defines the pair regions, is itself unstable
before the warmup ends.

**Entropy is a noisy and miscalibrated uncertainty proxy**, especially early in training and
under distribution shift. Softmax entropy conflates aleatoric and epistemic uncertainty. Better
proxies (MC-dropout variance, deep ensembles, evidential heads, Dirichlet posteriors) are
strictly more expensive and left as an axis; the warmup phase is a crude but necessary guard.

**First-order-only guarantees.** See §4.1. Momentum and adaptive preconditioning both degrade
the guarantee, and we do not have a clean theory for the interaction.

**Architectural coupling.** BatchNorm, weight decay, dropout, and shared trunks all break the
fiction that a step's effect on one sample is independent of the batch. Weight decay in
particular applies an unconstrained shrinkage that can violate every constraint at once — it
should either be included in \(d_{\mathrm{obj}}\) before projection or excluded deliberately,
and we flag the choice as consequential.

**Redundancy at scale.** Large models already exhibit implicit margin growth and implicit
curricula; the mechanism may be redundant there and most valuable in small-to-medium models,
label-noise regimes, continual learning, and fine-tuning where forgetting is the binding
constraint. We expect the strongest results in fine-tuning, not pretraining.

**Generalization sign is genuinely unknown.** Two competing effects — stabilization (helps) vs
premature freezing (hurts) — and we do not have a prior strong enough to predict the net. The
honest statement is that the method changes *training dynamics* in measurable, predicted ways (H1, H4, H8 are
near-certain); whether test accuracy improves is an open empirical question.

**Threshold transfer.** \(H_{\mathrm{mem}} = 0.5\) nats is a reasonable default for ~10 classes.
For \(K = 1000\) or for token-level language modelling, the appropriate value must be re-derived (plausibly as a
fraction of \(\log K\), or set adaptively as a quantile of the observed entropy
distribution — an adaptive-quantile gate is an attractive variant that removes the
hyperparameter entirely).

---

## 9. Extensions

- **Adaptive gate.** Replace the fixed \(H_{\mathrm{mem}}\) with a running quantile of batch
  entropy, making the memorized fraction a controlled quantity (e.g. "constrain the most
  confident 30%").
- **Hysteresis.** Use separate entry/exit thresholds (\(H_{\text{in}} < H_{\text{out}}\)) to
  prevent gate chatter, exactly as in a Schmitt trigger.
- **Per-token gating for sequence models.** Apply the gate at the token level; confident tokens
  become constraints, uncertain tokens remain objectives. This is a natural fit for fine-tuning
  without capability regression.
- **Price-based data diagnostics.** Use \(\sum_t\lambda_i\) to surface mislabeled, atypical, or
  conflicting examples — a byproduct that requires no extra machinery.
- **Fairness / group constraints.** Replace per-sample cones with per-group cones ("no
  subgroup's accuracy may decrease this step"), which is the same QP with group-mean margin
- **Second-order feasibility.** Add a curvature term via K-FAC or a Gauss–Newton approximation
  gradients — i.e. the per-class region policy of §3.7 (a) with groups in place of classes.
- **Learned region assignment.** Treat the partition as a latent variable and fit it to
  minimize *measured* post-step leakage rather than the dispersion proxy of Prop. 2, e.g. by
  moving samples between regions whose constraints they were observed to violate.
  to make the no-decrease guarantee hold beyond first order.

---

## 10. Conclusion

EG-PTGP proposes a small change in bookkeeping with a large change in semantics: a training
sample is either something the model is *trying to fit* or something the model is *obliged not
to break*, and the boundary between those roles is drawn by an information-theoretic certainty
threshold. Mechanically, the method reduces to a projection of the objective gradient onto a
polyhedral no-decrease cone generated by confidently-correct samples — machinery shared with
GEM/A-GEM and trust-region methods — combined with an entropy hinge that stops cross-entropy
from sharpening what is already known. How the protected set is carved into regions is itself
a design axis — one region, one per sample, or one per class, decision boundary, or
gradient-space cluster — and a simple leakage bound says the clustered middle is where
fidelity to individual samples and freedom to keep learning are balanced. The composition
yields a legible stopping criterion ("no remaining error is fixable without breaking something known"), an
instrumentable notion of
the price of stability (the dual multipliers), and a concrete set of predictions about churn,
calibration, margin bimodality, noise robustness, and the dead-zone failure mode. We have
deliberately stated the parts that are not new, the parts we expect to break, and the ablations
that would show the idea is empty. The next step is to run them.

---

## Appendix A: Why the constraint only binds under aggregation

**Lemma A.1.** For cross-entropy and a correctly-classified sample \(i\), the direction
\(-g_i\) does not decrease \(m_i\) to first order; i.e. \(\langle v_i, g_i\rangle \ge 0\)
under a mild condition on the runner-up class.

*Sketch.* Write \(\ell_i = -\log p_{y_i}\). In logit space,
\(\partial \ell_i/\partial z_j = p_j - \mathbb{1}[j = y_i]\), so the negative logit-gradient
raises \(z_{y_i}\) by \(1 - p_{y_i} > 0\) and lowers every other logit by \(p_j > 0\) —
including the runner-up \(r = \arg\max_{j\ne y_i} z_j\). Since
\(m_i = z_{y_i} - z_r\), the logit-space directional derivative of \(m_i\) along \(-\partial\ell_i/\partial z\)
is \( (1 - p_{y_i}) + p_r > 0\). Pushing through the parameter Jacobian,
\(\langle v_i, g_i\rangle = \big (\partial m_i/\partial z\big)^\top J_i J_i^\top \big (\partial \ell_i/\partial
z\big)\),
which is non-negative whenever the Gram (NTK) block \(J_iJ_i^\top\) is sufficiently close to
isotropic on the two-dimensional span of \(\{e_{y_i} - e_r,\ p - e_{y_i}\}\). \(\square\)

**Consequence.** The naive reading of "route the correct sample's gradient into a no-decrease
region" is a no-op: \(g_i\) already lies in sample \(i\)'s own no-decrease half-space. The
content of the idea is that the *batch* direction \(d_{\mathrm{obj}} = \sum_{j} \alpha_j g_j\)
need not, because error gradients from other samples can have
\(\langle v_i, g_j\rangle > 0\). EG-PTGP therefore constrains the aggregate, which is exactly
the regime in which the geometry is non-trivial and the dual multipliers are informative. This
also explains why the method should be most valuable when batch gradients are *conflicted*
(fine-tuning, continual learning, noisy labels) and least valuable when they are aligned (early
pretraining).

## Appendix B: Dual derivation

With \(V \in \mathbb{R}^{|M|\times P}\) stacking the \(v_i^\top\), the primal is
\(\min_d \tfrac12\lVert d - d_{\mathrm{obj}}\rVert^2\) s.t. \(Vd \le \varepsilon\). The
Lagrangian \(\mathcal{L} (d,\lambda) = \tfrac12\lVert d - d_{\mathrm{obj}}\rVert^2 + \lambda^\top (Vd - \varepsilon)\)
gives stationarity \(d = d_{\mathrm{obj}} - V^\top\lambda\) and the dual

$$
\max_{\lambda \ge 0}\ -\tfrac12 \lambda^\top V V^\top \lambda + \lambda^\top (Vd_{\mathrm{obj}} - \varepsilon),
$$

an NNLS problem of size \(|M|\) with Gram \(G = VV^\top\). For \(|M| = 1\) the solution is
\(\lambda = [\, (\langle v, d_{\mathrm{obj}}\rangle - \varepsilon)/\lVert v\rVert^2\,]_+\),
recovering §3.4. For \(\varepsilon = 0\) and orthogonal \(v_i\), the projection reduces to
removing each offending component independently — the orthogonal-gradient-descent special case.