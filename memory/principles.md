# What AGPT's equivalences do and don't buy

*Updated 2026-11-10*

These lessons came from the optimizer and equivalence work. Check a new idea
against them before writing any code.

- **AGPT is a framework, not a model.**
  - The paper (`docs/paper.md` §2–5) fixes the trie aggregation and the
    gradient factorization. It leaves the recurrence f_θ open.
  - Attention is one expensive choice of f_θ. Tanh, linear and GRU
    recurrences also train through the trie (`rnd/tanh-recurrence`,
    `rnd/linear-recurrence`).
  - Say which level a claim belongs to: a framework claim carries over to
    every f_θ, an implementation claim does not.
- **Test every extension against the SGD-equivalence.**
  - After the change, a node's gradient must still be a sum of per-instance
    gradients of one shared loss.
  - An add-on with its own backprop path makes a different system that
    contains AGPT. Examples are an encoder, an auxiliary loss, or a side
    input injected into the state. The precondition encoder, cap-recurrence
    and the masked-LM combiner were all of this kind, and all were null.
  - Changes to the aggregation graph itself stay in the family: tree
    orientation, partitioning.
- **Know exactly which two quantities an equivalence equates.**
  - The trie's summed gradient equals the full-batch gradient at one θ. So
    AGPT is full-batch gradient descent, not the SGD process.
  - The modified-loss term (η/4)·mean‖g_i‖² reproduces SGD's mean path, but
    on held-out it was worse than plain full-batch descent (Exps 5, 5b).
  - Each time, the theorem held and an inference built on top of it broke.
    Ask whether either equated quantity is one that pays: wall clock, or
    held-out loss. Prefer claims of one measurable rate (passes to parity).
- **Cadence advantage is per unit of compute.**
  - Aggregation wins per step. Cheap noisy steps win, by orders of magnitude,
    per position processed (Exp 5b).
  - Full-batch steps are capped by curvature, so no first-order correction to
    the aggregated step closes the gap.
  - Iterate averaging is the free part, worth about 0.03 nats (Exp 4b).
- **Optimizer mismatch: partly borne out.**
  - The hypothesis (2026-06-08): Adam is built for noisy gradients, and
    exact gradients call for curvature methods.
  - The result (Exp 8, 2026-09-28): L-BFGS needs the exact gradient and works
    on it, accepting 98% of its steps. It takes pd=0 from 12.05 to 5.02. At
    equal traversals, though, pd=1 Adam still wins by 5–6%.
  - The May 2026 verdict that L-BFGS "can't work" was about one history
    shared across partitions, whose curvature pairs mix different subtrees'
    losses. It does not apply to full-pass L-BFGS (`7444685`).
- **Partition units must be mixed in their targets.**
  - Suffix-tree pd=1 units group contexts by the newest character, so each is
    a target-sorted batch. Their target skew (mean KL of a unit's targets from
    the corpus's) is 0.86 nats (context-tree, 2026-10-03 and 2026-11-10).
  - Prefix-trie units group by the oldest, least predictive character (skew
    0.09). At an equal number of steps they train as well as shuffled
    minibatches. This is why AGPT's partitioned training has worked.
  - With the tree removed, the grouping alone costs about two thirds of the
    suffix tree's pd=1 gap.
  - Check any new partitioning's target skew, and its gradient cosine within
    units against across units.
- **Under full-batch training, the tree is bookkeeping.**
  - For a given f_θ, the exact gradient is the same whichever tree computes
    it.
  - Coherence falling near a stationary point partly just means convergence.
    Coherence matters for partitioned updates applied in sequence.
- **When training on samples, mass belongs in the forward pass, not the
  gradient.** Sampling already supplies the mass, and the trust-weighted
  overlay variant counted it twice (closure memo on
  `exp/linear-recurrence-final`).
