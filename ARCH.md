# Cooperative Micro-Model Architecture with Shared Stream Communication
## A Framework for Scalable Distributed Language Model Inference and Continual Learning

**Thomas Sawyer** · Draft 1.0

---

## Abstract

We propose a novel architecture for large language model deployment combining four cooperative systems: (1) an ensemble of small specialized language models (μGPTs) communicating through a narrow shared stream interface, (2) a stateful router operating exclusively on the shared stream, (3) a client-distributed inference and training system with adaptive pulse-based backpressure, and (4) a hierarchical efficiency stack targeting effective parameter capacities far exceeding physical model size. Together these systems suggest a path toward 7T effective parameter capacity from modest hardware, while remaining practically deployable at small scale. The architecture is novel in that cooperation, modularity, and distributed compute are designed in from first principles rather than patched onto a monolithic base.

---

## 1. Introduction

Scaling language models has historically meant scaling monolithic parameter counts. This approach faces hard physical limits: memory bandwidth, GPU VRAM, training stability, and inference cost all grow with model size in ways that compound unfavorably. Several partial solutions exist — quantization, mixture of experts, speculative decoding, federated learning — but these are typically applied independently to monolithic architectures not designed to receive them.

We propose instead to start from the constraints and design upward. If the deployment target is a distributed system with heterogeneous client hardware, variable connectivity, and a need for continual learning, the architecture should reflect that from layer zero.

The key contributions of this paper are:

- A **μGPT cooperative ensemble** architecture where small parallel models communicate only through a constrained shared stream
- A **stream-gated router** that makes routing decisions based solely on accumulated stream state, not raw input
- An **adaptive pulse system** for distributed client-side training contribution with server-side backpressure
- An **efficiency compounding analysis** showing how quantization, sparsity, speculative decoding, and client compute multiply to yield effective capacities orders of magnitude beyond physical parameter count

---

## 2. Background

### 2.1 Mixture of Experts

Standard MoE architectures route tokens to one or more expert sub-networks, activating only a fraction of total parameters per forward pass. This yields sparse activation — a 70B MoE model may activate only 7B parameters per token. The routing mechanism typically operates on raw input representations, making routing decisions reactive rather than stateful. Expert collapse (a small subset of experts handling most tokens) is a known failure mode addressed with auxiliary load-balancing loss terms.

### 2.2 Federated and Continual Learning

Federated learning distributes training across client devices, aggregating gradient updates without centralizing raw data. Continual learning addresses catastrophic forgetting when models are updated incrementally. Both fields have developed largely independently of inference architecture design.

### 2.3 Speculative Decoding

Speculative decoding uses a small draft model to propose token sequences which a larger verifier model accepts or rejects in parallel. This yields 2–4× throughput improvements with no quality loss. The draft model is typically a smaller version of the same architecture running on the same hardware.

### 2.4 LoRA and Parameter-Efficient Fine-Tuning

Low-Rank Adaptation (LoRA) constrains weight updates to low-rank decompositions, reducing trainable parameters by 99%+ while preserving adaptation quality. This makes per-interaction gradient steps computationally feasible.

---

## 3. The μGPT Cooperative Ensemble

### 3.1 Motivation

The μGPT ensemble emerges from a simple question: what happens if you dissect a language model into modular components, force each to specialize, but allow them to cooperate through a minimal shared interface? Rather than one model handling all competencies, N small models handle subsets — syntax, factual recall, reasoning, style, domain knowledge — while remaining aware of each other's state through a narrow communication channel.

This is not standard MoE. The critical differences are:

1. μGPTs run **in parallel**, not as alternatives to each other
2. Communication happens through a **persistent shared stream**, not just routing weights
3. The router is **input-blind**, seeing only stream state
4. Each μGPT is **independently deployable and replaceable**

### 3.2 Architecture

Each μGPT is a small transformer (targeting 1–7B parameters) that takes two inputs: the token sequence and the current shared stream state. It produces two outputs: a contribution to the final output distribution and a delta to the shared stream.
```
input sequence x
       │
       ├──────────────────────────────────┐
       │                                  │
   μGPT_1(x, s) → (o_1, Δs_1)       μGPT_N(x, s) → (o_N, Δs_N)
       │                                  │
       └──────────────┬───────────────────┘
                      │
              aggregate outputs:
              o = Σ w_i · o_i          (router-weighted)
              s' = s + Σ Δs_i          (stream update)
```

The shared stream `s` is a vector of dimension `d`, where `d << hidden_dim`. Suggested range: `d = 64–256`. This constraint forces experts to compress their state into a bottleneck representation — they must summarize rather than broadcast.

### 3.3 The Shared Stream as Working Memory

The stream persists across forward passes within a session, giving the system a form of working memory that survives individual token generation steps. This has several implications:

- Early tokens can influence routing decisions for later tokens through stream accumulation
- The system develops session-level state without explicit memory mechanisms
- Incremental training (Section 5) can target the stream preferentially, as its small `d` allows fast gradient updates

### 3.4 Structured Stream

A flat `d`-dimensional vector risks experts interfering with each other's writes. A structured stream — `k` slots of dimension `d/k` — allows experts to write to named regions, reducing interference:
```
stream = [slot_0 | slot_1 | ... | slot_k]
each μGPT writes preferentially to assigned slot(s)
router attends over all slots
```

This adds an inductive bias that experts maintain distinct communication channels while still allowing cross-slot attention in the router. The optimal `k` and slot assignment strategy requires empirical investigation (see Section 8).

---

## 4. The Stream-Gated Router

### 4.1 Input-Blind Routing

Standard MoE routers see the input token representation and decide which experts to activate. We propose a router that sees **only the shared stream state**. This seemingly restrictive design has important properties:

- Routing is **stateful**: decisions reflect accumulated session context, not just current token
- Routing is **decoupled from input encoding**: the stream becomes the sole coordination surface
- Experts must influence future routing by writing useful signal to the stream, creating an implicit incentive for cooperation

### 4.2 Router Mechanics
```
router:  p(μGPT_i | s) = softmax(W_r · s + b_r)
weights: w_i = p(μGPT_i | s)
output:  o = Σ w_i · o_i
```

The router is a small linear layer over stream state. Its simplicity is intentional — complexity belongs in the experts, not the coordinator.

### 4.3 Routing Stability

Because the router is input-blind, it cannot react to token-level surprises. This is a feature in normal operation (stable, stateful routing) but may be a limitation for inputs that require sharp expert switches. A small residual connection from input to routing logits — weighted by a learned scalar near zero at initialization — could provide an escape valve without compromising the stateful character of routing under normal conditions.

---

## 5. Adaptive Pulse: Distributed Training

### 5.1 Interaction as Training Signal

Each client interaction carries implicit training signal: which completions were accepted, where corrections were made, dwell time, explicit feedback. Rather than discarding this signal or batching it for centralized training runs, the pulse system harvests it continuously.

Each interaction produces:
- A **gradient signal scalar** (or small vector): how surprised was the model by the user's actual behavior?
- A **LoRA adapter delta**: the weight update implied by one gradient step

### 5.2 Client-Side Pre-Computation

To minimize server load, clients pre-compute what they can:

| Operation | Location | Cost |
|---|---|---|
| Tokenization | Client | Negligible |
| Embedding lookup | Client (cached) | ~zero after warmup |
| Attention mask | Client | Negligible |
| Loss signal | Client (with local draft model) | Low |
| LoRA delta | Client (with adapter snapshot) | Medium |
| Full forward pass | Server | High |

The embedding table is frozen and CDN-cacheable. Clients hold a local copy after first fetch. The LoRA adapter snapshot (~20–40 MB at rank 8) is periodically synced from the server.

### 5.3 The Pulse Mechanism

The server embeds a `pulse` parameter in each response:
```
response metadata: { pulse: N, adapter_version: K }

client:
  interaction_count++
  if interaction_count % pulse == 0:
    upload(gradient_signal, lora_delta, adapter_version)
```

The server adjusts `pulse` based on queue depth:
```
queue_depth < LOW_WATERMARK  → pulse = 1   (train every interaction)
queue_depth < MID_WATERMARK  → pulse = 5
queue_depth < HIGH_WATERMARK → pulse = 10
queue_depth > HIGH_WATERMARK → pulse = 50  (inference only)
```

Clients never need to understand the backpressure mechanism — they observe only a changing `N`.

### 5.4 Gradient Aggregation

Client deltas are processed through a serialized queue. FedAvg over a window of K clients produces each new adapter checkpoint:
```
new_adapter = (1/K) Σ client_delta_i
```

Serialization sidesteps gradient staleness — updates are applied in order, not merged from simultaneous states. Queue throughput, not correctness, is the scaling bottleneck.

Robust aggregation (trimmed mean, coordinate-wise median) should be applied to defend against gradient poisoning from adversarial clients.

---

## 6. Speculative Decoding at the Client Boundary

### 6.1 Client as Draft Model Host

The client-side draft model (targeting ~500M–1B parameters, ~250–500 MB INT4) serves dual purpose: it computes the loss signal for training contribution, and it proposes token sequences for speculative decoding.
```
client draft model → proposed tokens [t_1, t_2, ..., t_k]
server verifier   → accept/reject each token in parallel
accepted tokens   → returned to client immediately
first rejection   → server generates correct token, continues
```

This offloads 30–50% of generation compute to client hardware with no quality loss.

### 6.2 Heterogeneous Client Handling

Not all clients can run a draft model. The pulse system naturally handles this:

- Capable clients: run draft model, contribute speculative tokens + gradient signal
- Incapable clients: receive full server inference, contribute gradient signal only
- Minimal clients: receive full server inference, no training contribution

The server treats all three identically at the inference level. Training contribution is opportunistic.

---

## 7. Efficiency Compounding and Effective Scale

### 7.1 The Multiplier Stack

Starting from a 7B physical model, efficiency techniques compound:

| Technique | Multiplier | Notes |
|---|---|---|
| INT4 quantization | ×4 | Minimal quality loss |
| Kernel fusion / IO optimization | ×2 | Memory bandwidth bound |
| MoE sparsity (10 experts, 1 active) | ×10 | Sparse activation |
| Speculative decoding | ×3 | Client draft model |
| Client-side compute | ×2–5 | Embedding + prefill |

Conservative product: **7B × 4 × 2 × 10 × 3 × 2 = ~3.4T effective capacity**

With hardware scale to 70B physical and the full efficiency stack: **~34T effective capacity**. The 7T target is conservative within this analysis.

### 7.2 Distinction: Capacity vs Quality

Effective capacity is not the same as parameter quality. A 700B effective capacity system built from 7B physical parameters will not match a true 700B monolithic model on all benchmarks. The claim is throughput and coverage — the system can serve more concurrent sessions, handle more diverse queries through expert specialization, and adapt faster through continual learning. Quality per token, for a given expert, remains bounded by that expert's physical size.

### 7.3 The Cooperation Dividend

The μGPT ensemble offers an additional multiplier not captured above: **parallelism**. In a monolithic model, layers are sequential. In a μGPT ensemble, experts run simultaneously. Wall-clock latency for a 7×1B ensemble is closer to a single 1B model than a 7B model, while quality approaches the latter. This is the cooperation dividend — the architecture gets depth-of-knowledge from breadth-of-specialization at shallow-model latency.

---

## 8. Open Questions and Future Work

Several critical design questions require empirical investigation:

**Stream dimensionality** — What is the minimum `d` that preserves coordination quality? Is there a phase transition below which expert cooperation collapses?

**Slot structure** — Does structured stream (k named slots) outperform flat stream? What slot assignment strategy works best — fixed, learned, or dynamic?

**Router residual** — How much input signal should leak into routing? Is the learned scalar escape valve sufficient, or does full input-blind routing cause pathological behavior on distribution-shift inputs?

**Training convergence** — How many more training steps does the cooperative ensemble require versus a monolithic baseline of equivalent parameter count? Early experiments should measure this directly.

**Expert collapse under pulse training** — Does intermittent gradient signal (high pulse values) cause expert drift or collapse? Load-balancing loss terms from standard MoE literature may need adaptation for the stateful router setting.

**Stream persistence scope** — Should the stream reset between sessions, between conversations, or never? Persistent stream across users raises privacy concerns but might improve population-level coordination.

**Draft model architecture** — Should the client draft model be a distilled μGPT (same architecture, smaller) or a different architecture optimized for speculative proposal? The latter may yield better acceptance rates.

---

## 9. Conclusion

The architecture described here composes four independently motivated ideas — μGPT cooperation, stream-gated routing, adaptive pulse training, and efficiency compounding — into a unified system where each component reinforces the others. The pulse system feeds the training loop. The training loop updates LoRA adapters. The adapters tune the μGPTs. The μGPTs write to the stream. The stream gates the router. The router weights the cooperation dividend.

No component is novel in isolation. The contribution is the composition, and the observation that designing for distribution from first principles yields an architecture that scales more gracefully than monolithic models patched with post-hoc efficiency techniques.

The 7T effective capacity figure is a long-horizon target requiring hardware scale, engineering maturity, and resolution of the open questions above. But the path is composed entirely of solved or tractable subproblems. The architecture does not require a breakthrough — it requires careful construction.

---

## Notes on Completion

*The following sections require expansion in a full version of this paper:*

- **Section 3.4**: Empirical guidance on slot count `k` and assignment strategies
- **Section 4.2**: Full description of loss signal computation on client with draft model
- **Section 5.4**: Robust aggregation strategies and poisoning threat model
- **Section 6.1**: Detailed speculative decoding protocol and acceptance rate analysis
- **Section 7**: Benchmarks comparing effective capacity claims against monolithic baselines
- **Appendix A**: Formal description of stream update dynamics and convergence conditions
- **Appendix B**: Hardware sizing guide for various deployment scales
- **Appendix C**: Privacy analysis of client-side gradient signal

---

*Draft 1.0 — expanded from design conversation. Empirical validation pending.*
