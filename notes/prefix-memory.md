# AGPT Prefix Trie System — Formal Implementation Spec

## Core Concept

We construct a **depth-layered prefix trie** with:

1. **Per-depth storage (RAM + disk via mmap)**
2. **Checkpointed latent state (vector) storage**

The system trades **compute for memory** by:
- storing only selected latent states
- reconstructing others via forward replay

---

# 🧠 Stored Vector Definition

For each prefix at depth `d`, define:

```
h_d ∈ ℝ^{d_model}
```

Where:
- `h_d` = **final residual stream (last hidden state)** after processing the prefix
- This is:
  - NOT embeddings
  - NOT logits
  - NOT KV cache

👉 It is the model’s **internal representation of the prefix**

---

# ⚙️ Attack Surface 1: Per-Depth Storage

## Structure

For each depth `d`, maintain:

```
DepthFile[d]:
  NodeRecords[]
```

### NodeRecord

```
NodeRecord {
  parent_id: u32
  token: u8 | u16
  freq: u32              // optional
  vector_offset: u64     // 0/null if not checkpointed
}
```

---

## Execution Model

Process strictly **by depth**:

```
for d in 0..D:
  load DepthFile[d]
  build DepthFile[d+1]
  optionally release DepthFile[d-1] from RAM
```

---

## RAM Invariant

At any time, RAM holds:

- Depth `d`
- Depth `d+1` (being built)

Everything else is disk-backed (mmap or flat files)

---

# ⚙️ Attack Surface 2: Checkpointed Vector Storage

## Checkpoint Rule

Let checkpoint interval = `k`

Store vectors only when:

```
d % k == 0
```

---

## Storage Logic

```
if d % k == 0:
  store h_d in VectorFile
  node.vector_offset = pointer
else:
  node.vector_offset = null
```

---

## Reconstruction Logic

To compute `h_d`:

1. Find nearest checkpoint:

```
d0 = k * floor(d / k)
```

2. Load `h_d0`

3. Replay forward:

```
h = h_d0
for i in (d0+1 .. d):
  h = F(h, token_i)
```

---

# 🔁 Transformer Step

```
h_{i+1} = F(h_i, x_{i+1})
```

Where:
- `F` = forward pass step of transformer
- uses **ephemeral KV cache**
- KV cache is NOT stored

---

# 📦 Vector Storage

Separate file:

```
VectorFile:
  contiguous array of vectors
```

Vector format:
- float16 (recommended)
- size = `d_model`

---

# 🔁 Combined Behavior

## Memory Reduction

| Component        | Before          | After             |
|-----------------|----------------|------------------|
| Depth storage    | all depths      | 1–2 depths       |
| Vector storage   | every node      | every k-th node  |

Expected reduction:

```
≈ 2× (depth) × 2× (checkpoint) ≈ 4×
```

---

## Compute Tradeoff

Reconstruction cost:

```
O(k) forward steps per lookup
```

---

# 🧠 Mental Model

- Trie = symbolic structure
- Vector = latent state summary
- Checkpoints = keyframes
- Replay = reconstruction

---

# 🧪 Recommended Initial Parameters

```
checkpoint_interval k = 2
vector_type = float16
active_depths_in_ram = 2
```

---

# ⚠️ Constraints

## DO

- store vectors only at checkpoint depths
- recompute intermediate states via replay
- keep KV cache ephemeral
- process strictly by depth

## DO NOT

- store KV cache
- store vectors at every node
- rely on random cross-depth access

---

# 🧠 Final Summary

Store the last-layer hidden state only at checkpoint depths, reconstruct intermediate states via forward replay, and process the trie one depth at a time using disk-backed storage.
