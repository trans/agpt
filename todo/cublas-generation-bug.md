# TODO — cuBLAS backend produces NaN logits during generation

**Status:** Reproduced 2026-05-03. Affects `bin/microgpt --backend cublas
--gen` and `--eval`. Does NOT affect AGPT training or PPL eval (both
work correctly on cuBLAS). Workaround: use `--backend openblas` for
generation only.

## Symptoms

1. `bin/microgpt --backend cublas --gen` always produces output of the
   form `<seed>BzzzzzzzzzzzzZ...` regardless of model architecture or
   training quality. Always lands on vocab[-1] = 'z'.
2. Same models on `--backend openblas` produce normal English-shaped
   output:
   - At temp 0.1: real English words (e.g., "And so it goes that"
     → "seems and the boy")
   - At temp 0.8: glue-syllables but with proper character distribution
     (caps, punctuation, occasional words)
3. `bin/perplexity --backend cublas` works correctly on the same
   models — PPL numbers match expectations and reflect real model
   quality.
4. Reproduces with both AGPT-trained models AND fresh
   `microgpt --steps 1` random init. Not model-specific.
5. Reproduces at temp=0.1 AND temp=0.8. Not a sampling-fallback issue.

## Root-cause shape

The generation fallback in `MiniGPT#generate` (micro_gpt.cr:1593) only
defaults `chosen = vocab_size - 1` when softmax probabilities don't sum
to ≥ `r` (the random sampling threshold). With NaN values in the probs,
the cumulative comparison `r <= cumulative` always evaluates false, so
the loop exhausts and the default fires.

So **the cuBLAS forward path is producing NaN logits during generation**,
but NOT during PPL eval.

## Key difference: PPL vs generate inputs

| | input length | pos range used |
|---|---|---|
| **PPL eval** | always exactly `seq_len` | 0..seq_len-1 |
| **generate** | variable: starts at 16, grows up to seq_len | 0..context.size-1 |

PPL feeds a fixed-length window matching `model.config.seq_len`.
Generate feeds variable-length contexts (initial seed of 16 chars,
growing as it appends generated tokens).

The cuBLAS path likely has a buffer-sizing assumption that breaks when
the actual input length is shorter than the model's `max_seq_len`.

## Specific suspects (in `lib/microgpt/src/microgpt/backend.cr`,
`CuBLASBackend` class, and `lib/microgpt/src/microgpt/micro_gpt.cr`):

### 1. `embedding_gather` (backend.cr:548)
```crystal
LibCUDAKernels.cuda_embedding_gather(d_token_emb, d_ids, d_output, seq_len, d_model)
```
- Receives `ids.size` as `seq_len`. Output buffer sized to
  `seq_len * d_model`. *Probably* fine.

### 2. `rope_apply` (backend.cr:~544)
```crystal
LibCUDAKernels.cuda_rope_apply(d_x, d_cos, d_sin, x.rows, x.cols)
```
- `cos_cache` was built for `max_seq` positions (typically 128).
- Kernel receives `x.rows` (= context.size) and reads
  `cos_cache[pos]` for `pos in 0..x.rows-1`. *Should* work.
- Potential issue: if x is allocated as `[max_seq, dim]` but only
  filled in `[0..x.rows, dim]`, kernel might read uninitialized data
  past x.rows — depends on the CUDA kernel's bounds.

### 3. `causal_mask` (backend.cr:920)
```crystal
LibCUDAKernels.cuda_causal_mask(d_data, scores.rows)
```
- Only takes `scores.rows` (= n), assumes the matrix is n × n square.
- For attention scores [seq_len, seq_len], this is correct.
- *Potential bug*: if `scores` is sized to `[max_seq, max_seq]` but
  only valid in `[0..seq_len, 0..seq_len]`, the kernel masks the
  wrong region (masks at seq_len boundary instead of max_seq
  boundary).

### 4. Multi-head attention forward (micro_gpt.cr:~844, AttentionMultiHeads)
- Reshapes Q, K, V for head splitting. Reshape strides may assume
  `max_seq_len` rows but data only fills `seq_len` rows.
- This is the **most likely suspect** — a reshape/view with mismatched
  strides would produce uninitialized garbage in the unused rows,
  which propagates into Q·K^T scores → softmax → attention output →
  final logits.

### 5. `fused_attn_softmax` (backend.cr:903)
```crystal
def fused_attn_softmax(scores, scale)
  d_in = scores.gpu_ptr
  d_out = Pointer(Void).null
  LibCUDA.cudaMalloc(pointerof(d_out), scores.byte_size)
  LibCUDAKernels.cuda_fused_attn_softmax(d_in, d_out.as(Float32*), scale, scores.rows, scores.cols)
  Mat.new(scores.rows, scores.cols, d_out)
end
```
- Allocates a NEW uninitialized output buffer per call.
- Kernel writes to it — but only writes the `scores.rows × scores.cols`
  portion. If the buffer is larger (e.g., `byte_size` based on
  max_seq), the unwritten part is garbage but downstream reads it.

## Diagnostic test plan

### Quick sanity-check the variable-length hypothesis
Force PPL to feed a **shorter window than the model's max_seq_len**:
```sh
# Train model with max_seq_len=128 (default). Then eval with --seq-len 32.
bin/perplexity --model ... --seq-len 32 --backend cublas
```

This currently works because perplexity sets `config.seq_len = seq_len`,
making the model object's max_seq match the eval seq. To trigger the
bug, would need to construct the model with seq_len=128 and feed it
a 32-char window.

A more direct test: instrument `MiniGPT#generate` to print the logits
at the FIRST generation step (when context is just the 16-char seed).
If logits contain NaN/Inf at any vocab position, confirm the cuBLAS
forward is broken for shorter-than-max inputs.

### Bisect by backend op

Try replacing each suspect cuBLAS op with a CPU equivalent:
1. CPU embedding gather + cuBLAS rest → does generation work?
2. CPU rope + cuBLAS rest?
3. CPU softmax + cuBLAS rest?
4. CPU attention scores + cuBLAS rest?

Whichever swap makes it work pinpoints the broken op.

## Workaround

Use `--backend openblas` for any generation/eval-prompts work. CPU-only
but produces correct output. Slow but functional.

For PPL evaluation, cuBLAS works correctly — keep using it.

## Effort estimate

- Diagnose with bisect: 1-2 hours
- Likely fix: 30 min - few hours depending on which op is the bug
- Test + verify: 1 hour
- Total: half day to a day

## Why this hasn't bitten us harder

We've been doing PPL evaluation throughout the project, and PPL works
on cuBLAS. The first time we needed generation was 2026-05-03 when
comparing model quality between architectures. So the bug has been
latent in the cuBLAS path likely for the entire project lifetime.
