Findings

  1. High: the cuBLAS backend suppresses almost all CUDA/cuBLAS failures in the forward path, so a shape-specific generation bug can silently
     turn into garbage/NaN logits instead of throwing. The most suspicious forward op is transpose, because attention calls k.t on every
     generation step, and transpose uses cublasSgeam with no status check at all in lib/microgpt/src/microgpt/backend.cr:702. More generally,
     cudaMalloc, cudaMemcpy, and many kernel wrappers ignore return codes in lib/microgpt/src/microgpt/backend.cr:618, lib/microgpt/src/microgpt/
     backend.cr:628, lib/microgpt/src/microgpt/backend.cr:632, lib/microgpt/src/microgpt/backend.cr:887, and lib/microgpt/src/microgpt/
     backend.cr:903. matmul is checked; most of the rest is not. If cublasSgeam or a CUDA kernel is the thing that breaks on short variable
     contexts, the current codebase will not tell you.
  2. High: generation masks backend corruption as a sampling result. In lib/microgpt/src/microgpt/micro_gpt.cr:1588, softmax_rows is called on
     the last-row logits, and then sampling defaults to chosen = vocab_size - 1 if the cumulative sum never crosses r in lib/microgpt/src/
     microgpt/micro_gpt.cr:1593. With NaN probabilities, every r <= cumulative comparison is false, so the backend bug presents as “always pick
     last vocab entry,” which matches the z symptom exactly. This is not the root cause, but it is why the bug looks like a model issue instead
     of a forward-path failure.
  3. Medium: bin/perplexity is not actually a clean negative control for the suspected bug shape. It rebuilds the model with config.seq_len =
     seq_len in lib/microgpt/src/tools/perplexity.cr:107 and then always feeds exact-length windows in lib/microgpt/src/tools/perplexity.cr:142.
     So “perplexity works on cuBLAS” does not disprove “forward on shorter-than-model contexts is broken.” It only proves the fixed-window case
     works.
  4. Medium: the most suspicious implementation point for the actual corruption is the non-square attention path, especially transpose in lib/
     microgpt/src/microgpt/backend.cr:702. That is an inference from the code and your note, not a proven bug from this environment. The reason
     I’m pointing at it is: generation starts with a 16-char seed and then immediately moves through many short, changing context lengths;
     attention uses q * k.t in lib/microgpt/src/microgpt/micro_gpt.cr:758; and transpose is one of the few shape-sensitive forward ops that is
     both heavily used and completely unchecked.
  5. Low: the batched attention fast path is impossible to enter because the condition is self-contradictory: @heads.size > 1 && @heads.size <= 0
     in lib/microgpt/src/microgpt/micro_gpt.cr:860 and again in lib/microgpt/src/microgpt/micro_gpt.cr:882. That is unrelated to the current bug
     because the dead path is never used, but it is a real bug.

  Open questions / limits

  - I could not reproduce locally because cuBLAS fails to initialize in this environment, so this is a static review.
  - I do think your note’s “variable-length generate vs fixed-length eval” hypothesis is directionally right.
  - If I had to bet on one code area first, I would inspect the unchecked transpose/attention forward path before the simpler kernels like
    embedding gather or RoPE.

  The highest-signal takeaway is: this still looks like a backend-forward bug, but the code currently makes it much harder to localize because it
  swallows the failure and then samples z.

