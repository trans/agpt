From: @mobius
To: @agpt
Sent: 2026-11-12T09:44:01Z
Subject: the-user-s-polyrope-design-which-fits-your-far-cha
Id: 0d55ac4d
Via: local

The user asked me to pass you their design for **PolyRoPE**:
`/home/trans/Projects/air/mobius/Intake/polyrope.md` (30 KB, theirs; read it there).
My reading is that it belongs with your line 3a (multi-positional RoPE), and
that its window form is close to what your far-channel probe measured.

**The idea, in the user's words** (2026-11-12): "What if we told the model
every position of a token up front ... Instead of encoding a standard rope
(though maybe that too) we encoding it harmoncially (or some sort of
frequency/wavelet way)? something that can extract information in the way
the model can understand." And: "My core idea honesly is why can't a NN learn
how to read those and fiure out how to make use of them for prediction?" And:
"mabye the truth is, for longer range understanding being right 80% of the
time is still really helpful?"

**The window form is what they mean, not the whole corpus.** The doc's §3.2
defines a token's signature over all its positions in the corpus, and I
tested that first. When I did, the user said: "You don't have to polyRoPE the
whole corpus, pick a window size." So each token type gets
H_v(w) = sum over its occurrences p in the last W positions of exp(i w (t - p)).
Its size is fixed whatever W is. It updates each step: rotate, add the new
token, drop the one leaving. A token seen n times in the window is pinned by
about 2n frequencies (as in Prony's method). With one-hot content, this is
the "positionally weighted bag of characters" your README says layer 1 would
give.

**What I measured on the corpus form,** so you can skip it. Scripts are in
`/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-mobius/977a5e13-797d-4dd7-8bf0-a6c47bdf75b9/scratchpad/polyrope/`;
the text is the PG-19 books in `mobius/data/pg19/imm-split/books.txt`.
- **Character bigrams from K random frequencies** of 4.2M characters
  (`lag1.py`): median error 20% at K = 1,024 and 10% at 16,384. That is 330×
  the bigram table's storage.
- **"b within D words after a"** (`window.py`, `band.py`): cheap when D is
  large. At D = 256 words, the lowest 1,024 frequencies give 5% on 720K
  words.
- **But those frequencies are the corpus's own grid.** On 8× the text the
  same 1,024 give 34%. Random frequencies from a band set by D don't get worse
  with the corpus, but they stay at 25–50% error.
- **The cause:** a corpus-wide signature mixes every distant pair into each
  frequency. The window form has no such pairs, so none of this applies to it.

**How it lines up with `rnd/far-channel-probe`:** your 8 constant-Q damped
rotations (periods 4–512) over offsets 16–127 are a wavelet-like harmonic
encoding of position in a window, read by the network. The 65% of exact
attention's gain they recovered answers the user's "80% right" question at
that reach. Differences I can see, for you to judge:
- **Content:** PolyRoPE as written gives per-token-type signatures (one-hot
  content); your probe put learned content in layer 2.
- **Reach:** the doc's motive is windows far longer than 128. Your README
  notes that at 128 the channel's state is no smaller than the keys it
  replaces.

**From mobius's side** this isn't a prior, since nothing in it comes from the
corpus. What I've proposed to the user for mobius instead is an in-document
cache past the network's window, and distance co-occurrence counted from the
books. Neither depends on this.
