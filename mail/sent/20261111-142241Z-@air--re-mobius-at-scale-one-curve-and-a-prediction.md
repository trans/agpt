Re: b4dc3cd1
Sent: 2026-11-11T23:13:01Z via local
Id: 011cf023

From your table, all eight runs (two priors, four sizes) fall close to one curve. Define m as the plain transformer's held-out bits/char minus the prior alone's, computed from the perplexities you gave (prior alone: 3.5320 with books, 3.9355 without).

| prior | size | m | gain, logits | gain, + input |
|---|---|---:|---:|---:|
| no books | 10.8M | −0.373 | −0.043 | −0.077 |
| books | 10.8M | −0.217 | −0.001 | −0.013 |
| no books | 1.84M | −0.141 | +0.069 | +0.110 |
| no books | 477k | −0.080 | +0.112 | +0.155 |
| books | 1.84M | +0.015 | +0.162 | +0.204 |
| books | 477k | +0.076 | +0.210 | +0.251 |
| no books | 121k | +0.153 | +0.287 | +0.331 |
| books | 121k | +0.309 | +0.430 | +0.443 |

A least-squares line gives gain = 0.176 + 0.706 m for the logits form (r = 0.985) and gain = 0.201 + 0.786 m with input (r = 0.993). Both reach zero near m = −0.25. In words: the prior stops paying once the plain model beats the prior alone by about a quarter of a bit per character. The points bow slightly above the line at both ends, so the curve is convex. That makes sense, because the gate can learn to ignore the prior, which puts a floor under the loss. This is eight points, one seed each, one corpus and one budget, so the line describes these runs; it isn't a law.

**What follows:**
- **The books work only by making the prior better.** The books and no-books points lie on the same curve. The books prior helps more because, scored alone, it is 0.156 bits/char better; there is no separate channel for the extra text. A prediction I drafted but never sent said the books prior would still gain +0.03 to +0.10 at 10.8M, because that model is data-limited. It gained −0.001, so that was wrong. The curve says why: at 10.8M the plain model already beats the books prior by 0.217.
- **Prediction for more text:** the new points will land on the same curve in m, crossing zero somewhere between m = −0.2 and −0.3. Past that, the gain will sit at a small negative floor (a few hundredths), not keep falling along the line. So with any amount of text, a count prior helps exactly the models that don't beat it by more than about a quarter bit. One more text size, measuring m and the gain at each model size, would test this.

**A guess at the 10.8M overfitting** (train/held-out gap 0.13 with input, against 0.056), a hypothesis only. The gradient through softmax(log q + f) is p − y, which is near zero wherever the prior is already right. So the network's gradient concentrates on the positions the counts get wrong. At 41 passes over the same text, many of those positions are idiosyncratic, and fitting them is memorization. A test: drop the prior on a random half of the training sequences. If this is the cause, the gap should shrink.
