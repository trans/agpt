# AGPT — a trie-shaped language-model framework.
#
# The whole architecture in one page. The cleverness lives in the geometry
# we worked out (count-weighted CE = corpus NLL; the head's Fisher is the
# optimizer). The code is deliberately boring: a tree walk around one hole.
#
#   trie    → corpus turned into nodes, edges, counts
#   forward → walk from the root, call f_θ at each node, memoize state
#   head    → softmax(W_o · h_p)              (fixed; where the exactness lives)
#   loss    → Σ counts_p · CE(counts_p, π_p)  (= corpus NLL)
#   f_θ     → the one hole. A proc. Everything else ignores what you plug in.
#
# Numerics here are plain Float64 arrays so the structure stays unobscured —
# swap `Lin` for num.cr / your tensor backend. Not compiled in this sandbox.

alias Token = Int32
alias Vec   = Array(Float64)

# ── boring numeric core (placeholder — swap for your backend) ───────────
module Lin
  extend self

  def dot(a : Vec, b : Vec) : Float64
    a.zip(b).sum { |(x, y)| x * y }
  end

  def matvec(m : Array(Vec), v : Vec) : Vec
    m.map { |row| dot(row, v) }
  end

  def add(a : Vec, b : Vec) : Vec
    a.zip(b).map { |(x, y)| x + y }
  end

  def scale(v : Vec, s : Float64) : Vec
    v.map { |x| x * s }
  end

  def tanh(v : Vec) : Vec
    v.map { |x| Math.tanh(x) }
  end

  def softmax(v : Vec) : Vec
    m = v.max
    e = v.map { |x| Math.exp(x - m) }
    s = e.sum
    e.map { |x| x / s }
  end
end

# ── the trie: corpus → nodes, edges, counts ─────────────────────────────
class Node
  getter token : Token                          # edge label leading into this node
  getter children = {} of Token => Node
  getter counts   = Hash(Token, Int32).new(0)   # next-token counts at this prefix
  property state  = [] of Float64               # memoized h_p (filled by `forward`)

  def initialize(@token : Token); end

  def insert(window : Array(Token)) : Nil
    node = self
    window.each do |x|
      node.counts[x] += 1
      node = (node.children[x] ||= Node.new(x))
    end
  end
end

def build_trie(corpus : Array(Token), depth : Int32) : Node
  root = Node.new(-1)                            # root = the empty prefix
  (0...corpus.size).each { |i| root.insert(corpus[i, depth]) }
  root
end

# ── the one hole ────────────────────────────────────────────────────────
# f_θ : (ancestor states root…parent, incoming token) → this node's state.
# A pure function of the prefix, so every node memoizes.
alias Fθ = Proc(Array(Vec), Token, Vec)

# ── framework: walk the trie, thread states. The stack IS the path. ─────
def forward(node : Node, path : Array(Vec), fθ : Fθ) : Nil
  node.state = fθ.call(path, node.token)         # ancestors = root…parent
  path.push(node.state)
  node.children.each_value { |c| forward(c, path, fθ) }
  path.pop
end

# ── fixed head + objective:  Σ counts · CE  =  corpus NLL ───────────────
def nll(node : Node, w_o : Array(Vec)) : Float64
  pi   = Lin.softmax(Lin.matvec(w_o, node.state))
  here = node.counts.sum(0.0) { |x, c| -c * Math.log(pi[x]) }
  here + node.children.values.sum(0.0) { |c| nll(c, w_o) }
end

# ── f_θ #1: a vanilla Elman cell. Reads only the parent (path.last). ────
def elman(w_h : Array(Vec), w_x : Array(Vec), b : Vec, emb : Array(Vec)) : Fθ
  zero = b.map { 0.0 }
  ->(path : Array(Vec), x : Token) {
    h_p = path.empty? ? zero : path.last         # root has no parent
    e   = x < 0 ? zero : emb[x]                  # root has no token
    Lin.tanh(Lin.add(Lin.add(Lin.matvec(w_h, h_p), Lin.matvec(w_x, e)), b))
  }
end

# ── f_θ #2: attention over the whole ancestor path. Same framework. ─────
def attn(w_q : Array(Vec), w_k : Array(Vec), w_v : Array(Vec), emb : Array(Vec)) : Fθ
  ->(path : Array(Vec), x : Token) {
    e = x < 0 ? emb[0].map { 0.0 } : emb[x]
    if path.empty?
      e                                          # root (illustrative; project to model dim)
    else
      q   = Lin.matvec(w_q, e)
      wts = Lin.softmax(path.map { |h| Lin.dot(q, Lin.matvec(w_k, h)) })
      out = Lin.matvec(w_v, path.first).map { 0.0 }
      path.each_with_index { |h, i| out = Lin.add(out, Lin.scale(Lin.matvec(w_v, h), wts[i])) }
      out
    end
  }
end

# ── putting it together — swap one line, the framework is identical ─────
corpus = [] of Token                             # your token ids
root   = build_trie(corpus, depth: 11)           # D ≈ the corpus MI horizon

fθ = elman(w_h, w_x, b, emb)                      # ← this line …
# fθ = attn(w_q, w_k, w_v, emb)                   #   … or this. Nothing else moves.

forward(root, [] of Vec, fθ)
loss = nll(root, w_o)

# Training attaches here and nowhere else: gradients flow through `fθ` and
# `w_o` via your autodiff backend. The metric — Jᵀ(diag(π) − ππᵀ)J plus
# damping — is a property of `nll` (the head), not of `fθ`. That's why the
# hole is free: the optimizer lives downstream of every choice you can make.
