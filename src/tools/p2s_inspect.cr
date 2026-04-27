# Inspect a p2s match index: sample N prefix leaves, decode their path chars
# and the chars of each candidate suffix, print human-readable matches.
#
# Used to sanity-check that the match index is producing semantically
# sensible matches before any model training begins (Phase 4 prerequisite).

require "option_parser"
require "../agpt"

prefix_dir = ""
suffix_dir = ""
match_path = ""
corpus_path = ""
n_samples = 20
seed = 42_u32

OptionParser.parse do |parser|
  parser.banner = "Usage: bin/agpt_p2s_inspect --prefix DIR --suffix DIR --match PATH --corpus PATH"
  parser.on("--prefix DIR", "Prefix radix tree dir") { |v| prefix_dir = v }
  parser.on("--suffix DIR", "Suffix radix tree dir") { |v| suffix_dir = v }
  parser.on("--match PATH", "Match index binary file") { |v| match_path = v }
  parser.on("--corpus PATH", "Corpus text file (for char vocab)") { |v| corpus_path = v }
  parser.on("-n N", "Number of samples to print (default 20)") { |v| n_samples = v.to_i }
  parser.on("--seed N", "RNG seed (default 42)") { |v| seed = v.to_u32 }
  parser.on("-h", "--help", "Help") { puts parser; exit 0 }
end

if prefix_dir.empty? || suffix_dir.empty? || match_path.empty? || corpus_path.empty?
  STDERR.puts "Error: --prefix, --suffix, --match, --corpus all required"
  exit 1
end

# Load corpus to build char vocab
text = File.read(corpus_path)
dataset = MicroGPT::CharDataset.new(text)
vocab_size = dataset.vocab_size
# Build token → char map (CharDataset's data is char codes, vocab is sorted unique chars)
chars_sorted = text.chars.uniq.sort
token_to_char = {} of Int32 => Char
chars_sorted.each_with_index do |c, i|
  token_to_char[i] = c
end

# Helper: render token sequence as a printable string
def render_tokens(tokens : Array(Int32) | Slice(Int32), token_to_char : Hash(Int32, Char)) : String
  result = String.build do |str|
    tokens.each do |t|
      c = token_to_char[t]?
      if c
        # Escape control chars to make output readable
        case c
        when '\n' then str << "\\n"
        when '\t' then str << "\\t"
        when '\r' then str << "\\r"
        else str << c
        end
      else
        str << "?"
      end
    end
  end
  result
end

# Load prefix tree, build id → (parent_id, edge_tokens) map
STDERR.puts "[inspect] Loading prefix tree from #{prefix_dir}..."
prefix_reader = MicroGPT::AGPT::RadixTrieReader.new(prefix_dir, max_cached: 4)
prefix_parent = Hash(Int32, Int32).new
prefix_edge   = Hash(Int32, Array(Int32)).new
prefix_reader.each do |r|
  prefix_parent[r.id] = r.parent_id
  prefix_edge[r.id] = r.edge_tokens
end
STDERR.puts "[inspect]   loaded #{prefix_parent.size} prefix records"

# Load suffix tree, build id → (parent_id, edge_tokens) map
STDERR.puts "[inspect] Loading suffix tree from #{suffix_dir}..."
suffix_reader = MicroGPT::AGPT::RadixTrieReader.new(suffix_dir, max_cached: 4)
suffix_parent = Hash(Int32, Int32).new
suffix_edge   = Hash(Int32, Array(Int32)).new
suffix_reader.each do |r|
  suffix_parent[r.id] = r.parent_id
  suffix_edge[r.id] = r.edge_tokens
end
STDERR.puts "[inspect]   loaded #{suffix_parent.size} suffix records"

# Helper: walk from a node id back to root, collecting tokens
def path_tokens(id : Int32, parent_map : Hash(Int32, Int32), edge_map : Hash(Int32, Array(Int32))) : Array(Int32)
  segments = [] of Array(Int32)
  cur = id
  while cur > 0
    if e = edge_map[cur]?
      segments << e
    end
    parent = parent_map[cur]?
    break unless parent
    cur = parent
  end
  segments.reverse.flat_map { |seg| seg }
end

# Read match index header
STDERR.puts "[inspect] Reading match index from #{match_path}..."
File.open(match_path, "rb") do |io|
  magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
  raise "Bad magic 0x#{magic.to_s(16)} (expected 0x50325343 'P2SC')" unless magic == 0x50325343_u32
  prefix_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
  suffix_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
  suffix_leaf_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
  STDERR.puts "[inspect]   prefix_count=#{prefix_count}, suffix_count=#{suffix_count}, suffix_leaves=#{suffix_leaf_count}"

  # First pass: count records and collect file offsets so we can sample randomly.
  STDERR.puts "[inspect] First pass: indexing match records..."
  record_offsets = [] of Int64
  while !io.peek.empty?
    record_offsets << io.pos.to_i64
    _pid = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    _maxk = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    n_sigmas = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    io.skip(n_sigmas * 4)
  end
  STDERR.puts "[inspect]   indexed #{record_offsets.size} match records"

  # Sample N random records (with seeded RNG for reproducibility)
  rng = Random.new(seed)
  picked = record_offsets.sample(n_samples, rng)
  picked = picked.sort   # stable order for output

  STDERR.puts ""
  STDERR.puts "[inspect] === Sample matches (n=#{picked.size}, seed=#{seed}) ==="
  STDERR.puts ""

  picked.each_with_index do |off, sample_idx|
    io.pos = off
    pid = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    maxk = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    n_sigmas = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    sigmas = Array(Int32).new(n_sigmas) { io.read_bytes(Int32, IO::ByteFormat::LittleEndian) }

    pi_path = path_tokens(pid, prefix_parent, prefix_edge)
    pi_str  = render_tokens(pi_path, token_to_char)

    puts "─" * 70
    puts "Sample ##{sample_idx + 1}:  prefix_leaf=#{pid}  max_k=#{maxk}  n_candidates=#{n_sigmas}"
    puts "  π full path (#{pi_path.size} chars):  «#{pi_str}»"

    # Show up to 4 candidates
    show_count = Math.min(n_sigmas, 4)
    sigmas[0...show_count].each_with_index do |sid, i|
      sigma_path = path_tokens(sid, suffix_parent, suffix_edge)
      # Suffix tree was built on REVERSED corpus. To display in ORIGINAL forward
      # order, reverse the path.
      sigma_fwd = sigma_path.reverse
      sigma_str = render_tokens(sigma_fwd, token_to_char)
      puts "  σ#{i+1} forward (#{sigma_fwd.size} chars):  «#{sigma_str}»"
      # Highlight where overlap is: σ.fwd starts with π's last max_k chars
      if maxk > 0 && pi_path.size >= maxk && sigma_fwd.size >= maxk
        pi_tail = pi_path[(pi_path.size - maxk)..]
        sg_head = sigma_fwd[0...maxk]
        match_str = render_tokens(pi_tail, token_to_char)
        if pi_tail == sg_head
          puts "       overlap=«#{match_str}» (k=#{maxk}) ✓"
        else
          puts "       OVERLAP MISMATCH: π.tail=«#{render_tokens(pi_tail, token_to_char)}» σ.head=«#{render_tokens(sg_head, token_to_char)}»"
        end
        # Show what σ predicts as the next-char (char at σ.fwd[overlap_k])
        if sigma_fwd.size > maxk
          predicted = token_to_char[sigma_fwd[maxk]]?
          if predicted
            disp = predicted == '\n' ? "\\n" : predicted == '\t' ? "\\t" : predicted.to_s
            puts "       σ predicts next-char: '#{disp}' (σ.fwd[#{maxk}])"
          end
        end
      end
    end
    puts "  ... (and #{n_sigmas - show_count} more)" if n_sigmas > show_count
  end
  puts "─" * 70
end

STDERR.puts ""
STDERR.puts "[inspect] Done."
