require "./executable_node"

# Executable graph — topologically sorted network of ExecutableNodes.
# Runs forward and backward passes by walking the graph in order,
# passing data along edges.

module ConstructionKit

  struct GraphEdge
    getter from_node : String
    getter from_port : String
    getter to_node : String
    getter to_port : String

    def initialize(@from_node, @from_port, @to_node, @to_port)
    end
  end

  class ExecutableGraph
    getter nodes : Hash(String, ExecutableNode)
    getter edges : Array(GraphEdge)
    getter topo_order : Array(String)       # node IDs in forward order
    getter loss_node_id : String?

    getter warnings : Array(String) = [] of String

    def initialize
      @nodes = {} of String => ExecutableNode
      @edges = [] of GraphEdge
      @topo_order = [] of String
    end

    def add_node(node : ExecutableNode)
      @nodes[node.id] = node
    end

    def add_edge(from_node : String, from_port : String, to_node : String, to_port : String)
      @edges << GraphEdge.new(from_node, from_port, to_node, to_port)
    end

    # Compute topological order using Kahn's algorithm
    def compute_topo_order!
      # Build adjacency: count incoming edges per node (skip boundary edges)
      in_degree = {} of String => Int32
      @nodes.each_key { |id| in_degree[id] = 0 }
      @edges.each do |e|
        # Only count edges from real nodes, not boundary
        next if e.from_node == "__boundary__"
        next unless @nodes.has_key?(e.to_node)
        in_degree[e.to_node] = (in_degree[e.to_node]? || 0) + 1
      end

      queue = [] of String
      in_degree.each { |id, deg| queue << id if deg == 0 }

      @topo_order = [] of String
      while !queue.empty?
        id = queue.shift
        @topo_order << id
        @edges.each do |e|
          next unless e.from_node == id
          next if e.to_node == "__boundary__"
          next unless in_degree.has_key?(e.to_node)
          in_degree[e.to_node] -= 1
          queue << e.to_node if in_degree[e.to_node] == 0
        end
      end

      if @topo_order.size != @nodes.size
        raise "Graph has a cycle (#{@topo_order.size} sorted vs #{@nodes.size} nodes)"
      end

      # Find loss node
      @loss_node_id = @topo_order.find { |id| @nodes[id].type == "loss" }
    end

    # ── Forward Pass ──────────────────────────────────────────────────────────

    def forward(boundary_inputs : Hash(String, Tensor)) : Float64
      activations = {} of String => Hash(String, Tensor)

      # Boundary inputs are keyed by port name (e.g., "in" for token IDs, "targets")
      activations["__boundary__"] = boundary_inputs

      @topo_order.each do |node_id|
        node = @nodes[node_id]

        # Gather inputs from incoming edges
        inputs = {} of String => Tensor
        @edges.each do |e|
          next unless e.to_node == node_id
          source_acts = activations[e.from_node]?
          if source_acts && (val = source_acts[e.from_port]?)
            inputs[e.to_port] = val
          end
        end

        # Skip nodes with missing required inputs
        begin
          outputs = node.forward(inputs)
          activations[node_id] = outputs
        rescue ex : KeyError
          msg = "Node #{node_id} (#{node.type}): missing input #{ex.message}"
          STDERR.puts "Graph forward: #{msg}"
          @warnings << msg unless @warnings.includes?(msg)
          activations[node_id] = {} of String => Tensor
        end
      end

      # Return loss value
      if lid = @loss_node_id
        node = @nodes[lid]
        if node.is_a?(MathLossExec)
          node.loss
        elsif node.is_a?(LossExec)
          node.loss
        else
          0.0
        end
      else
        0.0
      end
    end

    # ── Backward Pass ─────────────────────────────────────────────────────────

    def backward(lr : Float64)
      # Gradient accumulator: node_id -> port_id -> gradient Mat
      grad_accum = {} of String => Hash(String, MicroGPT::Mat)

      # Walk in reverse topological order
      @topo_order.reverse_each do |node_id|
        node = @nodes[node_id]

        # Gather output gradients for this node
        output_grads = grad_accum[node_id]? || {} of String => MicroGPT::Mat

        # Compute input gradients
        input_grads = node.backward(output_grads)

        # Update parameters
        node.update(lr)

        # Distribute input gradients to upstream nodes via edges
        input_grads.each do |port_id, grad|
          @edges.each do |e|
            next unless e.to_node == node_id && e.to_port == port_id
            next if e.from_node == "__boundary__"

            upstream_id = e.from_node
            upstream_port = e.from_port

            grad_accum[upstream_id] ||= {} of String => MicroGPT::Mat
            if existing = grad_accum[upstream_id][upstream_port]?
              # Fan-out: sum gradients from multiple consumers
              existing.add!(grad)
            else
              grad_accum[upstream_id][upstream_port] = grad
            end
          end
        end
      end
    end

    # ── Train Step ────────────────────────────────────────────────────────────

    def train_step(input_ids : Array(Int32), target_ids : Array(Int32), lr : Float64) : Float64
      # Provide boundary inputs under all possible port names so edges can match
      seq_len = input_ids.size
      # Zero stream for models without explicit stream initialization
      # Infer stream_dim from the first stream edge's target node, default 64
      zero_stream = MicroGPT::Mat.new(seq_len, 64)
      boundary = {
        "in"         => input_ids.as(Tensor),
        "input_ids"  => input_ids.as(Tensor),
        "token_ids"  => input_ids.as(Tensor),
        "targets"    => target_ids.as(Tensor),
        "target_ids" => target_ids.as(Tensor),
        "stream_in"  => zero_stream.as(Tensor),
        "stream_out" => zero_stream.as(Tensor),
      }
      loss = forward(boundary)
      backward(lr)
      loss
    end

    # ── Weight Management ─────────────────────────────────────────────────────

    def all_weight_mats : Array(MicroGPT::Mat)
      @topo_order.flat_map { |id| @nodes[id].weight_mats }
    end

    def all_adam_mats : Array(MicroGPT::Mat)
      @topo_order.flat_map { |id| @nodes[id].adam_mats }
    end

    def total_params : Int64
      @topo_order.sum { |id| @nodes[id].param_count }
    end

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(seed_ids : Array(Int32), max_tokens : Int32, temperature : Float64, seq_len : Int32) : Array(Int32)
      ids = seed_ids.dup
      max_tokens.times do
        # Use last seq_len tokens
        window = ids.size > seq_len ? ids[-seq_len..] : ids
        # Forward through all nodes except loss
        activations = {} of String => Hash(String, Tensor)
        activations["__boundary__"] = {
          "in" => window.as(Tensor),
          "input_ids" => window.as(Tensor),
          "token_ids" => window.as(Tensor),
        }

        logits_mat = nil
        @topo_order.each do |node_id|
          node = @nodes[node_id]
          next if node.type == "loss"

          inputs = {} of String => Tensor
          @edges.each do |e|
            next unless e.to_node == node_id
            source_acts = activations[e.from_node]?
            if source_acts && (val = source_acts[e.from_port]?)
              inputs[e.to_port] = val
            end
          end

          outputs = node.forward(inputs)
          activations[node_id] = outputs

          # Track the last Mat output (should be logits from output_head)
          if node.type == "output_head"
            logits_mat = outputs["out"]?.as?(MicroGPT::Mat)
          end
        end

        break unless logits_mat

        # Sample from last position
        logits = logits_mat.not_nil!
        last_row = logits.rows - 1
        vocab_size = logits.cols

        # Temperature scaling + softmax
        max_val = -Float32::INFINITY
        vocab_size.times { |j| max_val = logits[last_row, j] if logits[last_row, j] > max_val }

        probs = Array(Float64).new(vocab_size, 0.0)
        sum = 0.0
        vocab_size.times do |j|
          p = Math.exp((logits[last_row, j] - max_val) / temperature)
          probs[j] = p
          sum += p
        end
        probs.map! { |p| p / sum }

        # Multinomial sample
        r = Random.rand
        cumulative = 0.0
        next_token = vocab_size - 1
        probs.each_with_index do |p, j|
          cumulative += p
          if r <= cumulative
            next_token = j
            break
          end
        end

        ids << next_token
      end

      ids
    end
  end
end
