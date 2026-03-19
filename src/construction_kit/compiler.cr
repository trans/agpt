require "./graph"
require "./executable_graph"
require "./math_nodes"

# Compiler: translates a visual GraphData into an ExecutableGraph.
# Bottom-up approach: walks to the lowest level of the graph (math primitives)
# and creates ExecutableNodes that call the backend directly.
# Containers are transparent — the compiler recurses into their children.

module ConstructionKit
  class Compiler
    getter vocab_size : Int32
    getter seq_len : Int32
    getter lr : Float64

    def initialize(@vocab_size : Int32, @seq_len : Int32 = 128, @lr : Float64 = 0.0003)
    end

    # Compile a pipeline-level graph into an executable graph.
    def compile(graph : GraphData) : ExecutableGraph
      eg = ExecutableGraph.new

      # Recursively compile all nodes, flattening containers
      compile_scope(graph, nil, graph, eg, "")

      eg.compute_topo_order!
      # Debug: show edges to broken nodes
      STDERR.puts "Compiled: #{eg.nodes.size} nodes, #{eg.edges.size} edges"
      broken = Set{"4.885.948", "4.935.1299", "4.938.1308"}
      eg.edges.each do |e|
        if broken.includes?(e.to_node) || broken.includes?(e.from_node)
          STDERR.puts "  EDGE: #{e.from_node}:#{e.from_port} → #{e.to_node}:#{e.to_port}"
        end
      end
      eg
    end

    # Compile all nodes in a scope, resolving boundary ports to the parent.
    private def compile_scope(
      graph : GraphData,
      parent_node : Node?,
      parent_graph : GraphData,
      eg : ExecutableGraph,
      prefix : String
    )
      # First pass: compile each node
      graph.nodes.each do |node|
        node_id = make_id(prefix, node.id)

        if node.children && has_compilable_children?(node)
          # Container with compilable children — recurse
          compile_scope(node.children.not_nil!, node, graph, eg, node_id)
        elsif SKIP_TYPES.includes?(node.type)
          # Data pipeline node — skip
        else
          # Leaf node or compound fallback — create an ExecutableNode
          exec_node = create_math_node(node_id, node)
          eg.add_node(exec_node) if exec_node
        end
      end

      # Second pass: wire edges, resolving boundary ports
      graph.edges.each do |edge|
        from_id = resolve_from(edge.from, graph, parent_node, parent_graph, prefix)
        to_id = resolve_to(edge.to, graph, parent_node, parent_graph, prefix)
        next unless from_id && to_id

        # If source is a skipped node, trace back to boundary
        from_ok = from_id[0] == "__boundary__" || eg.nodes.has_key?(from_id[0])
        to_ok = to_id[0] == "__boundary__" || eg.nodes.has_key?(to_id[0])

        if !from_ok && to_ok
          traced = trace_through_skipped(edge.from, graph, prefix)
          if traced
            eg.add_edge(traced[0], traced[1], to_id[0], to_id[1])
          end
          next
        end

        next unless from_ok && to_ok
        eg.add_edge(from_id[0], from_id[1], to_id[0], to_id[1])
      end
    end

    # Types that should ALWAYS use compound fallback, never recurse into children.
    # These have unimplemented math primitives in their drill-down.
    FORCE_COMPOUND = Set{"layer_norm", "attention_layer", "global_router", "context_router", "gated_router"}

    # Does this node have children that should be recursed into?
    private def has_compilable_children?(node : Node) : Bool
      return false if FORCE_COMPOUND.includes?(node.type)
      children = node.children
      return false unless children
      return false if children.nodes.empty?
      children.nodes.any? { |n| compilable_node?(n) }
    end

    # Can this node be compiled (either directly or by recursing into children)?
    private def compilable_node?(node : Node) : Bool
      return true if math_primitive?(node.type)
      return true if node.children && has_compilable_children?(node)
      # Compound fallbacks: these types have create_math_node implementations
      # even without children
      COMPOUND_TYPES.includes?(node.type)
    end

    COMPOUND_TYPES = Set{
      "layer_norm", "attention_layer", "loss",
    }

    # Is this a math primitive type (leaf-level executable)?
    private def math_primitive?(type : String) : Bool
      MATH_TYPES.includes?(type)
    end

    MATH_TYPES = Set{
      "weight_param", "bias_param", "scale_param", "embedding_table",
      "matmul", "add_bias", "relu", "softmax", "lookup", "elem_mul",
      "layer_norm", "loss", "add", "mean_pool", "broadcast",
      "reduce_mean", "subtract", "reduce_var", "normalize",
      "attention_layer",  # compound — stays as single node
    }

    # Types that are skipped (data pipeline, not model computation)
    SKIP_TYPES = Set{
      "char_tokenizer", "bpe_tokenizer",
      "sequential_window", "sliding_window", "random_window",
      "zero_init", "random_init", "learned_init",
      "optimizer", "stream_projector",
    }

    private def make_id(prefix : String, node_id : Int32) : String
      prefix.empty? ? node_id.to_s : "#{prefix}.#{node_id}"
    end

    # Create a math-level ExecutableNode
    private def create_math_node(id : String, node : Node) : ExecutableNode?
      case node.type
      # Parameter nodes
      when "weight_param"
        rows = node.param_i("rows", 64)
        cols = node.param_i("cols", 64)
        WeightParamExec.new(id, rows, cols)

      when "bias_param"
        dim = node.param_i("dim", 64)
        BiasParamExec.new(id, dim)

      when "scale_param"
        dim = node.param_i("dim", 64)
        ScaleParamExec.new(id, dim)

      when "embedding_table"
        vocab_size = node.param_i("vocab_size", @vocab_size)
        d_model = node.param_i("d_model", 64)
        EmbeddingTableExec.new(id, vocab_size, d_model)

      # Compute nodes
      when "matmul"
        MatMulExec.new(id)

      when "add_bias"
        AddBiasExec.new(id)

      when "relu"
        ReLUExec.new(id)

      when "softmax"
        SoftmaxExec.new(id)

      when "lookup"
        LookupExec.new(id)

      when "elem_mul"
        ElemMulExec.new(id)

      when "add"
        AddExec.new(id)

      when "mean_pool"
        MeanPoolExec.new(id)

      when "broadcast"
        BroadcastExec.new(id)

      # Compound nodes (use Crystal classes for performance)
      when "attention_layer"
        d_model = node.param_i("d_model", 64)
        n_heads = node.param_i("n_heads", 4)
        AttentionCompoundExec.new(id, d_model, n_heads, @seq_len)

      when "layer_norm"
        # Use fused backend (compound) — the math decomposition children
        # (reduce_mean, subtract, etc.) aren't implemented yet.
        dim = node.param_i("dim", 64)
        LayerNormFusedExec.new(id, dim)

      when "loss"
        MathLossExec.new(id)

      # Router (compound for now)
      when "global_router"
        epsilon = node.param_f("epsilon", 0.2)
        stream_dim = node.param_i("stream_dim", 64)
        GlobalRouterExec.new(id, 2, stream_dim, epsilon)

      else
        nil  # Unknown or skip type
      end
    end

    # ── Boundary Port Resolution ──────────────────────────────────────────────

    private def resolve_from(
      port : PortRef, current_graph : GraphData,
      parent_node : Node?, parent_graph : GraphData, prefix : String
    ) : {String, String}?
      if port.nodeId == -1
        return {"__boundary__", port.portId} unless parent_node
        parent_graph.edges.each do |pe|
          if pe.to.nodeId == parent_node.id && pe.to.portId == port.portId
            if pe.from.nodeId == -1
              return {"__boundary__", pe.from.portId}
            else
              parent_prefix = strip_suffix(prefix, parent_node.id)
              from_node = parent_graph.find_node(pe.from.nodeId)
              if from_node && from_node.children && has_compilable_children?(from_node)
                from_prefix = make_id(parent_prefix, pe.from.nodeId)
                output_node = find_boundary_output_node(from_node.children.not_nil!, pe.from.portId, from_prefix)
                return output_node if output_node
              end
              return {make_id(parent_prefix, pe.from.nodeId), pe.from.portId}
            end
          end
        end
        nil
      else
        node = current_graph.find_node(port.nodeId)
        if node && node.children && has_compilable_children?(node)
          child_prefix = make_id(prefix, port.nodeId)
          output_node = find_boundary_output_node(node.children.not_nil!, port.portId, child_prefix)
          return output_node if output_node
        end
        {make_id(prefix, port.nodeId), port.portId}
      end
    end

    private def resolve_to(
      port : PortRef, current_graph : GraphData,
      parent_node : Node?, parent_graph : GraphData, prefix : String
    ) : {String, String}?
      if port.nodeId == -1
        return {"__boundary__", port.portId} unless parent_node
        parent_graph.edges.each do |pe|
          if pe.from.nodeId == parent_node.id && pe.from.portId == port.portId
            if pe.to.nodeId == -1
              return {"__boundary__", pe.to.portId}
            else
              parent_prefix = strip_suffix(prefix, parent_node.id)
              to_node = parent_graph.find_node(pe.to.nodeId)
              if to_node && to_node.children && has_compilable_children?(to_node)
                child_prefix = make_id(parent_prefix, pe.to.nodeId)
                input_node = find_boundary_input_node(to_node.children.not_nil!, pe.to.portId, child_prefix)
                return input_node if input_node
              end
              return {make_id(parent_prefix, pe.to.nodeId), pe.to.portId}
            end
          end
        end
        nil
      else
        node = current_graph.find_node(port.nodeId)
        if node && node.children && has_compilable_children?(node)
          child_prefix = make_id(prefix, port.nodeId)
          input_node = find_boundary_input_node(node.children.not_nil!, port.portId, child_prefix)
          return input_node if input_node
        end
        {make_id(prefix, port.nodeId), port.portId}
      end
    end

    private def strip_suffix(prefix : String, node_id : Int32) : String
      suffix = ".#{node_id}"
      prefix.ends_with?(suffix) ? prefix[0..-(suffix.size + 1)] : prefix
    end

    private def find_boundary_output_node(children : GraphData, port_id : String, prefix : String) : {String, String}?
      children.edges.each do |e|
        if e.to.nodeId == -1 && e.to.portId == port_id
          from_node = children.find_node(e.from.nodeId)
          if from_node && from_node.children && has_compilable_children?(from_node)
            child_prefix = make_id(prefix, e.from.nodeId)
            return find_boundary_output_node(from_node.children.not_nil!, e.from.portId, child_prefix)
          end
          return {make_id(prefix, e.from.nodeId), e.from.portId}
        end
      end
      nil
    end

    private def find_boundary_input_node(children : GraphData, port_id : String, prefix : String) : {String, String}?
      children.edges.each do |e|
        if e.from.nodeId == -1 && e.from.portId == port_id
          to_node = children.find_node(e.to.nodeId)
          if to_node && to_node.children && has_compilable_children?(to_node)
            child_prefix = make_id(prefix, e.to.nodeId)
            return find_boundary_input_node(to_node.children.not_nil!, e.to.portId, child_prefix)
          end
          return {make_id(prefix, e.to.nodeId), e.to.portId}
        end
      end
      nil
    end

    private def trace_through_skipped(port : PortRef, graph : GraphData, prefix : String) : {String, String}?
      node = graph.find_node(port.nodeId)
      return nil unless node
      graph.edges.each do |e|
        next unless e.to.nodeId == node.id
        if e.from.nodeId == -1
          return {"__boundary__", port.portId}
        else
          result = trace_through_skipped(e.from, graph, prefix)
          return {"__boundary__", port.portId} if result
        end
      end
      nil
    end
  end
end
