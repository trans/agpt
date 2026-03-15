require "json"

# Graph model for the visual construction kit.
# Validates connectivity, propagates dimension constraints,
# and provides the data structure that the builder translates
# into Crystal model objects.

module ConstructionKit

  # A port on a graph node (input or output)
  struct PortRef
    include JSON::Serializable
    property nodeId : Int32
    property portId : String

    def initialize(@nodeId = 0, @portId = "")
    end
  end

  # An edge connecting two ports
  class Edge
    include JSON::Serializable
    property from : PortRef
    property to : PortRef

    def initialize(@from = PortRef.new, @to = PortRef.new)
    end
  end

  # A node in the graph with its parameters
  class Node
    include JSON::Serializable
    property id : Int32 = 0
    property type : String = ""
    property x : Float64 = 0.0
    property y : Float64 = 0.0
    property params : Hash(String, JSON::Any) = {} of String => JSON::Any
    @[JSON::Field(emit_null: false)]
    property children : GraphData? = nil

    def param_i(key : String, default : Int32 = 0) : Int32
      if v = params[key]?
        v.as_i?.try(&.to_i32) || default
      else
        default
      end
    end

    def param_f(key : String, default : Float64 = 0.0) : Float64
      if v = params[key]?
        v.as_f? || default
      else
        default
      end
    end

    def param_s(key : String, default : String = "") : String
      if v = params[key]?
        v.as_s? || default
      else
        default
      end
    end
  end

  # The graph data structure (nodes + edges), used at every scope level
  class GraphData
    include JSON::Serializable
    property nodes : Array(Node)
    property edges : Array(Edge)

    def initialize(@nodes = [] of Node, @edges = [] of Edge)
    end

    def find_node(id : Int32) : Node?
      nodes.find { |n| n.id == id }
    end

    # Find all edges going into a specific node+port
    def edges_to(node_id : Int32, port_id : String) : Array(Edge)
      edges.select { |e| e.to.nodeId == node_id && e.to.portId == port_id }
    end

    # Find all edges coming from a specific node+port
    def edges_from(node_id : Int32, port_id : String) : Array(Edge)
      edges.select { |e| e.from.nodeId == node_id && e.from.portId == port_id }
    end

    # Find all source nodes connected to a given input port
    def sources_for(node_id : Int32, port_id : String) : Array(Node)
      edges_to(node_id, port_id).compact_map { |e| find_node(e.from.nodeId) }
    end
  end

  # Top-level graph document (versioned)
  class GraphDocument
    include JSON::Serializable
    property version : Int32 = 2
    property graph : GraphData

    def initialize(@version = 2, @graph = GraphData.new)
    end
  end

  # Validation error
  struct ValidationError
    getter message : String
    getter node_id : Int32?

    def initialize(@message, @node_id = nil)
    end

    def to_s : String
      if nid = node_id
        "Node ##{nid}: #{message}"
      else
        message
      end
    end
  end

  TOKENIZER_TYPES = ["char_tokenizer", "bpe_tokenizer"]
  WINDOWER_TYPES  = ["sequential_window", "sliding_window", "random_window", "dataset"]
  EXPERT_TYPES    = ["transformer", "counter", "bigram"]

  # Validate a graph for completeness and consistency
  def self.validate(graph : GraphData) : Array(ValidationError)
    errors = [] of ValidationError

    # Data source is now a workspace-level setting (Train panel), not a graph node.
    # Source nodes in the graph are ignored for validation.

    # Must have a windower (or legacy dataset) for seq_len
    windowers = graph.nodes.select { |n| WINDOWER_TYPES.includes?(n.type) }
    if windowers.empty?
      errors << ValidationError.new("Graph needs a windowing component (Sequential/Sliding/Random Window)")
    end

    # Must have a loss
    losses = graph.nodes.select { |n| n.type == "loss" }
    if losses.empty?
      errors << ValidationError.new("Graph needs a Loss component")
    end

    # Must have at least one cooperative ensemble
    coops = graph.nodes.select { |n| n.type == "cooperative" }
    if coops.empty?
      errors << ValidationError.new("Graph needs a Cooperative Ensemble")
    end

    # Check that loss has logits connected
    losses.each do |loss|
      if graph.edges_to(loss.id, "logits_in").empty?
        errors << ValidationError.new("Loss has no logits input connected", loss.id)
      end
      if graph.edges_to(loss.id, "targets").empty?
        errors << ValidationError.new("Loss has no targets input connected", loss.id)
      end
    end

    # Check cooperative ensembles have children (internal graph)
    coops.each do |coop|
      children = coop.children
      if children.nil? || children.nodes.empty?
        errors << ValidationError.new("Cooperative Ensemble has no internal components — double-click to add experts", coop.id)
        next
      end

      # Must have at least one expert inside
      experts = children.nodes.select { |n| EXPERT_TYPES.includes?(n.type) }
      if experts.empty?
        errors << ValidationError.new("Cooperative Ensemble needs at least one expert inside", coop.id)
      end

      # Must have exactly one router inside
      routers = children.nodes.select { |n| n.type.ends_with?("_router") }
      if routers.empty?
        errors << ValidationError.new("Cooperative Ensemble needs a router inside", coop.id)
      elsif routers.size > 1
        errors << ValidationError.new("Cooperative Ensemble should have exactly one router", coop.id)
      end

      # Check stream_dim consistency (experts inherit from ensemble)
      coop_stream_dim = coop.param_i("stream_dim", 64)
      stream_dims = experts.map { |e| e.param_i("stream_dim", coop_stream_dim) }.uniq
      if stream_dims.size > 1
        errors << ValidationError.new("Experts have mismatched stream_dim: #{stream_dims.join(", ")}", coop.id)
      end
    end

    errors
  end

  # Extract the model configuration from a validated pipeline graph.
  # data_file is passed separately (workspace-level setting, not in the graph).
  struct ModelConfig
    include JSON::Serializable
    property data_file : String
    property seq_len : Int32
    property stream_dim : Int32
    property expert_specs : Array(ExpertSpec)
    property router_type : String
    property router_epsilon : Float64
    property has_counter : Bool
    property learning_rate : Float64

    def initialize(
      @data_file, @seq_len, @stream_dim,
      @expert_specs, @router_type, @router_epsilon,
      @has_counter, @learning_rate
    )
    end
  end

  struct ExpertSpec
    include JSON::Serializable
    property type : String      # "transformer", "counter", "bigram"
    property d_model : Int32
    property n_layers : Int32
    property d_ff : Int32
    property stream_dim : Int32

    def initialize(@type, @d_model, @n_layers, @d_ff, @stream_dim)
    end

    # Format as the spec string used by main.cr: e.g. "d4x16n3"
    def to_spec_string : String
      n_heads = Math.max(1, d_model // 16)
      head_dim = d_model // n_heads
      "d#{n_heads}x#{head_dim}n#{n_layers}"
    end
  end

  # Extract a ModelConfig from the top-level pipeline graph
  def self.extract_config(graph : GraphData, data_file : String = "data/input.txt") : ModelConfig
    # Find windower (seq_len)
    windower = graph.nodes.find { |n| WINDOWER_TYPES.includes?(n.type) }
    coop = graph.nodes.find! { |n| n.type == "cooperative" }
    children = coop.children.not_nil!
    seq_len = windower.try(&.param_i("seq_len", 128)) || 128
    stream_dim = coop.param_i("stream_dim", 64)

    # Extract experts from children
    expert_nodes = children.nodes.select { |n| ["transformer", "counter", "bigram"].includes?(n.type) }
    expert_specs = expert_nodes.map do |e|
      ExpertSpec.new(
        type: e.type,
        d_model: e.param_i("d_model", 64),
        n_layers: e.param_i("n_layers", 3),
        d_ff: e.param_i("d_ff", e.param_i("d_model", 64) * 4),
        stream_dim: e.param_i("stream_dim", stream_dim),
      )
    end

    # Extract router from children
    router_node = children.nodes.find { |n| n.type.ends_with?("_router") }
    router_type = case router_node.try(&.type)
                  when "context_router" then "context"
                  when "gated_router"   then "gated"
                  else                       "global"
                  end
    router_epsilon = router_node.try(&.param_f("epsilon", 0.2)) || 0.2

    # Determine if there's a counter expert
    has_counter = expert_nodes.any? { |n| n.type == "counter" }

    # Find optimizer node (pipeline level only for now)
    # TODO: Optimizer inheritance — optimizer nodes can appear at any scope level
    # (pipeline, ensemble, transformer_internal, expert_internal). Deeper scopes
    # override shallower ones. Walk the graph tree top-down, resolve the effective
    # optimizer per expert/layer, and pass per-expert optimizer configs to the
    # builder so different experts can train with different learning rates, betas,
    # or even different algorithms (e.g. SGD for the counter, Adam for transformers).
    opt_node = graph.nodes.find { |n| n.type == "optimizer" }
    learning_rate = opt_node.try(&.param_f("learning_rate", 3e-4)) || 3e-4

    ModelConfig.new(
      data_file: data_file,
      seq_len: seq_len,
      stream_dim: stream_dim,
      expert_specs: expert_specs,
      router_type: router_type,
      router_epsilon: router_epsilon,
      has_counter: has_counter,
      learning_rate: learning_rate,
    )
  end
end
