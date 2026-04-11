module MicroGPT
  module AGPT
    # Per-block intermediates saved during incremental forward for backward use.
    class BlockStepState
      property x_input : Mat           # [1, d_model] — block input (for residual backward)
      property ln1_out : Mat           # [1, d_model] — LN1 output WITH gamma/beta = input to WQ/WK/WV
      property ln1_normed : Mat        # [1, d_model] — LN1 normalized WITHOUT gamma/beta (for LN backward)
      property ln1_std_inv : Float64   # scalar for single-row LN
      property q_parts : Array(Mat)    # per head [1, head_dim], post-RoPE
      property attn_weights : Array(Mat) # per head [1, prefix_len]
      property wo_input : Mat          # [1, d_model] — concat of head outputs, input to WO
      property x_after_attn : Mat      # [1, d_model] — after first residual
      property ln2_out : Mat           # [1, d_model] — LN2 output WITH gamma/beta = input to FFN L1
      property ln2_normed : Mat        # [1, d_model] — LN2 normalized WITHOUT gamma/beta (for LN backward)
      property ln2_std_inv : Float64
      property ff_relu_out : Mat       # [1, d_ff] — L1 output after ReLU = input to L2
      property ff_relu_mask : Mat      # [1, d_ff] — 1.0 where positive, 0.0 where negative

      def initialize(
        @x_input, @ln1_out, @ln1_normed, @ln1_std_inv,
        @q_parts, @attn_weights, @wo_input,
        @x_after_attn, @ln2_out, @ln2_normed, @ln2_std_inv,
        @ff_relu_out, @ff_relu_mask
      )
      end
    end

    # All intermediates for one token step through the full model.
    class NodeForwardState
      property block_states : Array(BlockStepState)
      property final_x : Mat           # [1, d_model] — last block output = final norm input
      property final_normed : Mat      # [1, d_model] — after normalization (before gamma/beta)
      property final_std_inv : Float64
      property final_norm_out : Mat    # [1, d_model] — final norm output = output proj input
      property token_id : Int32
      property position : Int32
      property ancestor_ids : Array(Int32) # trie node IDs for positions 0..current

      def initialize(
        @block_states, @final_x, @final_normed, @final_std_inv,
        @final_norm_out, @token_id, @position, @ancestor_ids
      )
      end
    end

    # Per-node gradient accumulator for dK/dV contributions from descendants.
    # Indexed by [layer][head] → [1, head_dim].
    class NodeGradAccum
      getter dk : Array(Array(Mat))  # [n_layers][n_heads] → [1, head_dim]
      getter dv : Array(Array(Mat))  # same

      def initialize(n_layers : Int32, head_dims : Array(Int32))
        @dk = Array.new(n_layers) { head_dims.map { |hd| Mat.new(1, hd) } }
        @dv = Array.new(n_layers) { head_dims.map { |hd| Mat.new(1, hd) } }
      end

      def add_dk(layer : Int32, head : Int32, row : Mat)
        @dk[layer][head].add!(row)
      end

      def add_dv(layer : Int32, head : Int32, row : Mat)
        @dv[layer][head].add!(row)
      end
    end
  end
end
