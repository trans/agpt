require "http/server"
require "json"
require "file_utils"
require "./graph"
require "./builder"

# HTTP server for the microGPT Construction Kit.
# Serves the static frontend and provides REST + WebSocket APIs
# for model construction, validation, and training.

module ConstructionKit

  class ModelSlot
    property builder : Builder? = nil
    property training : Bool = false
    property train_step : Int32 = 0
    property train_steps : Int32 = 0
    property avg_loss : Float64 = 0.0
    property train_channel : Channel(TrainEvent) = Channel(TrainEvent).new(256)
    property train_started_at : Time = Time.utc
    property train_log : IO? = nil
    property train_run_name : String = ""
    property train_metrics : Array(MetricPoint) = [] of MetricPoint
  end

  class Server
    STATIC_DIR = File.join(__DIR__, "public")

    DEFAULT_DATA_DIR = File.join(
      ENV.fetch("XDG_DATA_HOME", File.join(Path.home.to_s, ".local", "share")),
      "microgpt"
    )

    def initialize(@host : String = "127.0.0.1", @port : Int32 = 8080, @data_dir : String = DEFAULT_DATA_DIR)
      @saves_dir = File.join(@data_dir, "saves")
      @logs_dir = File.join(@data_dir, "logs")
      @slots = {} of String => ModelSlot
    end

    private def slot_for(card_id : String) : ModelSlot
      @slots[card_id] ||= ModelSlot.new
    end

    def start
      server = HTTP::Server.new do |ctx|
        handle_request(ctx)
      end

      puts "microGPT Construction Kit"
      puts "  http://#{@host}:#{@port}"
      puts "  Static: #{STATIC_DIR}"
      puts "  Data:   #{@data_dir}"
      puts

      server.bind_tcp(@host, @port)
      server.listen
    end

    private def handle_request(ctx : HTTP::Server::Context)
      path = ctx.request.path
      method = ctx.request.method

      # API routes
      if path.starts_with?("/api/")
        ctx.response.content_type = "application/json"
        handle_api(ctx, path, method)
        return
      end

      # WebSocket upgrade for training
      if path == "/ws/train"
        handle_websocket(ctx)
        return
      end

      # Static file serving
      serve_static(ctx, path)
    end

    # Extract card_id from JSON body (for POST/DELETE) or query param (for GET)
    private def extract_card_id(ctx, params : JSON::Any? = nil) : String?
      if ctx.request.method == "GET" || ctx.request.method == "DELETE"
        ctx.request.query_params["card_id"]?
      else
        params.try(&.["card_id"]?.try(&.as_s?))
      end
    end

    private def require_card_id(ctx, params : JSON::Any? = nil) : String?
      card_id = extract_card_id(ctx, params)
      unless card_id
        ctx.response.status = HTTP::Status.new(400)
        ctx.response.print({error: "card_id required"}.to_json)
      end
      card_id
    end

    private def handle_api(ctx, path, method)
      case {method, path}
      when {"POST", "/api/validate"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        begin
          doc = GraphDocument.from_json(body)
          errors = ConstructionKit.validate(doc.graph)
          if errors.empty?
            ctx.response.print({valid: true, errors: [] of String}.to_json)
          else
            ctx.response.print({valid: false, errors: errors.map(&.to_s)}.to_json)
          end
        rescue ex
          ctx.response.status = HTTP::Status.new(400)
          ctx.response.print({valid: false, errors: ["Invalid JSON: #{ex.message}"]}.to_json)
        end

      when {"POST", "/api/build"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        begin
          params = JSON.parse(body)
          card_id = require_card_id(ctx, params)
          return unless card_id

          doc = GraphDocument.from_json(body)
          errors = ConstructionKit.validate(doc.graph)
          unless errors.empty?
            ctx.response.status = HTTP::Status.new(400)
            ctx.response.print({error: "Validation failed", details: errors.map(&.to_s)}.to_json)
            return
          end

          data_file = params["data_file"]?.try(&.as_s?) || "data/input.txt"
          config = ConstructionKit.extract_config(doc.graph, data_file)
          client_hash = params["hash"]?.try(&.as_s?)
          server_hash = compute_graph_hash(doc.graph)
          # Use server hash when available, fall back to client hash
          effective_hash = server_hash || client_hash
          slot = slot_for(card_id)
          slot.builder = Builder.new(config)
          summary = slot.builder.not_nil!.summary

          # Save graph definition under its hash
          if effective_hash
            hash_dir = File.join(@saves_dir, effective_hash)
            Dir.mkdir_p(hash_dir)
            File.write(File.join(hash_dir, "graph.json"), doc.graph.to_json)
          end

          # TODO: Once compute_graph_hash is implemented, compare server_hash
          # against client_hash and reject on mismatch to catch canonicalization
          # divergence early.
          hash_match = server_hash.nil? || client_hash.nil? || server_hash == client_hash
          ctx.response.print({
            built:       true,
            summary:     summary,
            model_hash:  effective_hash,
            hash_match:  hash_match,
          }.to_json)
        rescue ex
          ctx.response.status = HTTP::Status.new(400)
          ctx.response.print({error: ex.message}.to_json)
        end

      when {"GET", "/api/summary"}
        card_id = require_card_id(ctx)
        return unless card_id
        slot = slot_for(card_id)
        if b = slot.builder
          ctx.response.print(b.summary.to_json)
        else
          ctx.response.status = HTTP::Status.new(404)
          ctx.response.print({error: "No model built yet"}.to_json)
        end

      when {"POST", "/api/train/start"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        params = JSON.parse(body)
        card_id = require_card_id(ctx, params)
        return unless card_id
        slot = slot_for(card_id)

        steps = params["steps"]?.try(&.as_i?) || 1000
        run_name = params["name"]?.try(&.as_s?) || ""

        if slot.builder.nil?
          ctx.response.status = HTTP::Status.new(400)
          ctx.response.print({error: "No model built — call /api/build first"}.to_json)
          return
        end

        if slot.training
          ctx.response.status = HTTP::Status.new(409)
          ctx.response.print({error: "Training already in progress"}.to_json)
          return
        end

        start_training(slot, steps, run_name)
        ctx.response.print({started: true, steps: steps, run: slot.train_run_name}.to_json)

      when {"POST", "/api/train/stop"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        params = JSON.parse(body)
        card_id = require_card_id(ctx, params)
        return unless card_id
        slot = slot_for(card_id)
        slot.training = false
        ctx.response.print({stopped: true, step: slot.train_step}.to_json)

      when {"GET", "/api/train/status"}
        card_id = require_card_id(ctx)
        return unless card_id
        slot = slot_for(card_id)
        elapsed = (Time.utc - slot.train_started_at).total_seconds
        ctx.response.print({
          training:    slot.training,
          step:        slot.train_step,
          steps:       slot.train_steps,
          avg_loss:    slot.avg_loss,
          elapsed_sec: elapsed.round(1),
        }.to_json)

      when {"POST", "/api/generate"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        params = JSON.parse(body)
        card_id = require_card_id(ctx, params)
        return unless card_id
        slot = slot_for(card_id)

        seed = params["seed"]?.try(&.as_s?) || "First Citizen:\n"
        max_tokens = params["max_tokens"]?.try(&.as_i?) || 200
        temperature = params["temperature"]?.try(&.as_f?) || 0.8

        if b = slot.builder
          text = b.generate(seed, max_tokens, temperature)
          ctx.response.print({text: text}.to_json)
        else
          ctx.response.status = HTTP::Status.new(400)
          ctx.response.print({error: "No model built"}.to_json)
        end

      when {"POST", "/api/save"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        params = JSON.parse(body)
        card_id = require_card_id(ctx, params)
        return unless card_id
        slot = slot_for(card_id)

        hash = params["hash"]?.try(&.as_s?) || "unknown"
        name = params["name"]?.try(&.as_s?) || "default"
        save_dir = File.join(@saves_dir, hash, name)

        if b = slot.builder
          b.save_weights(save_dir)
          # Save metrics alongside weights
          metrics_data = {
            metrics:    slot.train_metrics,
            run_name:   slot.train_run_name,
            final_loss: slot.train_metrics.last?.try(&.avg_loss) || slot.avg_loss,
            total_steps: slot.train_metrics.last?.try(&.step) || slot.train_step,
            saved_at:   Time.utc.to_rfc3339,
          }
          File.write(File.join(save_dir, "metrics.json"), metrics_data.to_json)
          ctx.response.print({saved: true, path: save_dir}.to_json)
        else
          ctx.response.status = HTTP::Status.new(400)
          ctx.response.print({error: "No model built"}.to_json)
        end

      when {"POST", "/api/load"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        params = JSON.parse(body)
        card_id = require_card_id(ctx, params)
        return unless card_id
        slot = slot_for(card_id)

        hash = params["hash"]?.try(&.as_s?) || "unknown"
        name = params["name"]?.try(&.as_s?) || "default"
        save_dir = File.join(@saves_dir, hash, name)

        if b = slot.builder
          unless Dir.exists?(save_dir)
            ctx.response.status = HTTP::Status.new(404)
            ctx.response.print({error: "No save found: #{hash}/#{name}"}.to_json)
            return
          end
          b.load_weights(save_dir)
          ctx.response.print({loaded: true, path: save_dir}.to_json)
        else
          ctx.response.status = HTTP::Status.new(400)
          ctx.response.print({error: "No model built — build first, then load weights"}.to_json)
        end

      when {"DELETE", "/api/save"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        params = JSON.parse(body)
        hash = params["hash"]?.try(&.as_s?) || ""
        name = params["name"]?.try(&.as_s?) || ""
        if hash.empty? || name.empty?
          ctx.response.status = HTTP::Status.new(400)
          ctx.response.print({error: "hash and name required"}.to_json)
          return
        end
        save_dir = File.join(@saves_dir, hash, name)
        if Dir.exists?(save_dir)
          FileUtils.rm_rf(save_dir)
          # Clean up empty hash dir
          hash_dir = File.join(@saves_dir, hash)
          if Dir.exists?(hash_dir) && Dir.children(hash_dir).empty?
            FileUtils.rm_rf(hash_dir)
          end
          ctx.response.print({deleted: true, path: save_dir}.to_json)
        else
          ctx.response.status = HTTP::Status.new(404)
          ctx.response.print({error: "No save found: #{hash}/#{name}"}.to_json)
        end

      when {"DELETE", "/api/slot"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        params = JSON.parse(body)
        card_id = require_card_id(ctx, params)
        return unless card_id
        if slot = @slots.delete(card_id)
          slot.training = false
          ctx.response.print({deleted: true, card_id: card_id}.to_json)
        else
          ctx.response.print({deleted: false, card_id: card_id}.to_json)
        end

      when {"GET", "/api/runs"}
        runs = [] of NamedTuple(name: String, size: Int64, modified: String)
        if Dir.exists?(@logs_dir)
          Dir.each_child(@logs_dir) do |name|
            next unless name.ends_with?(".jsonl")
            path = File.join(@logs_dir, name)
            info = File.info(path)
            runs << {
              name:     name.rchop(".jsonl"),
              size:     info.size,
              modified: info.modification_time.to_rfc3339,
            }
          end
        end
        runs.sort_by! { |r| r[:modified] }.reverse!
        ctx.response.print({runs: runs}.to_json)

      else
        # Dynamic routes
        if path.starts_with?("/api/saves/") && method == "GET"
          parts = path.sub("/api/saves/", "").split('/')
          if parts.size == 2
            # GET /api/saves/<hash>/<name> — fetch metrics for a specific save
            hash, name = parts
            metrics_path = File.join(@saves_dir, hash, name, "metrics.json")
            if File.exists?(metrics_path)
              ctx.response.print File.read(metrics_path)
            else
              ctx.response.status = HTTP::Status.new(404)
              ctx.response.print({error: "No metrics for #{hash}/#{name}"}.to_json)
            end
          elsif parts.size == 1 && !parts[0].empty?
            # GET /api/saves/<hash> — list saves for a model hash
            hash = parts[0]
            hash_dir = File.join(@saves_dir, hash)
            saves = [] of NamedTuple(name: String, modified: String, final_loss: Float64?)
            if Dir.exists?(hash_dir)
              Dir.each_child(hash_dir) do |name|
                sub = File.join(hash_dir, name)
                if Dir.exists?(sub) && File.exists?(File.join(sub, "meta.json"))
                  info = File.info(File.join(sub, "meta.json"))
                  # Try to read final_loss from metrics
                  final_loss : Float64? = nil
                  m_path = File.join(sub, "metrics.json")
                  if File.exists?(m_path)
                    begin
                      m = JSON.parse(File.read(m_path))
                      final_loss = m["final_loss"]?.try(&.as_f?)
                    rescue
                    end
                  end
                  saves << {
                    name:       name,
                    modified:   info.modification_time.to_rfc3339,
                    final_loss: final_loss,
                  }
                end
              end
            end
            saves.sort_by! { |s| s[:modified] }.reverse!
            ctx.response.print({saves: saves}.to_json)
          else
            ctx.response.status = HTTP::Status.new(400)
            ctx.response.print({error: "Invalid saves path"}.to_json)
          end

        elsif path.starts_with?("/api/runs/") && method == "GET"
          run_name = path.sub("/api/runs/", "")
          log_path = File.join(@logs_dir, "#{run_name}.jsonl")
          if File.exists?(log_path)
            ctx.response.content_type = "application/x-ndjson"
            ctx.response.print File.read(log_path)
          else
            ctx.response.status = HTTP::Status.new(404)
            ctx.response.print({error: "Run not found: #{run_name}"}.to_json)
          end
        else
          ctx.response.status = HTTP::Status.new(404)
          ctx.response.print({error: "Unknown API endpoint"}.to_json)
        end
      end
    end

    private def start_training(slot : ModelSlot, steps : Int32, run_name : String = "")
      slot.training = true
      slot.train_step = 0
      slot.train_steps = steps
      slot.avg_loss = 0.0
      slot.train_started_at = Time.utc
      slot.train_metrics = [] of MetricPoint
      # Fresh channel for this training run (old WS readers will get ClosedError)
      slot.train_channel.close rescue nil
      slot.train_channel = Channel(TrainEvent).new(256)

      # Open log file
      Dir.mkdir_p(@logs_dir)
      timestamp = Time.utc.to_s("%Y%m%d_%H%M%S")
      prefix = run_name.empty? ? "run" : run_name.gsub(/[^a-zA-Z0-9_-]/, "_")
      slot.train_run_name = "#{prefix}_#{timestamp}"
      log_path = File.join(@logs_dir, "#{slot.train_run_name}.jsonl")
      log_file = File.open(log_path, "w")
      slot.train_log = log_file

      # Write run header
      header = {
        type:     "run_start",
        name:     slot.train_run_name,
        steps:    steps,
        time:     Time.utc.to_rfc3339,
        summary:  slot.builder.try(&.summary),
      }
      log_file.puts(header.to_json)
      log_file.flush

      spawn do
        b = slot.builder.not_nil!
        steps.times do |step|
          break unless slot.training

          result = b.train_step
          slot.train_step = step + 1
          loss = result.loss
          slot.avg_loss = step == 0 ? loss : 0.99 * slot.avg_loss + 0.01 * loss
          elapsed = (Time.utc - slot.train_started_at).total_seconds

          # Sample metrics for persistence (every 10 steps, plus first and last)
          metric_interval = Math.max(1, steps // 500)  # ~500 points max
          if step == 0 || (step + 1) % metric_interval == 0 || step + 1 == steps
            slot.train_metrics << MetricPoint.new(
              step: step + 1,
              loss: loss,
              avg_loss: slot.avg_loss,
              grad_norm: result.grad_norm,
              elapsed: elapsed.round(2),
            )
          end

          # Emit event for WebSocket clients
          event = TrainEvent.new(
            step: step + 1,
            steps: steps,
            loss: loss,
            avg_loss: slot.avg_loss,
            grad_norm: result.grad_norm,
            elapsed_sec: elapsed.round(2),
            router_weights: result.router_weights,
          )

          # Generate sample every 50 steps
          if step % 50 == 0
            sample = b.generate("First Citizen:\n", 100)
            event.sample = sample
          end

          # Write to log
          log_entry = {
            type:           "step",
            step:           step + 1,
            loss:           loss,
            avg_loss:       slot.avg_loss,
            grad_norm:      result.grad_norm,
            router_weights: result.router_weights,
            elapsed_sec:    elapsed.round(2),
            sample:         event.sample,
          }
          log_file.puts(log_entry.to_json)
          log_file.flush if step % 10 == 0

          begin
            slot.train_channel.send(event)
          rescue Channel::ClosedError
          end

          Fiber.yield  # let HTTP server process requests
          GC.collect if step % 10 == 0
        end

        # Write run footer
        elapsed = (Time.utc - slot.train_started_at).total_seconds
        footer = {
          type:        "run_end",
          final_step:  slot.train_step,
          final_loss:  slot.avg_loss,
          elapsed_sec: elapsed.round(2),
          completed:   slot.train_step >= steps,
          time:        Time.utc.to_rfc3339,
        }
        log_file.puts(footer.to_json)
        log_file.close
        slot.train_log = nil

        slot.training = false

        # Send done event to WebSocket clients
        done_event = TrainEvent.new(
          step: slot.train_step,
          steps: steps,
          loss: slot.avg_loss,
          avg_loss: slot.avg_loss,
          grad_norm: 0.0,
          elapsed_sec: elapsed.round(2),
          router_weights: "",
        )
        done_event.type = "done"
        begin
          slot.train_channel.send(done_event)
        rescue Channel::ClosedError
        end
      end
    end

    private def handle_websocket(ctx)
      # Extract card_id from query params
      card_id = ctx.request.query_params["card_id"]?
      unless card_id
        ctx.response.status = HTTP::Status.new(400)
        ctx.response.print "card_id required"
        return
      end
      slot = slot_for(card_id)

      ws = HTTP::WebSocketHandler.new do |socket, _ctx|
        # Forward training events to this WebSocket
        spawn do
          loop do
            begin
              event = slot.train_channel.receive
              socket.send(event.to_json)
            rescue Channel::ClosedError
              break
            rescue IO::Error
              break
            end
          end
        end

        socket.on_message do |msg|
          # Client can send commands: {"action": "start", "steps": 5000}
          begin
            cmd = JSON.parse(msg)
            case cmd["action"]?.try(&.as_s?)
            when "start"
              steps = cmd["steps"]?.try(&.as_i?) || 1000
              start_training(slot, steps) unless slot.training
            when "stop"
              slot.training = false
            when "generate"
              if b = slot.builder
                seed = cmd["seed"]?.try(&.as_s?) || "First Citizen:\n"
                text = b.generate(seed, 100)
                socket.send({type: "generation", text: text}.to_json)
              end
            end
          rescue ex
            socket.send({type: "error", message: ex.message}.to_json)
          end
        end

        socket.on_close do
          # Client disconnected
        end
      end

      ws.call(ctx)
    end

    # TODO: Implement proper canonical graph hashing in Crystal to match the
    # client-side SHA-256 (canonicalGraphObj → JSON.stringify → SHA-256 → first 8 hex chars).
    # For now this is a stub that uses Crystal's built-in hash. Once implemented,
    # compare against the client hash and reject on mismatch to catch serialization
    # divergence early. The client hash is passed as `hash` in the build request and
    # the server returns `hash_match: bool` so the frontend can warn if they differ.
    private def compute_graph_hash(graph : GraphData) : String?
      # Stub: return nil until proper canonicalization is implemented
      nil
    end

    private def serve_static(ctx, path)
      # Default to index.html
      path = "/index.html" if path == "/"

      file_path = File.join(STATIC_DIR, path.lstrip('/'))

      unless File.exists?(file_path) && !File.directory?(file_path)
        ctx.response.status = HTTP::Status.new(404)
        ctx.response.content_type = "text/plain"
        ctx.response.print "Not found: #{path}"
        return
      end

      ctx.response.content_type = mime_type(file_path)
      ctx.response.print File.read(file_path)
    end

    private def mime_type(path : String) : String
      case File.extname(path)
      when ".html" then "text/html"
      when ".css"  then "text/css"
      when ".js"   then "application/javascript"
      when ".json" then "application/json"
      when ".svg"  then "image/svg+xml"
      when ".png"  then "image/png"
      when ".ico"  then "image/x-icon"
      else              "application/octet-stream"
      end
    end
  end

  struct TrainEvent
    include JSON::Serializable
    property type : String = "step"
    property step : Int32
    property steps : Int32 = 0
    property loss : Float64
    property avg_loss : Float64
    property grad_norm : Float64
    property elapsed_sec : Float64 = 0.0
    property router_weights : String
    property sample : String? = nil

    def initialize(@step, @steps, @loss, @avg_loss, @grad_norm, @elapsed_sec, @router_weights)
    end
  end

  struct MetricPoint
    include JSON::Serializable
    property step : Int32
    property loss : Float64
    property avg_loss : Float64
    property grad_norm : Float64
    property elapsed : Float64

    def initialize(@step, @loss, @avg_loss, @grad_norm, @elapsed)
    end
  end
end

# CLI entry point
host = "127.0.0.1"
port = 8080
backend = "crystal"
data_dir = ConstructionKit::Server::DEFAULT_DATA_DIR

ARGV.each_with_index do |arg, i|
  case arg
  when "--port"     then port = ARGV[i + 1]?.try(&.to_i) || port
  when "--host"     then host = ARGV[i + 1]? || host
  when "--backend"  then backend = ARGV[i + 1]? || backend
  when "--data-dir" then data_dir = ARGV[i + 1]? || data_dir
  when "--cublas"   then backend = "cublas"
  when "--help", "-h"
    puts "Usage: construction-kit [options]"
    puts "  --port PORT      Server port (default: 8080)"
    puts "  --host HOST      Server host (default: 127.0.0.1)"
    puts "  --backend NAME   Compute backend: crystal, openblas, cublas"
    puts "  --cublas          Shorthand for --backend cublas"
    puts "  --data-dir DIR   Data directory for saves and logs"
    puts "                   (default: ~/.local/share/microgpt)"
    exit 0
  end
end

case backend
when "openblas" then MicroGPT.use_openblas!
when "cublas"   then MicroGPT.use_cublas!
else                 MicroGPT.use_crystal!
end

server = ConstructionKit::Server.new(host, port, data_dir)
server.start
