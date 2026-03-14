require "http/server"
require "json"
require "file_utils"
require "./graph"
require "./builder"

# HTTP server for the microGPT Construction Kit.
# Serves the static frontend and provides REST + WebSocket APIs
# for model construction, validation, and training.

module ConstructionKit

  class Server
    STATIC_DIR = File.join(__DIR__, "public")

    getter builder : Builder? = nil
    getter training : Bool = false
    getter train_step : Int32 = 0
    getter train_steps : Int32 = 0
    getter avg_loss : Float64 = 0.0
    getter train_channel : Channel(TrainEvent) = Channel(TrainEvent).new(256)
    getter train_started_at : Time = Time.utc
    getter train_log : IO? = nil
    getter train_run_name : String = ""
    getter train_metrics : Array(MetricPoint) = [] of MetricPoint

    DEFAULT_DATA_DIR = File.join(
      ENV.fetch("XDG_DATA_HOME", File.join(Path.home.to_s, ".local", "share")),
      "microgpt"
    )

    def initialize(@host : String = "127.0.0.1", @port : Int32 = 8080, @data_dir : String = DEFAULT_DATA_DIR)
      @saves_dir = File.join(@data_dir, "saves")
      @logs_dir = File.join(@data_dir, "logs")
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
          doc = GraphDocument.from_json(body)
          errors = ConstructionKit.validate(doc.graph)
          unless errors.empty?
            ctx.response.status = HTTP::Status.new(400)
            ctx.response.print({error: "Validation failed", details: errors.map(&.to_s)}.to_json)
            return
          end

          config = ConstructionKit.extract_config(doc.graph)
          @builder = Builder.new(config)
          summary = @builder.not_nil!.summary
          ctx.response.print({built: true, summary: summary}.to_json)
        rescue ex
          ctx.response.status = HTTP::Status.new(400)
          ctx.response.print({error: ex.message}.to_json)
        end

      when {"GET", "/api/summary"}
        if b = @builder
          ctx.response.print(b.summary.to_json)
        else
          ctx.response.status = HTTP::Status.new(404)
          ctx.response.print({error: "No model built yet"}.to_json)
        end

      when {"POST", "/api/train/start"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        params = JSON.parse(body)
        steps = params["steps"]?.try(&.as_i?) || 1000
        run_name = params["name"]?.try(&.as_s?) || ""

        if @builder.nil?
          ctx.response.status = HTTP::Status.new(400)
          ctx.response.print({error: "No model built — call /api/build first"}.to_json)
          return
        end

        if @training
          ctx.response.status = HTTP::Status.new(409)
          ctx.response.print({error: "Training already in progress"}.to_json)
          return
        end

        start_training(steps, run_name)
        ctx.response.print({started: true, steps: steps, run: @train_run_name}.to_json)

      when {"POST", "/api/train/stop"}
        @training = false
        ctx.response.print({stopped: true, step: @train_step}.to_json)

      when {"GET", "/api/train/status"}
        elapsed = (Time.utc - @train_started_at).total_seconds
        ctx.response.print({
          training:    @training,
          step:        @train_step,
          steps:       @train_steps,
          avg_loss:    @avg_loss,
          elapsed_sec: elapsed.round(1),
        }.to_json)

      when {"POST", "/api/generate"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        params = JSON.parse(body)
        seed = params["seed"]?.try(&.as_s?) || "First Citizen:\n"
        max_tokens = params["max_tokens"]?.try(&.as_i?) || 200
        temperature = params["temperature"]?.try(&.as_f?) || 0.8

        if b = @builder
          text = b.generate(seed, max_tokens, temperature)
          ctx.response.print({text: text}.to_json)
        else
          ctx.response.status = HTTP::Status.new(400)
          ctx.response.print({error: "No model built"}.to_json)
        end

      when {"POST", "/api/save"}
        body = ctx.request.body.try(&.gets_to_end) || "{}"
        params = JSON.parse(body)
        hash = params["hash"]?.try(&.as_s?) || "unknown"
        name = params["name"]?.try(&.as_s?) || "default"
        save_dir = File.join(@saves_dir, hash, name)

        if b = @builder
          b.save_weights(save_dir)
          # Save metrics alongside weights
          metrics_data = {
            metrics:    @train_metrics,
            run_name:   @train_run_name,
            final_loss: @train_metrics.last?.try(&.avg_loss) || @avg_loss,
            total_steps: @train_metrics.last?.try(&.step) || @train_step,
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
        hash = params["hash"]?.try(&.as_s?) || "unknown"
        name = params["name"]?.try(&.as_s?) || "default"
        save_dir = File.join(@saves_dir, hash, name)

        if b = @builder
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

    private def start_training(steps : Int32, run_name : String = "")
      @training = true
      @train_step = 0
      @train_steps = steps
      @avg_loss = 0.0
      @train_started_at = Time.utc
      @train_metrics = [] of MetricPoint

      # Open log file
      Dir.mkdir_p(@logs_dir)
      timestamp = Time.utc.to_s("%Y%m%d_%H%M%S")
      prefix = run_name.empty? ? "run" : run_name.gsub(/[^a-zA-Z0-9_-]/, "_")
      @train_run_name = "#{prefix}_#{timestamp}"
      log_path = File.join(@logs_dir, "#{@train_run_name}.jsonl")
      log_file = File.open(log_path, "w")
      @train_log = log_file

      # Write run header
      header = {
        type:     "run_start",
        name:     @train_run_name,
        steps:    steps,
        time:     Time.utc.to_rfc3339,
        summary:  @builder.try(&.summary),
      }
      log_file.puts(header.to_json)
      log_file.flush

      spawn do
        b = @builder.not_nil!
        steps.times do |step|
          break unless @training

          result = b.train_step
          @train_step = step + 1
          loss = result.loss
          @avg_loss = step == 0 ? loss : 0.99 * @avg_loss + 0.01 * loss
          elapsed = (Time.utc - @train_started_at).total_seconds

          # Sample metrics for persistence (every 10 steps, plus first and last)
          metric_interval = Math.max(1, steps // 500)  # ~500 points max
          if step == 0 || (step + 1) % metric_interval == 0 || step + 1 == steps
            @train_metrics << MetricPoint.new(
              step: step + 1,
              loss: loss,
              avg_loss: @avg_loss,
              grad_norm: result.grad_norm,
              elapsed: elapsed.round(2),
            )
          end

          # Emit event for WebSocket clients
          event = TrainEvent.new(
            step: step + 1,
            loss: loss,
            avg_loss: @avg_loss,
            grad_norm: result.grad_norm,
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
            avg_loss:       @avg_loss,
            grad_norm:      result.grad_norm,
            router_weights: result.router_weights,
            elapsed_sec:    elapsed.round(2),
            sample:         event.sample,
          }
          log_file.puts(log_entry.to_json)
          log_file.flush if step % 10 == 0

          begin
            train_channel.send(event)
          rescue Channel::ClosedError
          end

          Fiber.yield  # let HTTP server process requests
          GC.collect if step % 10 == 0
        end

        # Write run footer
        elapsed = (Time.utc - @train_started_at).total_seconds
        footer = {
          type:        "run_end",
          final_step:  @train_step,
          final_loss:  @avg_loss,
          elapsed_sec: elapsed.round(2),
          completed:   @train_step >= steps,
          time:        Time.utc.to_rfc3339,
        }
        log_file.puts(footer.to_json)
        log_file.close
        @train_log = nil

        @training = false
      end
    end

    private def handle_websocket(ctx)
      ws = HTTP::WebSocketHandler.new do |socket, _ctx|
        # Forward training events to this WebSocket
        spawn do
          loop do
            begin
              event = train_channel.receive
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
              start_training(steps) unless @training
            when "stop"
              @training = false
            when "generate"
              if b = @builder
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
    property step : Int32
    property loss : Float64
    property avg_loss : Float64
    property grad_norm : Float64
    property router_weights : String
    property sample : String? = nil

    def initialize(@step, @loss, @avg_loss, @grad_norm, @router_weights)
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
