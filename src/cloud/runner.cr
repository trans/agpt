require "./provider"
require "./vast_ai"

# Cloud experiment runner — deploys microgpt to a rented GPU and runs experiments.
#
# Usage:
#   cloud = Cloud::Runner.new(Cloud::VastAI.new)
#   cloud.run_experiments(
#     data: "data/addition.txt",
#     eval: "data/eval_add.txt",
#     config: "models.yml",
#     runs: ["add-5e-calc", "add-4e-baseline"],
#     gpu: "RTX 4090",
#     max_price: 1.0,
#   )

module Cloud

class Runner
  getter provider : Provider

  # Files/dirs to upload for a microgpt experiment
  DEPLOY_FILES = [
    "bin/microgpt",
    "models.yml",
  ]

  def initialize(@provider : Provider)
  end

  # List available GPU offers
  def search(gpu : String? = nil, max_price : Float64 = 2.0, limit : Int32 = 10)
    puts "Searching #{@provider.name} for GPUs..."
    offers = @provider.search(gpu: gpu, max_price: max_price, limit: limit)
    if offers.empty?
      puts "No offers found."
    else
      puts "Found #{offers.size} offers:"
      offers.each_with_index do |o, i|
        puts "  [#{i}] #{o}"
      end
    end
    offers
  end

  # List active instances
  def instances
    list = @provider.list
    if list.empty?
      puts "No active instances."
    else
      puts "Active instances:"
      list.each { |i| puts "  #{i}" }
    end
    list
  end

  # Rent a GPU, deploy code, run experiments, collect results, destroy
  def run_experiments(
    data : String,
    config : String,
    runs : Array(String),
    eval_file : String? = nil,
    gpu : String? = nil,
    max_price : Float64 = 1.5,
    min_reliability : Float64 = 0.95,
    disk_gb : Int32 = 16
  )
    # 1. Find cheapest suitable GPU
    puts "=== Searching for GPU ==="
    offers = @provider.search(gpu: gpu, max_price: max_price,
                               min_reliability: min_reliability, limit: 5)
    if offers.empty?
      puts "No suitable GPU offers found. Try increasing max_price or relaxing constraints."
      return
    end

    offer = offers.first
    puts "Selected: #{offer}"
    est_cost = offer.price_per_hour * 0.5 # rough estimate: 30 min
    puts "Estimated cost: ~$#{"%.2f" % est_cost} (assuming ~30 min)"
    puts

    # 2. Create instance
    puts "=== Creating instance ==="
    instance = @provider.create(offer, disk_gb: disk_gb)
    puts "Instance created: ##{instance.id}"
    puts

    begin
      # 3. Wait for ready
      puts "=== Waiting for instance to be ready ==="
      instance = @provider.wait_ready(instance.id, timeout: 600)
      puts "Instance ready: #{instance}"
      puts

      # 4. Upload code and data
      puts "=== Deploying ==="
      deploy_files = DEPLOY_FILES + [data, config]
      deploy_files << eval_file if eval_file
      deploy_files.each do |f|
        if File.exists?(f)
          puts "  Uploading #{f}..."
          @provider.upload(instance.id, f)
        else
          puts "  Warning: #{f} not found, skipping"
        end
      end

      # Make binary executable
      @provider.exec(instance.id, "chmod +x /workspace/bin/microgpt")
      puts "  Deploy complete."
      puts

      # 5. Run experiments
      puts "=== Running #{runs.size} experiments ==="
      runs.each_with_index do |run_id, i|
        log = "/workspace/run_#{run_id}.log"
        cmd = "/workspace/bin/microgpt /workspace/#{data} --config /workspace/#{config} --run #{run_id}"
        cmd += " --eval /workspace/#{eval_file}" if eval_file

        puts "  [#{i + 1}/#{runs.size}] #{run_id} (log: #{log})..."

        # Launch in background with log, then poll until done
        @provider.exec(instance.id, "nohup sh -c '#{cmd} > #{log} 2>&1; echo DONE >> #{log}' &")
        sleep 2.seconds

        # Tail log until we see DONE or process exits
        last_lines = 0
        loop do
          result = @provider.exec(instance.id, "wc -l < #{log} 2>/dev/null || echo 0")
          current_lines = result.output.strip.to_i rescue 0

          if current_lines > last_lines
            tail = @provider.exec(instance.id, "tail -n +#{last_lines + 1} #{log} | head -n #{current_lines - last_lines}")
            output = tail.output
            if output.includes?("DONE")
              output = output.gsub("DONE\n", "").gsub("DONE", "")
              print output unless output.empty?
              break
            end
            print output
            last_lines = current_lines
          end

          # Check if process is still running
          check = @provider.exec(instance.id, "pgrep -f 'microgpt.*#{run_id}' > /dev/null 2>&1 && echo RUNNING || echo STOPPED")
          if check.output.strip == "STOPPED" && current_lines == last_lines
            # Grab any remaining output
            tail = @provider.exec(instance.id, "tail -n +#{last_lines + 1} #{log} 2>/dev/null")
            print tail.output.gsub("DONE\n", "").gsub("DONE", "")
            break
          end

          sleep 5.seconds
        end
        puts
      end

      # 6. Download results
      puts "=== Downloading results ==="
      results_path = File.join(File.dirname(data), "results.tsv")
      @provider.download(instance.id, "/workspace/#{results_path}", ".")
      puts "Results saved to #{results_path}"

    ensure
      # 7. Always destroy to avoid charges
      puts
      if instance.id > 0
        puts "=== Destroying instance ##{instance.id} ==="
        if @provider.destroy(instance.id)
          puts "Instance destroyed."
        else
          puts "WARNING: Failed to destroy instance ##{instance.id}! Check #{@provider.name} console."
        end
      else
        puts "=== Skipping destroy (no valid instance ID) ==="
      end
    end
  end

  # Destroy a specific instance
  def destroy(instance_id : Int64)
    if @provider.destroy(instance_id)
      puts "Instance ##{instance_id} destroyed."
    else
      puts "Failed to destroy instance ##{instance_id}."
    end
  end

  # Destroy all active instances
  def destroy_all
    list = @provider.list
    if list.empty?
      puts "No active instances."
      return
    end
    list.each do |inst|
      puts "Destroying ##{inst.id}..."
      @provider.destroy(inst.id)
    end
    puts "#{list.size} instances destroyed."
  end
end

end
