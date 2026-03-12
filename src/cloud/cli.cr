require "./runner"

# CLI for cloud GPU experiment management
#
# Usage:
#   cloud search [--gpu RTX_4090] [--max-price 1.5]
#   cloud list
#   cloud run --config models.yml --runs add-5e-calc,add-4e-baseline --data data/addition.txt
#   cloud destroy <instance_id>
#   cloud destroy-all

module Cloud
  def self.main
    if ARGV.empty?
      print_usage
      return
    end

    provider = VastAI.new
    runner = Runner.new(provider)

    case ARGV[0]
    when "search"
      gpu = flag_value("--gpu")
      max_price = (flag_value("--max-price") || "2.0").to_f
      runner.search(gpu: gpu, max_price: max_price)

    when "list"
      runner.instances

    when "run"
      config = flag_value("--config") || "models.yml"
      data = flag_value("--data") || "data/addition.txt"
      eval_file = flag_value("--eval")
      gpu = flag_value("--gpu")
      max_price = (flag_value("--max-price") || "1.5").to_f
      runs_str = flag_value("--runs")

      unless runs_str
        STDERR.puts "Error: --runs required (comma-separated model IDs)"
        return
      end

      runs = runs_str.split(",").map(&.strip)
      runner.run_experiments(
        data: data,
        config: config,
        runs: runs,
        eval_file: eval_file,
        gpu: gpu,
        max_price: max_price,
      )

    when "destroy"
      id = ARGV[1]?.try(&.to_i64)
      if id
        runner.destroy(id)
      else
        STDERR.puts "Error: instance ID required"
      end

    when "destroy-all"
      runner.destroy_all

    else
      print_usage
    end
  end

  private def self.flag_value(flag : String) : String?
    idx = ARGV.index(flag)
    return nil unless idx
    ARGV[idx + 1]?
  end

  private def self.print_usage
    puts "Cloud GPU experiment runner"
    puts
    puts "Commands:"
    puts "  search [--gpu NAME] [--max-price N]     Search available GPUs"
    puts "  list                                     List active instances"
    puts "  run --runs ID,ID [OPTIONS]               Run experiments on rented GPU"
    puts "    --config FILE   Model config (default: models.yml)"
    puts "    --data FILE     Training data (default: data/addition.txt)"
    puts "    --eval FILE     Eval prompts file"
    puts "    --gpu NAME      GPU model filter (e.g. RTX_4090)"
    puts "    --max-price N   Max $/hr (default: 1.5)"
    puts "  destroy ID                               Destroy an instance"
    puts "  destroy-all                              Destroy all instances"
    puts
    puts "Environment:"
    puts "  VAST_API_KEY      vast.ai API key (or put in ~/.config/vastai/api_key)"
  end
end

Cloud.main
