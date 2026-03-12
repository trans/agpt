require "http/client"
require "json"
require "socket"

# Cloud GPU provider abstraction for running experiments on rented hardware.
#
# Usage:
#   provider = Cloud::VastAI.new(api_key: "...")
#   offers = provider.search(gpu: "RTX 4090", max_price: 1.0)
#   instance = provider.create(offers.first)
#   provider.wait_ready(instance.id)
#   provider.upload(instance.id, "build/", "data/")
#   provider.exec(instance.id, "./microgpt data/addition.txt --config models.yml --run add-5e-calc")
#   provider.download(instance.id, "data/results.tsv")
#   provider.destroy(instance.id)

module Cloud

# A GPU offer from a provider
struct Offer
  include JSON::Serializable

  getter id : Int64
  getter gpu_name : String
  getter num_gpus : Int32
  getter gpu_ram_mb : Int64       # GPU VRAM in MB
  getter cpu_cores : Int32
  getter cpu_ram_mb : Int64       # System RAM in MB
  getter disk_gb : Float64
  getter price_per_hour : Float64 # $/hr
  getter reliability : Float64    # 0.0-1.0
  getter location : String

  def initialize(@id, @gpu_name, @num_gpus, @gpu_ram_mb, @cpu_cores,
                 @cpu_ram_mb, @disk_gb, @price_per_hour, @reliability, @location)
  end

  def to_s(io : IO)
    io << "##{@id} #{@gpu_name}"
    io << "×#{@num_gpus}" if @num_gpus > 1
    io << " #{@gpu_ram_mb / 1024}GB"
    io << " $#{"%.3f" % @price_per_hour}/hr"
    io << " rel=#{"%.2f" % @reliability}"
    io << " [#{@location}]"
  end
end

# A running instance
struct Instance
  include JSON::Serializable

  getter id : Int64
  getter status : String      # "running", "loading", "exited", etc.
  getter gpu_name : String
  getter ssh_host : String?
  getter ssh_port : Int32?
  getter price_per_hour : Float64
  getter label : String?

  def initialize(@id, @status, @gpu_name, @price_per_hour,
                 @ssh_host = nil, @ssh_port = nil, @label = nil)
  end

  def ready? : Bool
    @status == "running" && @ssh_host != nil
  end

  def to_s(io : IO)
    io << "##{@id} #{@gpu_name} [#{@status}]"
    io << " $#{"%.3f" % @price_per_hour}/hr"
    if (h = @ssh_host) && (p = @ssh_port)
      io << " ssh://#{h}:#{p}"
    end
    io << " (#{@label})" if @label
  end
end

# Result of a remote command execution
struct ExecResult
  getter output : String
  getter exit_code : Int32

  def initialize(@output, @exit_code = 0)
  end

  def success? : Bool
    @exit_code == 0
  end
end

# Abstract provider interface — implement this for each cloud GPU service
abstract class Provider
  # Human-readable provider name
  abstract def name : String

  # Search for available GPU offers
  abstract def search(
    gpu : String? = nil,
    min_gpu_ram : Int32 = 0,
    max_price : Float64 = 10.0,
    min_reliability : Float64 = 0.9,
    num_gpus : Int32 = 1,
    limit : Int32 = 10
  ) : Array(Offer)

  # Create an instance from an offer
  abstract def create(offer : Offer, label : String = "microgpt",
                      disk_gb : Int32 = 16, image : String = "nvidia/cuda:13.1.0-runtime-ubuntu24.04") : Instance

  # Get current instance status
  abstract def status(instance_id : Int64) : Instance

  # List all active instances
  abstract def list : Array(Instance)

  # Execute a command on the instance
  abstract def exec(instance_id : Int64, command : String) : ExecResult

  # Destroy an instance (irreversible)
  abstract def destroy(instance_id : Int64) : Bool

  # Wait for instance to be ready (SSH available)
  def wait_ready(instance_id : Int64, timeout : Int32 = 300, poll_interval : Int32 = 5) : Instance
    elapsed = 0
    inst = Instance.new(id: instance_id, status: "loading", gpu_name: "unknown", price_per_hour: 0.0)
    loop do
      inst = status(instance_id)
      if inst.ready?
        # Verify SSH is actually accepting connections
        if ssh_connectable?(inst)
          return inst
        end
      end
      raise "Instance ##{instance_id} failed: #{inst.status}" if inst.status == "exited"
      raise "Timeout waiting for instance ##{instance_id}" if elapsed >= timeout
      sleep poll_interval.seconds
      elapsed += poll_interval
    end
  end

  # Check if SSH port is actually accepting connections
  private def ssh_connectable?(inst : Instance) : Bool
    host = inst.ssh_host || return false
    port = inst.ssh_port || return false
    begin
      sock = TCPSocket.new(host, port, connect_timeout: 5)
      sock.close
      true
    rescue
      false
    end
  end

  # SSH key path — override with CLOUD_SSH_KEY env var
  def ssh_key : String
    ENV["CLOUD_SSH_KEY"]? || File.expand_path("~/.ssh/id_vastai")
  end

  private def ssh_opts : Array(String)
    key = ssh_key
    opts = ["-o", "StrictHostKeyChecking=no"]
    if File.exists?(key)
      opts += ["-i", key]
    end
    opts
  end

  # Upload files/directories to instance via SCP
  def upload(instance_id : Int64, *paths : String, remote_dir : String = "/workspace")
    inst = status(instance_id)
    host = inst.ssh_host || raise "No SSH host for instance ##{instance_id}"
    port = inst.ssh_port || raise "No SSH port for instance ##{instance_id}"

    # Ensure remote directory structure exists
    remote_dirs = paths.compact_map { |p| File.directory?(p) ? nil : "#{remote_dir}/#{File.dirname(p)}" }.uniq
    remote_dirs << remote_dir
    mkdir_cmd = "mkdir -p #{remote_dirs.join(" ")}"
    ssh_exec(host, port, mkdir_cmd)

    paths.each do |path|
      remote_path = "#{remote_dir}/#{File.dirname(path)}/"
      args = ssh_opts + ["-P", port.to_s]
      if File.directory?(path)
        args << "-r"
        remote_path = "#{remote_dir}/"
      end
      args += [path, "root@#{host}:#{remote_path}"]
      system("scp #{Process.quote(args)}")
    end
  end

  # Download files from instance via SCP
  def download(instance_id : Int64, remote_path : String, local_path : String = ".")
    inst = status(instance_id)
    host = inst.ssh_host || raise "No SSH host for instance ##{instance_id}"
    port = inst.ssh_port || raise "No SSH port for instance ##{instance_id}"

    args = ssh_opts + ["-r", "-P", port.to_s, "root@#{host}:#{remote_path}", local_path]
    system("scp #{Process.quote(args)}")
  end

  # SSH into instance interactively
  def ssh(instance_id : Int64)
    inst = status(instance_id)
    host = inst.ssh_host || raise "No SSH host for instance ##{instance_id}"
    port = inst.ssh_port || raise "No SSH port for instance ##{instance_id}"

    args = ssh_opts + ["-p", port.to_s, "root@#{host}"]
    system("ssh #{Process.quote(args)}")
  end

  # Run a command via SSH (for providers without exec API)
  protected def ssh_exec(host : String, port : Int32, command : String) : ExecResult
    output = IO::Memory.new
    err = IO::Memory.new
    args = ssh_opts + ["-p", port.to_s, "root@#{host}", command]
    status = Process.run("ssh", args: args, output: output, error: err)
    ExecResult.new(output.to_s + err.to_s, status.exit_code)
  end
end

end
