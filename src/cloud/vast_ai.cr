require "./provider"

module Cloud

class VastAI < Provider
  BASE_URL = "https://cloud.vast.ai"

  @api_key : String
  @http : HTTP::Client?

  def initialize(@api_key : String)
  end

  def initialize
    # Read API key: env var → XDG config → legacy fallback
    @api_key = ENV["VAST_API_KEY"]? || begin
      xdg_config = ENV["XDG_CONFIG_HOME"]? || Path.home.join(".config").to_s
      key_path = File.join(xdg_config, "vastai", "api_key")
      if File.exists?(key_path)
        File.read(key_path).strip
      else
        raise "No vast.ai API key found. Set VAST_API_KEY or create #{key_path}"
      end
    end
  end

  def name : String
    "vast.ai"
  end

  def search(gpu : String? = nil, min_gpu_ram : Int32 = 0, max_price : Float64 = 10.0,
             min_reliability : Float64 = 0.9, num_gpus : Int32 = 1, limit : Int32 = 10) : Array(Offer)
    filters = {} of String => JSON::Any

    filters["verified"] = json_filter("eq", true)
    filters["rentable"] = json_filter("eq", true)
    filters["num_gpus"] = json_filter("gte", num_gpus)
    filters["dph_total"] = json_filter("lte", max_price)
    filters["reliability"] = json_filter("gte", min_reliability)
    filters["cuda_max_good"] = json_filter("gte", 13.1)

    if g = gpu
      filters["gpu_name"] = json_filter("eq", g)
    end

    if min_gpu_ram > 0
      filters["gpu_ram"] = json_filter("gte", min_gpu_ram * 1024) # API uses MB
    end

    body = filters.merge({
      "limit" => JSON::Any.new(limit.to_i64),
      "order" => JSON::Any.new([
        JSON::Any.new([JSON::Any.new("dph_total"), JSON::Any.new("asc")]),
      ]),
    })

    resp = api_post("/api/v0/bundles/", body)
    parse_offers(resp)
  end

  def create(offer : Offer, label : String = "microgpt",
             disk_gb : Int32 = 16, image : String = "nvidia/cuda:13.1.0-runtime-ubuntu24.04") : Instance
    body = {
      "image"   => JSON::Any.new(image),
      "runtype" => JSON::Any.new("ssh_direct"),
      "disk"    => JSON::Any.new(disk_gb.to_i64),
      "label"   => JSON::Any.new(label),
      "onstart" => JSON::Any.new("apt-get update && apt-get install -y libopenblas64-0 libgc1 libyaml-0-2 && ln -sf /usr/lib/x86_64-linux-gnu/libopenblas64.so.0 /usr/lib/x86_64-linux-gnu/libopenblas_64.so.0"),
    }

    resp = api_put("/api/v0/asks/#{offer.id}/", body)
    json = JSON.parse(resp)

    contract_id = json["new_contract"]?.try(&.as_i64)
    raise "vast.ai create failed: no contract ID in response: #{resp}" unless contract_id

    # Return minimal instance — wait_ready will poll for full details
    Instance.new(
      id: contract_id,
      status: "loading",
      gpu_name: offer.gpu_name,
      price_per_hour: offer.price_per_hour,
    )
  end

  def status(instance_id : Int64) : Instance
    resp = api_get("/api/v0/instances/#{instance_id}/")
    json = JSON.parse(resp)
    # Single instance endpoint wraps data in {"instances": {...}}
    inst = json["instances"]? || json
    parse_instance(inst)
  end

  def list : Array(Instance)
    resp = api_get("/api/v0/instances/")
    json = JSON.parse(resp)
    instances = json["instances"].as_a
    instances.map { |j| parse_instance(j) }
  end

  def exec(instance_id : Int64, command : String) : ExecResult
    # Try API exec first, fall back to SSH
    begin
      body = {"command" => JSON::Any.new(command)}
      resp = api_put("/api/v0/instances/command/#{instance_id}/", body)
      ExecResult.new(resp)
    rescue
      # Fall back to SSH exec
      inst = status(instance_id)
      host = inst.ssh_host || raise "No SSH host for instance ##{instance_id}"
      port = inst.ssh_port || raise "No SSH port for instance ##{instance_id}"
      ssh_exec(host, port, command)
    end
  end

  def destroy(instance_id : Int64) : Bool
    resp = api_delete("/api/v0/instances/#{instance_id}/")
    json = JSON.parse(resp)
    json["success"]?.try(&.as_bool) || false
  end

  # --- Private helpers ---

  private def client : HTTP::Client
    @http ||= begin
      c = HTTP::Client.new(URI.parse(BASE_URL))
      c.connect_timeout = 30.seconds
      c.read_timeout = 60.seconds
      c
    end
  end

  private def headers : HTTP::Headers
    HTTP::Headers{
      "Authorization" => "Bearer #{@api_key}",
      "Content-Type"  => "application/json",
      "Accept"        => "application/json",
    }
  end

  private def api_get(path : String) : String
    resp = client.get(path, headers: headers)
    check_response(resp)
    resp.body
  end

  private def api_post(path : String, body) : String
    resp = client.post(path, headers: headers, body: body.to_json)
    check_response(resp)
    resp.body
  end

  private def api_put(path : String, body) : String
    resp = client.put(path, headers: headers, body: body.to_json)
    check_response(resp)
    resp.body
  end

  private def api_delete(path : String) : String
    resp = client.delete(path, headers: headers)
    check_response(resp)
    resp.body
  end

  private def check_response(resp : HTTP::Client::Response)
    return if resp.status_code >= 200 && resp.status_code < 300
    msg = begin
      JSON.parse(resp.body)["msg"]?.try(&.as_s) || resp.body
    rescue
      resp.body
    end
    raise "vast.ai API error #{resp.status_code}: #{msg}"
  end

  # --- JSON helpers: null-safe field extraction ---

  private def jstr(j : JSON::Any, key : String, fallback : String = "") : String
    j[key]?.try(&.as_s?) || fallback
  end

  private def jstr?(j : JSON::Any, *keys : String) : String?
    keys.each { |k| jstr = j[k]?.try(&.as_s?); return jstr if jstr }
    nil
  end

  private def jint(j : JSON::Any, key : String, fallback : Int64 = 0_i64) : Int64
    j[key]?.try(&.as_i64?) || fallback
  end

  private def jint?(j : JSON::Any, *keys : String) : Int32?
    keys.each { |k| v = j[k]?.try(&.as_i?); return v.to_i32 if v }
    nil
  end

  private def jfloat(j : JSON::Any, key : String, fallback : Float64 = 0.0) : Float64
    j[key]?.try(&.as_f?) || fallback
  end

  private def jfloat?(j : JSON::Any, *keys : String) : Float64?
    keys.each { |k| v = j[k]?.try(&.as_f?); return v if v }
    nil
  end

  private def json_filter(op : String, value) : JSON::Any
    JSON::Any.new({op => JSON::Any.new(value)} of String => JSON::Any)
  end

  private def json_filter(op : String, value : Float64) : JSON::Any
    JSON::Any.new({op => JSON::Any.new(value)} of String => JSON::Any)
  end

  private def json_filter(op : String, value : Bool) : JSON::Any
    JSON::Any.new({op => JSON::Any.new(value)} of String => JSON::Any)
  end

  private def json_filter(op : String, value : Int32) : JSON::Any
    JSON::Any.new({op => JSON::Any.new(value.to_i64)} of String => JSON::Any)
  end

  private def parse_offers(body : String) : Array(Offer)
    json = JSON.parse(body)
    offers = json["offers"]?.try(&.as_a) || return [] of Offer

    offers.compact_map do |o|
      Offer.new(
        id: o["id"].as_i64,
        gpu_name: jstr(o, "gpu_name", "unknown"),
        num_gpus: (o["num_gpus"]?.try(&.as_i?) || 1).to_i32,
        gpu_ram_mb: (jfloat?(o, "gpu_total_ram", "gpu_ram") || 0.0).to_i64,
        cpu_cores: (jfloat?(o, "cpu_cores_effective") || jint?(o, "cpu_cores") || 0).to_i32,
        cpu_ram_mb: jfloat(o, "cpu_ram").to_i64,
        disk_gb: jfloat(o, "disk_space"),
        price_per_hour: jfloat(o, "dph_total"),
        reliability: jfloat?(o, "reliability2", "reliability") || 0.0,
        location: jstr(o, "geolocation", "unknown"),
      )
    rescue
      nil
    end
  end

  private def parse_instance(j : JSON::Any) : Instance
    Instance.new(
      id: jint(j, "id"),
      status: jstr?(j, "actual_status", "cur_state") || "unknown",
      gpu_name: jstr(j, "gpu_name", "unknown"),
      price_per_hour: jfloat(j, "dph_total"),
      ssh_host: jstr?(j, "ssh_host", "public_ipaddr"),
      ssh_port: jint?(j, "ssh_port", "direct_port_start"),
      label: j["label"]?.try(&.as_s?),
    )
  end
end

end
