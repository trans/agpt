// YAML config loader for v1 (src/cuda/agpt_train.cu).
//
// Mirrors src/cudax/yaml_config_v2.cuh's pattern: read a new-schema YAML
// (docs/yaml-schema.md), validate the field registry strictly, and
// populate the locals that v1's main() would otherwise read from
// --model / --trie-dir / --epochs / etc.
//
// Out-of-schema research toggles (rope-mode, lightning-*, per-rc-adam,
// curriculum, etc.) stay CLI-only. Use legacy CLI invocation when those
// matter; --config covers the canonical schema only.
//
// Depends on enums + helper parsers defined just before the include
// site in agpt_train.cu:
//   OptimizerKind, LRSchedule, MassWeightMode
//   parse_optimizer_kind_v1(), parse_lr_schedule_v1(),
//   parse_mass_weight_v1(), parse_fire_norm_v1()

#ifndef AGPT_V1_YAML_CONFIG_CUH
#define AGPT_V1_YAML_CONFIG_CUH

struct YamlScalarV1 {
    std::string value;
    yam_mark mark{};
};

struct YamlDocV1 {
    std::unordered_map<std::string, YamlScalarV1> scalars;
    std::unordered_set<std::string> blocks;
};

struct YamlConfigV1 {
    // Paths
    std::string model_path;        // model.init_file
    std::string trie_dir;          // trie.path
    std::string corpus_path;       // corpus.path (informational; v1 reads from trie)
    std::string save_path;         // model.save_file

    // Whether model.init_file was absent → fresh init mode
    bool init_mode = false;

    // Required scalar
    int epochs = 1;

    // Optional scalars (v1 defaults)
    float lr = 3e-4f;
    float entropy_lambda = 0.0f;
    float momentum_beta = 0.9f;
    float rmsprop_beta = 0.999f;
    float weight_decay = 0.0f;
    float grad_clip_norm = 0.0f;
    int warmup_epochs = 0;
    int partition_depth = 1;
    int chunk_queries = 0;

    // Enums
    OptimizerKind optimizer = OptimizerKind::Adam;
    LRSchedule lr_schedule = LRSchedule::Constant;
    MassWeightMode mass_weight = MassWeightMode::Linear;

    // Bool toggles
    bool ce_only = false;
    bool fire_norm_by_mass = true;
    bool fire_norm_by_weight = false;
    bool fire_norm_none = false;
    bool explicit_anc_grad = false;
    bool ablate_anc_grad = false;

    // Seeds
    unsigned shuffle_seed = 0xa17b1edu;
    bool seed_set = false;
    int seed = 42;

    // Init-mode architecture (only when model.init_file absent)
    int init_d_model = 64;
    int init_n_heads = 4;
    int init_n_layers = 2;
    int init_d_ff = 256;
    uint32_t init_seed = 42;

    // Cross-check tracking
    bool has_max_depth = false;
    bool has_seq_len = false;
    int max_depth = 0;
    int seq_len = 0;

    bool has_model_d_model = false;
    bool has_model_n_layers = false;
    bool has_model_n_heads = false;
    bool has_model_d_ff = false;
    bool has_model_head_dim = false;
    int model_d_model = 0;
    int model_n_layers = 0;
    int model_n_heads = 0;
    int model_d_ff = 0;
    int model_head_dim = 0;
};

static std::string yam_str_to_std_v1(yam_str s) {
    if (!s.data || s.len == 0) return std::string();
    return std::string(s.data, s.len);
}

static std::string yaml_path_join_v1(const std::string& prefix, const std::string& key) {
    if (prefix.empty()) return key;
    return prefix + "." + key;
}

static bool parse_yaml_node_v1(const std::vector<yam_event>& events,
                               size_t& idx,
                               const std::string& path,
                               YamlDocV1& doc,
                               std::string& error) {
    if (idx >= events.size()) {
        error = "unexpected end of YAML event stream";
        return false;
    }
    const yam_event& evt = events[idx++];
    if (evt.type == YAM_EVT_SCALAR) {
        if (path.empty()) {
            error = "top-level YAML document must be a mapping";
            return false;
        }
        doc.scalars[path] = YamlScalarV1{yam_str_to_std_v1(evt.value), evt.start};
        return true;
    }
    if (evt.type == YAM_EVT_MAPPING_START) {
        if (!path.empty()) doc.blocks.insert(path);
        while (idx < events.size() && events[idx].type != YAM_EVT_MAPPING_END) {
            const yam_event& key_evt = events[idx++];
            if (key_evt.type != YAM_EVT_SCALAR) {
                error = "YAML mapping keys must be scalars";
                return false;
            }
            std::string key = yam_str_to_std_v1(key_evt.value);
            if (key.empty()) {
                error = "YAML mapping keys must not be empty";
                return false;
            }
            if (key.find('.') != std::string::npos) {
                error = "YAML mapping keys must use nested maps, not dotted keys: " + key;
                return false;
            }
            if (!parse_yaml_node_v1(events, idx, yaml_path_join_v1(path, key), doc, error)) {
                return false;
            }
        }
        if (idx >= events.size() || events[idx].type != YAM_EVT_MAPPING_END) {
            error = "unterminated YAML mapping";
            return false;
        }
        idx++;
        return true;
    }
    if (evt.type == YAM_EVT_SEQUENCE_START) {
        error = "YAML sequences are not part of the trainer config schema at " + path;
        return false;
    }
    if (evt.type == YAM_EVT_ALIAS) {
        error = "unresolved YAML alias at " + path;
        return false;
    }
    error = "unexpected YAML event while reading " + path + ": " + yam_event_type_str(evt.type);
    return false;
}

static bool load_yaml_doc_v1(const char* path, YamlDocV1& doc) {
    yam_arena* arena = yam_arena_new(65536);
    if (!arena) {
        std::fprintf(stderr, "agpt_train: failed to allocate YAML arena\n");
        return false;
    }
    yam_str data = yam_read_file(path, arena);
    if (!data.data) {
        std::fprintf(stderr, "agpt_train: failed to read YAML config: %s\n", path);
        yam_arena_free(arena);
        return false;
    }
    yam_parser* parser = yam_parser_new(data.data, data.len, arena);
    if (!parser) {
        std::fprintf(stderr, "agpt_train: failed to allocate YAML parser\n");
        yam_arena_free(arena);
        return false;
    }
    yam_parser_set_merge(parser, true);
    yam_parser_set_resolve(parser, true);

    std::vector<yam_event> events;
    while (true) {
        yam_event evt{};
        yam_status st = yam_parse_next(parser, &evt);
        if (st != YAM_OK) {
            yam_mark mark = yam_parser_error_mark(parser);
            std::fprintf(stderr, "agpt_train: YAML parse error at %zu:%zu: %s\n",
                         mark.line, mark.col, yam_parser_error(parser));
            yam_parser_free(parser);
            yam_arena_free(arena);
            return false;
        }
        events.push_back(evt);
        if (evt.type == YAM_EVT_STREAM_END) break;
    }
    yam_parser_free(parser);

    size_t idx = 0;
    if (idx < events.size() && events[idx].type == YAM_EVT_STREAM_START) idx++;
    if (idx < events.size() && events[idx].type == YAM_EVT_DOC_START) idx++;
    std::string error;
    if (!parse_yaml_node_v1(events, idx, "", doc, error)) {
        std::fprintf(stderr, "agpt_train: YAML config error: %s\n", error.c_str());
        yam_arena_free(arena);
        return false;
    }
    if (idx < events.size() && events[idx].type == YAM_EVT_DOC_END) idx++;
    if (idx >= events.size() || events[idx].type != YAM_EVT_STREAM_END) {
        std::fprintf(stderr, "agpt_train: YAML config must contain exactly one document\n");
        yam_arena_free(arena);
        return false;
    }

    yam_arena_free(arena);
    return true;
}

static bool yaml_is_ignored_field_v1(const std::string& path) {
    if (path == "description" || path == "experiment" || path == "run_slug") return true;
    if (path == "corpus.heldout") return true;
    if (path.rfind("corpus.carve.", 0) == 0) return true;
    if (path.rfind("eval.", 0) == 0) return true;
    return false;
}

static bool yaml_is_consumed_field_v1(const std::string& path) {
    static const std::unordered_set<std::string> fields = {
        "corpus.path",
        "corpus.vocab_source",
        "trie.max_depth",
        "trie.prune_min_mass",
        "trie.prune_min_depth",
        "trie.path",
        "trie.virtual_tree",
        "model.d_model",
        "model.n_layers",
        "model.n_heads",
        "model.d_ff",
        "model.head_dim",
        "model.init_file",
        "model.init_seed",
        "model.save_file",
        "train.budget.unit",
        "train.budget.value",
        "train.seed",
        "train.quiet",
        "train.optimizer.name",
        "train.optimizer.lr",
        "train.optimizer.beta",
        "train.optimizer.momentum_beta",
        "train.optimizer.weight_decay",
        "train.optimizer.grad_clip_norm",
        "train.lr_schedule.name",
        "train.lr_schedule.warmup_epochs",
        "train.seq_len",
        "train.max_depth",
        "train.partition_depth",
        "train.chunk_queries",
        "train.anc_grad",
        "train.mass_weight",
        "train.fire_norm",
        "train.entropy_lambda",
        "train.ce_only",
    };
    return fields.find(path) != fields.end();
}

static bool yaml_is_known_non_v1_field_v1(const std::string& path) {
    // train.growth.* is microgpt+v2 today (v1 has no growth yet — task #31).
    // train.{backend,heads,lookahead} are microgpt-only.
    static const std::unordered_set<std::string> fields = {
        "train.growth.divisions",
        "train.growth.min_epochs",
        "train.growth.epoch_ramp",
        "train.backend",
        "train.heads",
        "train.lookahead",
    };
    return fields.find(path) != fields.end();
}

static bool yaml_is_experimental_field_v1(const std::string& path) {
    return path.rfind("experimental.", 0) == 0 && path.size() > std::strlen("experimental.");
}

static bool warn_unknown_experimental_flags_v1(const YamlDocV1& doc) {
    std::vector<std::string> flags;
    for (const auto& item : doc.scalars) {
        const std::string& path = item.first;
        if (!yaml_is_experimental_field_v1(path)) continue;
        flags.push_back(path.substr(std::strlen("experimental.")));
    }
    std::sort(flags.begin(), flags.end());
    for (const std::string& flag : flags) {
        std::fprintf(stderr, "WARN: unknown experimental flag %s; ignoring\n", flag.c_str());
    }
    return true;
}

static bool validate_yaml_registry_v1(const YamlDocV1& doc) {
    static const std::unordered_set<std::string> blocks = {
        "corpus",
        "corpus.carve",
        "trie",
        "model",
        "train",
        "train.budget",
        "train.optimizer",
        "train.lr_schedule",
        "train.growth",
        "eval",
        "experimental",
    };
    for (const auto& block : doc.blocks) {
        if (blocks.find(block) == blocks.end()) {
            std::fprintf(stderr, "agpt_train: unknown YAML block: %s\n", block.c_str());
            return false;
        }
    }
    for (const auto& item : doc.scalars) {
        const std::string& path = item.first;
        if (yaml_is_experimental_field_v1(path)) continue;
        if (yaml_is_ignored_field_v1(path) || yaml_is_consumed_field_v1(path)) continue;
        if (yaml_is_known_non_v1_field_v1(path)) {
            std::fprintf(stderr, "agpt_train: YAML field is not supported by v1: %s\n", path.c_str());
        } else {
            std::fprintf(stderr, "agpt_train: unknown YAML field: %s\n", path.c_str());
        }
        return false;
    }
    return true;
}

static const YamlScalarV1* yaml_find_v1(const YamlDocV1& doc, const char* path) {
    auto it = doc.scalars.find(path);
    return it == doc.scalars.end() ? nullptr : &it->second;
}

static bool yaml_parse_long_v1(const char* field, const YamlScalarV1& scalar, long& out) {
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(scalar.value.c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0') {
        std::fprintf(stderr, "agpt_train: YAML field %s must be an integer (got %s)\n",
                     field, scalar.value.c_str());
        return false;
    }
    out = value;
    return true;
}

static bool yaml_parse_int_v1(const char* field, const YamlScalarV1& scalar, int& out) {
    long value = 0;
    if (!yaml_parse_long_v1(field, scalar, value)) return false;
    if (value < INT_MIN || value > INT_MAX) {
        std::fprintf(stderr, "agpt_train: YAML field %s is out of int range: %s\n",
                     field, scalar.value.c_str());
        return false;
    }
    out = (int)value;
    return true;
}

static bool yaml_parse_float_v1(const char* field, const YamlScalarV1& scalar, float& out) {
    errno = 0;
    char* end = nullptr;
    float value = std::strtof(scalar.value.c_str(), &end);
    if (errno != 0 || !end || *end != '\0') {
        std::fprintf(stderr, "agpt_train: YAML field %s must be a number (got %s)\n",
                     field, scalar.value.c_str());
        return false;
    }
    out = value;
    return true;
}

static bool yaml_parse_bool_v1(const char* field, const YamlScalarV1& scalar, bool& out) {
    if (scalar.value == "true" || scalar.value == "True" || scalar.value == "TRUE") {
        out = true;
        return true;
    }
    if (scalar.value == "false" || scalar.value == "False" || scalar.value == "FALSE") {
        out = false;
        return true;
    }
    std::fprintf(stderr, "agpt_train: YAML field %s must be true or false (got %s)\n",
                 field, scalar.value.c_str());
    return false;
}

static bool yaml_get_int_v1(const YamlDocV1& doc, const char* field, int& out, bool* present = nullptr) {
    const YamlScalarV1* scalar = yaml_find_v1(doc, field);
    if (present) *present = scalar != nullptr;
    if (!scalar) return true;
    return yaml_parse_int_v1(field, *scalar, out);
}

static bool yaml_get_float_v1(const YamlDocV1& doc, const char* field, float& out, bool* present = nullptr) {
    const YamlScalarV1* scalar = yaml_find_v1(doc, field);
    if (present) *present = scalar != nullptr;
    if (!scalar) return true;
    return yaml_parse_float_v1(field, *scalar, out);
}

static bool yaml_get_bool_v1(const YamlDocV1& doc, const char* field, bool& out, bool* present = nullptr) {
    const YamlScalarV1* scalar = yaml_find_v1(doc, field);
    if (present) *present = scalar != nullptr;
    if (!scalar) return true;
    return yaml_parse_bool_v1(field, *scalar, out);
}

static bool yaml_expect_string_v1(const YamlDocV1& doc, const char* field, std::string& out, bool required) {
    const YamlScalarV1* scalar = yaml_find_v1(doc, field);
    if (!scalar) {
        if (required) {
            std::fprintf(stderr, "agpt_train: YAML field %s is required\n", field);
            return false;
        }
        return true;
    }
    out = scalar->value;
    return true;
}

static bool apply_yaml_config_v1(const char* config_path, YamlConfigV1& yaml_cfg) {
    YamlDocV1 doc;
    if (!load_yaml_doc_v1(config_path, doc)) return false;

    // v1 has no growth mode yet (task #31). Check this BEFORE the registry
    // validation so the user gets a "what + why + workaround" message
    // rather than a generic "field not supported" on one of the inner keys.
    bool has_growth_block = doc.blocks.find("train.growth") != doc.blocks.end()
        || yaml_find_v1(doc, "train.growth.divisions")
        || yaml_find_v1(doc, "train.growth.min_epochs")
        || yaml_find_v1(doc, "train.growth.epoch_ramp");
    if (has_growth_block) {
        std::fprintf(stderr,
                     "agpt_train: train.growth is not supported by v1 yet (task #31). "
                     "Use --trainer v2 for growth training, or run v1 without a growth block.\n");
        return false;
    }

    if (!validate_yaml_registry_v1(doc)) return false;
    if (!warn_unknown_experimental_flags_v1(doc)) return false;

    // ---- Paths ----
    if (!yaml_expect_string_v1(doc, "model.init_file", yaml_cfg.model_path, false)) return false;
    yaml_cfg.init_mode = yaml_cfg.model_path.empty();
    if (!yaml_expect_string_v1(doc, "model.save_file", yaml_cfg.save_path, false)) return false;
    if (!yaml_expect_string_v1(doc, "corpus.path", yaml_cfg.corpus_path, true)) return false;
    if (!yaml_expect_string_v1(doc, "trie.path", yaml_cfg.trie_dir, true)) return false;

    // ---- Budget ----
    std::string budget_unit;
    if (!yaml_expect_string_v1(doc, "train.budget.unit", budget_unit, true)) return false;
    const YamlScalarV1* budget_scalar = yaml_find_v1(doc, "train.budget.value");
    if (!budget_scalar) {
        std::fprintf(stderr, "agpt_train: YAML field train.budget.value is required\n");
        return false;
    }
    int budget_value = 0;
    if (!yaml_parse_int_v1("train.budget.value", *budget_scalar, budget_value)) return false;
    if (budget_unit == "epochs") {
        yaml_cfg.epochs = budget_value;
    } else {
        std::fprintf(stderr,
                     "agpt_train: YAML train.budget.unit=%s is not supported yet; v1 YAML supports epochs only\n",
                     budget_unit.c_str());
        return false;
    }

    // ---- Seed + quiet ----
    if (!yaml_get_int_v1(doc, "train.seed", yaml_cfg.seed, &yaml_cfg.seed_set)) return false;
    if (yaml_cfg.seed_set) yaml_cfg.shuffle_seed = (unsigned)yaml_cfg.seed;
    // train.quiet: v1 has no quiet flag yet — accept the field, no effect.
    bool quiet_dummy = false;
    if (!yaml_get_bool_v1(doc, "train.quiet", quiet_dummy)) return false;

    // ---- Optimizer ----
    const YamlScalarV1* optimizer_name = yaml_find_v1(doc, "train.optimizer.name");
    if (!optimizer_name) {
        std::fprintf(stderr, "agpt_train: YAML field train.optimizer.name is required\n");
        return false;
    }
    if (!parse_optimizer_kind_v1(optimizer_name->value.c_str(), yaml_cfg.optimizer)) {
        std::fprintf(stderr, "agpt_train: unsupported YAML train.optimizer.name: %s\n",
                     optimizer_name->value.c_str());
        return false;
    }
    const YamlScalarV1* lr = yaml_find_v1(doc, "train.optimizer.lr");
    if (!lr) {
        std::fprintf(stderr, "agpt_train: YAML field train.optimizer.lr is required\n");
        return false;
    }
    if (!yaml_parse_float_v1("train.optimizer.lr", *lr, yaml_cfg.lr)) return false;
    if (!yaml_get_float_v1(doc, "train.optimizer.beta", yaml_cfg.rmsprop_beta)) return false;
    if (!yaml_get_float_v1(doc, "train.optimizer.momentum_beta", yaml_cfg.momentum_beta)) return false;
    if (!yaml_get_float_v1(doc, "train.optimizer.weight_decay", yaml_cfg.weight_decay)) return false;
    if (!yaml_get_float_v1(doc, "train.optimizer.grad_clip_norm", yaml_cfg.grad_clip_norm)) return false;

    // ---- LR schedule ----
    const YamlScalarV1* lr_sched = yaml_find_v1(doc, "train.lr_schedule.name");
    if (lr_sched) {
        if (!parse_lr_schedule_v1(lr_sched->value.c_str(), yaml_cfg.lr_schedule)) {
            std::fprintf(stderr, "agpt_train: unsupported YAML train.lr_schedule.name: %s\n",
                         lr_sched->value.c_str());
            return false;
        }
    }
    if (!yaml_get_int_v1(doc, "train.lr_schedule.warmup_epochs", yaml_cfg.warmup_epochs)) return false;

    // ---- AGPT knobs ----
    if (!yaml_get_int_v1(doc, "train.partition_depth", yaml_cfg.partition_depth)) return false;
    if (!yaml_get_int_v1(doc, "train.chunk_queries", yaml_cfg.chunk_queries)) return false;
    if (!yaml_get_float_v1(doc, "train.entropy_lambda", yaml_cfg.entropy_lambda)) return false;
    if (!yaml_get_bool_v1(doc, "train.ce_only", yaml_cfg.ce_only)) return false;

    // anc_grad: schema bool → v1's two CLI bools.
    const YamlScalarV1* anc = yaml_find_v1(doc, "train.anc_grad");
    if (anc) {
        bool anc_val = false;
        if (!yaml_parse_bool_v1("train.anc_grad", *anc, anc_val)) return false;
        if (anc_val) {
            yaml_cfg.explicit_anc_grad = true;
        } else {
            yaml_cfg.ablate_anc_grad = true;
        }
    }

    // mass_weight: enum string.
    const YamlScalarV1* mw = yaml_find_v1(doc, "train.mass_weight");
    if (mw) {
        if (!parse_mass_weight_v1(mw->value.c_str(), yaml_cfg.mass_weight)) {
            std::fprintf(stderr, "agpt_train: unsupported YAML train.mass_weight: %s\n",
                         mw->value.c_str());
            return false;
        }
    }

    // fire_norm: enum string → multi-bool.
    const YamlScalarV1* fn = yaml_find_v1(doc, "train.fire_norm");
    if (fn) {
        if (!parse_fire_norm_v1(fn->value.c_str(),
                                yaml_cfg.fire_norm_by_mass,
                                yaml_cfg.fire_norm_by_weight,
                                yaml_cfg.fire_norm_none)) {
            std::fprintf(stderr, "agpt_train: unsupported YAML train.fire_norm: %s\n",
                         fn->value.c_str());
            return false;
        }
    }

    // ---- Context window — cross-checks ----
    if (!yaml_get_int_v1(doc, "train.max_depth", yaml_cfg.max_depth, &yaml_cfg.has_max_depth)) return false;
    if (!yaml_get_int_v1(doc, "train.seq_len", yaml_cfg.seq_len, &yaml_cfg.has_seq_len)) return false;
    const YamlScalarV1* trie_max_depth = yaml_find_v1(doc, "trie.max_depth");
    if (trie_max_depth) {
        int trie_depth = 0;
        if (!yaml_parse_int_v1("trie.max_depth", *trie_max_depth, trie_depth)) return false;
        if (yaml_cfg.has_max_depth && trie_depth != yaml_cfg.max_depth) {
            std::fprintf(stderr,
                         "agpt_train: trie.max_depth (%d) must match train.max_depth (%d)\n",
                         trie_depth, yaml_cfg.max_depth);
            return false;
        }
        if (!yaml_cfg.has_max_depth) {
            yaml_cfg.max_depth = trie_depth;
            yaml_cfg.has_max_depth = true;
        }
    }
    if (yaml_cfg.has_seq_len && yaml_cfg.has_max_depth && yaml_cfg.seq_len != yaml_cfg.max_depth) {
        std::fprintf(stderr,
                     "agpt_train: train.seq_len (%d) must match train.max_depth (%d)\n",
                     yaml_cfg.seq_len, yaml_cfg.max_depth);
        return false;
    }
    // v1 derives seq_len from trie meta — max_depth here is just a cross-check, not required.

    // ---- Model arch (init mode + cross-check with header otherwise) ----
    if (!yaml_get_int_v1(doc, "model.d_model", yaml_cfg.model_d_model, &yaml_cfg.has_model_d_model)) return false;
    if (!yaml_get_int_v1(doc, "model.n_layers", yaml_cfg.model_n_layers, &yaml_cfg.has_model_n_layers)) return false;
    if (!yaml_get_int_v1(doc, "model.n_heads", yaml_cfg.model_n_heads, &yaml_cfg.has_model_n_heads)) return false;
    if (!yaml_get_int_v1(doc, "model.d_ff", yaml_cfg.model_d_ff, &yaml_cfg.has_model_d_ff)) return false;
    if (!yaml_get_int_v1(doc, "model.head_dim", yaml_cfg.model_head_dim, &yaml_cfg.has_model_head_dim)) return false;
    if (yaml_cfg.init_mode) {
        if (yaml_cfg.has_model_d_model)  yaml_cfg.init_d_model  = yaml_cfg.model_d_model;
        if (yaml_cfg.has_model_n_layers) yaml_cfg.init_n_layers = yaml_cfg.model_n_layers;
        if (yaml_cfg.has_model_n_heads)  yaml_cfg.init_n_heads  = yaml_cfg.model_n_heads;
        if (yaml_cfg.has_model_d_ff)     yaml_cfg.init_d_ff     = yaml_cfg.model_d_ff;
    }

    // model.init_seed
    int init_seed_yaml = 0;
    bool has_init_seed = false;
    if (!yaml_get_int_v1(doc, "model.init_seed", init_seed_yaml, &has_init_seed)) return false;
    if (has_init_seed) yaml_cfg.init_seed = (uint32_t)init_seed_yaml;
    else if (yaml_cfg.seed_set) yaml_cfg.init_seed = (uint32_t)yaml_cfg.seed;

    // ---- Trie-level fields (mostly orchestrator hints; v1 reads from trie meta) ----
    bool virtual_tree = false;
    if (!yaml_get_bool_v1(doc, "trie.virtual_tree", virtual_tree)) return false;
    if (virtual_tree) {
        std::fprintf(stderr,
                     "agpt_train: trie.virtual_tree=true is not honored via YAML in v1; "
                     "use the --virtual-tree <path> CLI flag instead.\n");
        return false;
    }
    int unused_int = 0;
    if (!yaml_get_int_v1(doc, "trie.prune_min_mass", unused_int)) return false;
    if (!yaml_get_int_v1(doc, "trie.prune_min_depth", unused_int)) return false;

    // ---- Validation ----
    if (yaml_cfg.epochs <= 0) {
        std::fprintf(stderr, "agpt_train: train.budget.value must be positive for epoch budgets\n");
        return false;
    }
    if (yaml_cfg.partition_depth < 0) {
        std::fprintf(stderr, "agpt_train: train.partition_depth must be >= 0\n");
        return false;
    }
    if (yaml_cfg.init_mode &&
        (!yaml_cfg.has_model_d_model || !yaml_cfg.has_model_n_layers || !yaml_cfg.has_model_n_heads)) {
        std::fprintf(stderr,
                     "agpt_train: model.init_file is absent (fresh init mode) but model.d_model / "
                     "model.n_layers / model.n_heads are required for fresh init.\n");
        return false;
    }
    return true;
}

#endif
