#ifndef AGPT_V2_YAML_CONFIG_CUH
#define AGPT_V2_YAML_CONFIG_CUH

struct YamlScalarV2 {
    std::string value;
    yam_mark mark{};
};

struct YamlDocV2 {
    std::unordered_map<std::string, YamlScalarV2> scalars;
    std::unordered_map<std::string, std::vector<YamlScalarV2>> sequences;
    std::unordered_set<std::string> blocks;
};

struct YamlConfigV2 {
    std::string model_path;
    std::string trie_dir;
    std::string corpus_path;
    std::string save_path;
    std::string position_data_dir;

    int seed = 42;
    bool seed_set = false;
    bool has_growth = false;
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
    std::vector<int> checkpoint_epochs;
};

static std::string yam_str_to_std_v2(yam_str s) {
    if (!s.data || s.len == 0) return std::string();
    return std::string(s.data, s.len);
}

static std::string yaml_path_join_v2(const std::string& prefix, const std::string& key) {
    if (prefix.empty()) return key;
    return prefix + "." + key;
}

static bool parse_yaml_node_v2(const std::vector<yam_event>& events,
                               size_t& idx,
                               const std::string& path,
                               YamlDocV2& doc,
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
        doc.scalars[path] = YamlScalarV2{yam_str_to_std_v2(evt.value), evt.start};
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
            std::string key = yam_str_to_std_v2(key_evt.value);
            if (key.empty()) {
                error = "YAML mapping keys must not be empty";
                return false;
            }
            if (key.find('.') != std::string::npos) {
                error = "YAML mapping keys must use nested maps, not dotted keys: " + key;
                return false;
            }
            if (!parse_yaml_node_v2(events, idx, yaml_path_join_v2(path, key), doc, error)) {
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
        if (path.empty()) {
            error = "top-level YAML document must be a mapping";
            return false;
        }
        std::vector<YamlScalarV2> values;
        while (idx < events.size() && events[idx].type != YAM_EVT_SEQUENCE_END) {
            const yam_event& item_evt = events[idx++];
            if (item_evt.type != YAM_EVT_SCALAR) {
                error = "YAML sequence items must be scalars at " + path;
                return false;
            }
            values.push_back(YamlScalarV2{yam_str_to_std_v2(item_evt.value), item_evt.start});
        }
        if (idx >= events.size() || events[idx].type != YAM_EVT_SEQUENCE_END) {
            error = "unterminated YAML sequence at " + path;
            return false;
        }
        idx++;
        doc.sequences[path] = std::move(values);
        return true;
    }
    if (evt.type == YAM_EVT_ALIAS) {
        error = "unresolved YAML alias at " + path;
        return false;
    }
    error = "unexpected YAML event while reading " + path + ": " + yam_event_type_str(evt.type);
    return false;
}

static bool load_yaml_doc_v2(const char* path, YamlDocV2& doc) {
    yam_arena* arena = yam_arena_new(65536);
    if (!arena) {
        std::fprintf(stderr, "agpt_train_v2: failed to allocate YAML arena\n");
        return false;
    }
    yam_str data = yam_read_file(path, arena);
    if (!data.data) {
        std::fprintf(stderr, "agpt_train_v2: failed to read YAML config: %s\n", path);
        yam_arena_free(arena);
        return false;
    }
    yam_parser* parser = yam_parser_new(data.data, data.len, arena);
    if (!parser) {
        std::fprintf(stderr, "agpt_train_v2: failed to allocate YAML parser\n");
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
            std::fprintf(stderr, "agpt_train_v2: YAML parse error at %zu:%zu: %s\n",
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
    if (!parse_yaml_node_v2(events, idx, "", doc, error)) {
        std::fprintf(stderr, "agpt_train_v2: YAML config error: %s\n", error.c_str());
        yam_arena_free(arena);
        return false;
    }
    if (idx < events.size() && events[idx].type == YAM_EVT_DOC_END) idx++;
    if (idx >= events.size() || events[idx].type != YAM_EVT_STREAM_END) {
        std::fprintf(stderr, "agpt_train_v2: YAML config must contain exactly one document\n");
        yam_arena_free(arena);
        return false;
    }

    yam_arena_free(arena);
    return true;
}

static bool yaml_is_ignored_field_v2(const std::string& path) {
    if (path == "description" || path == "experiment" || path == "run_slug") return true;
    if (path == "corpus.heldout") return true;
    if (path.rfind("corpus.carve.", 0) == 0) return true;
    if (path.rfind("eval.", 0) == 0) return true;
    return false;
}

static bool yaml_is_consumed_field_v2(const std::string& path) {
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
        "train.optimizer.eps",
        "train.optimizer.weight_decay",
        "train.optimizer.grad_clip_norm",
        "train.lr_schedule.name",
        "train.lr_schedule.warmup_epochs",
        "train.seq_len",
        "train.max_depth",
        "train.partition_depth",
        "train.chunk_queries",
        "train.checkpoint_epochs",
        "train.anc_grad",
        "train.mass_weight",
        "train.fire_norm",
        "train.entropy_lambda",
        "train.ce_only",
        "train.growth.divisions",
        "train.growth.min_epochs",
        "train.growth.epoch_ramp",
    };
    return fields.find(path) != fields.end();
}

static bool yaml_is_known_non_v2_field_v2(const std::string& path) {
    static const std::unordered_set<std::string> fields = {
        "train.backend",
        "train.heads",
        "train.lookahead",
    };
    return fields.find(path) != fields.end();
}

static bool yaml_is_experimental_field_v2(const std::string& path) {
    return path.rfind("experimental.", 0) == 0 && path.size() > std::strlen("experimental.");
}

static bool yaml_is_known_experimental_field_v2(const std::string& path) {
    static const std::unordered_set<std::string> fields = {
        "experimental.rope_position_mode",
        "experimental.rope_position_offset",
        "experimental.phase_order",
        "experimental.phase_order_seed",
        "experimental.position_data_dir",
        "experimental.pos_sample_seed",
    };
    return fields.find(path) != fields.end();
}

static bool warn_unknown_experimental_flags_v2(const YamlDocV2& doc) {
    std::vector<std::string> flags;
    for (const auto& item : doc.scalars) {
        const std::string& path = item.first;
        if (!yaml_is_experimental_field_v2(path)) continue;
        if (yaml_is_known_experimental_field_v2(path)) continue;
        flags.push_back(path.substr(std::strlen("experimental.")));
    }
    for (const auto& item : doc.sequences) {
        const std::string& path = item.first;
        if (!yaml_is_experimental_field_v2(path)) continue;
        if (yaml_is_known_experimental_field_v2(path)) continue;
        flags.push_back(path.substr(std::strlen("experimental.")));
    }
    std::sort(flags.begin(), flags.end());
    for (const std::string& flag : flags) {
        std::fprintf(stderr, "WARN: unknown experimental flag %s; ignoring\n", flag.c_str());
    }
    return true;
}

static bool validate_yaml_registry_v2(const YamlDocV2& doc) {
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
            std::fprintf(stderr, "agpt_train_v2: unknown YAML block: %s\n", block.c_str());
            return false;
        }
    }
    for (const auto& item : doc.scalars) {
        const std::string& path = item.first;
        if (yaml_is_experimental_field_v2(path)) continue;
        if (yaml_is_ignored_field_v2(path) || yaml_is_consumed_field_v2(path)) continue;
        if (yaml_is_known_non_v2_field_v2(path)) {
            std::fprintf(stderr, "agpt_train_v2: YAML field is not supported by v2: %s\n", path.c_str());
        } else {
            std::fprintf(stderr, "agpt_train_v2: unknown YAML field: %s\n", path.c_str());
        }
        return false;
    }
    for (const auto& item : doc.sequences) {
        const std::string& path = item.first;
        if (yaml_is_experimental_field_v2(path)) continue;
        if (yaml_is_ignored_field_v2(path) || yaml_is_consumed_field_v2(path)) continue;
        if (yaml_is_known_non_v2_field_v2(path)) {
            std::fprintf(stderr, "agpt_train_v2: YAML field is not supported by v2: %s\n", path.c_str());
        } else {
            std::fprintf(stderr, "agpt_train_v2: unknown YAML field: %s\n", path.c_str());
        }
        return false;
    }
    return true;
}

static const YamlScalarV2* yaml_find_v2(const YamlDocV2& doc, const char* path) {
    auto it = doc.scalars.find(path);
    return it == doc.scalars.end() ? nullptr : &it->second;
}

static bool yaml_parse_long_v2(const char* field, const YamlScalarV2& scalar, long& out) {
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(scalar.value.c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0') {
        std::fprintf(stderr, "agpt_train_v2: YAML field %s must be an integer (got %s)\n",
                     field, scalar.value.c_str());
        return false;
    }
    out = value;
    return true;
}

static bool yaml_parse_int_v2(const char* field, const YamlScalarV2& scalar, int& out) {
    long value = 0;
    if (!yaml_parse_long_v2(field, scalar, value)) return false;
    if (value < INT_MIN || value > INT_MAX) {
        std::fprintf(stderr, "agpt_train_v2: YAML field %s is out of int range: %s\n",
                     field, scalar.value.c_str());
        return false;
    }
    out = (int)value;
    return true;
}

static bool yaml_parse_float_v2(const char* field, const YamlScalarV2& scalar, float& out) {
    errno = 0;
    char* end = nullptr;
    float value = std::strtof(scalar.value.c_str(), &end);
    if (errno != 0 || !end || *end != '\0') {
        std::fprintf(stderr, "agpt_train_v2: YAML field %s must be a number (got %s)\n",
                     field, scalar.value.c_str());
        return false;
    }
    out = value;
    return true;
}

static bool yaml_parse_bool_v2(const char* field, const YamlScalarV2& scalar, bool& out) {
    if (scalar.value == "true" || scalar.value == "True" || scalar.value == "TRUE") {
        out = true;
        return true;
    }
    if (scalar.value == "false" || scalar.value == "False" || scalar.value == "FALSE") {
        out = false;
        return true;
    }
    std::fprintf(stderr, "agpt_train_v2: YAML field %s must be true or false (got %s)\n",
                 field, scalar.value.c_str());
    return false;
}

static bool parse_mass_weight_v2(const char* text, agpt_v2::MassWeightModeV2& out) {
    if (std::strcmp(text, "off") == 0) {
        out = agpt_v2::MassWeightModeV2::Off;
        return true;
    }
    if (std::strcmp(text, "linear") == 0) {
        out = agpt_v2::MassWeightModeV2::Linear;
        return true;
    }
    if (std::strcmp(text, "sqrt") == 0) {
        out = agpt_v2::MassWeightModeV2::Sqrt;
        return true;
    }
    if (std::strcmp(text, "log") == 0) {
        out = agpt_v2::MassWeightModeV2::Log;
        return true;
    }
    if (std::strcmp(text, "inv-log") == 0) {
        out = agpt_v2::MassWeightModeV2::InvLog;
        return true;
    }
    if (std::strcmp(text, "inv-linear") == 0) {
        out = agpt_v2::MassWeightModeV2::InvLinear;
        return true;
    }
    return false;
}

static bool yaml_get_int_v2(const YamlDocV2& doc, const char* field, int& out, bool* present = nullptr) {
    const YamlScalarV2* scalar = yaml_find_v2(doc, field);
    if (present) *present = scalar != nullptr;
    if (!scalar) return true;
    return yaml_parse_int_v2(field, *scalar, out);
}

static bool yaml_get_float_v2(const YamlDocV2& doc, const char* field, float& out, bool* present = nullptr) {
    const YamlScalarV2* scalar = yaml_find_v2(doc, field);
    if (present) *present = scalar != nullptr;
    if (!scalar) return true;
    return yaml_parse_float_v2(field, *scalar, out);
}

static bool yaml_get_bool_v2(const YamlDocV2& doc, const char* field, bool& out, bool* present = nullptr) {
    const YamlScalarV2* scalar = yaml_find_v2(doc, field);
    if (present) *present = scalar != nullptr;
    if (!scalar) return true;
    return yaml_parse_bool_v2(field, *scalar, out);
}

static bool yaml_get_int_sequence_v2(const YamlDocV2& doc, const char* field, std::vector<int>& out) {
    auto it = doc.sequences.find(field);
    if (it == doc.sequences.end()) return true;
    out.clear();
    out.reserve(it->second.size());
    for (const YamlScalarV2& scalar : it->second) {
        int value = 0;
        if (!yaml_parse_int_v2(field, scalar, value)) return false;
        out.push_back(value);
    }
    return true;
}

static bool yaml_expect_string_v2(const YamlDocV2& doc, const char* field, std::string& out, bool required) {
    const YamlScalarV2* scalar = yaml_find_v2(doc, field);
    if (!scalar) {
        if (required) {
            std::fprintf(stderr, "agpt_train_v2: YAML field %s is required\n", field);
            return false;
        }
        return true;
    }
    out = scalar->value;
    return true;
}

static bool yaml_reject_non_default_str_v2(const YamlDocV2& doc,
                                           const char* field,
                                           const char* default_value,
                                           const char* reason) {
    const YamlScalarV2* scalar = yaml_find_v2(doc, field);
    if (!scalar || scalar->value == default_value) return true;
    std::fprintf(stderr,
                 "agpt_train_v2: YAML field %s=%s is not currently honored by v2 (%s); only %s is accepted for now\n",
                 field, scalar->value.c_str(), reason, default_value);
    return false;
}

static bool yaml_reject_non_default_float_v2(const YamlDocV2& doc,
                                             const char* field,
                                             float default_value,
                                             const char* reason) {
    const YamlScalarV2* scalar = yaml_find_v2(doc, field);
    if (!scalar) return true;
    float value = 0.0f;
    if (!yaml_parse_float_v2(field, *scalar, value)) return false;
    if (value == default_value) return true;
    std::fprintf(stderr,
                 "agpt_train_v2: YAML field %s=%s is not currently honored by v2 (%s); only %.6g is accepted for now\n",
                 field, scalar->value.c_str(), reason, default_value);
    return false;
}

static bool yaml_reject_non_default_bool_v2(const YamlDocV2& doc,
                                            const char* field,
                                            bool default_value,
                                            const char* reason) {
    const YamlScalarV2* scalar = yaml_find_v2(doc, field);
    if (!scalar) return true;
    bool value = false;
    if (!yaml_parse_bool_v2(field, *scalar, value)) return false;
    if (value == default_value) return true;
    std::fprintf(stderr,
                 "agpt_train_v2: YAML field %s=%s is not currently honored by v2 (%s); only %s is accepted for now\n",
                 field, scalar->value.c_str(), reason, default_value ? "true" : "false");
    return false;
}

static bool apply_yaml_config_v2(const char* config_path,
                                 agpt_v2::TrainerConfig& cfg,
                                 YamlConfigV2& yaml_cfg,
                                 V2Mode& mode,
                                 int& steps,
                                 int& unit_limit,
                                 int& growth_max_depth,
                                 int& growth_min_epochs,
                                 int& growth_divisions,
                                 int& growth_final_frontier,
                                 double& growth_train_frac,
                                 GrowthEpochScheduleV2& growth_epoch_schedule,
                                 bool& explicit_anc_grad,
                                 bool& ablate_anc_grad) {
    (void)steps;
    (void)growth_final_frontier;
    (void)growth_train_frac;
    (void)explicit_anc_grad;
    (void)ablate_anc_grad;

    YamlDocV2 doc;
    if (!load_yaml_doc_v2(config_path, doc)) return false;
    if (!validate_yaml_registry_v2(doc)) return false;
    if (!warn_unknown_experimental_flags_v2(doc)) return false;

    yaml_cfg.has_growth = doc.blocks.find("train.growth") != doc.blocks.end() ||
        yaml_find_v2(doc, "train.growth.divisions") ||
        yaml_find_v2(doc, "train.growth.min_epochs") ||
        yaml_find_v2(doc, "train.growth.epoch_ramp");
    mode = yaml_cfg.has_growth ? V2Mode::TrainGrowth : V2Mode::TrainEpoch;
    cfg.accumulate = true;

    if (!yaml_expect_string_v2(doc, "model.init_file", yaml_cfg.model_path, true)) return false;
    if (!yaml_expect_string_v2(doc, "model.save_file", yaml_cfg.save_path, false)) return false;
    if (!yaml_expect_string_v2(doc, "corpus.path", yaml_cfg.corpus_path, true)) return false;
    if (!yaml_cfg.has_growth) {
        if (!yaml_expect_string_v2(doc, "trie.path", yaml_cfg.trie_dir, true)) return false;
    } else {
        yaml_expect_string_v2(doc, "trie.path", yaml_cfg.trie_dir, false);
    }

    std::string budget_unit;
    if (!yaml_expect_string_v2(doc, "train.budget.unit", budget_unit, true)) return false;
    int budget_value = 0;
    const YamlScalarV2* budget_scalar = yaml_find_v2(doc, "train.budget.value");
    if (!budget_scalar) {
        std::fprintf(stderr, "agpt_train_v2: YAML field train.budget.value is required\n");
        return false;
    }
    if (!yaml_parse_int_v2("train.budget.value", *budget_scalar, budget_value)) return false;
    if (budget_unit == "epochs") {
        cfg.epochs = budget_value;
    } else {
        std::fprintf(stderr,
                     "agpt_train_v2: YAML train.budget.unit=%s is not supported yet; v2 YAML currently supports epochs\n",
                     budget_unit.c_str());
        return false;
    }

    if (!yaml_get_int_v2(doc, "train.seed", yaml_cfg.seed, &yaml_cfg.seed_set)) return false;
    if (yaml_cfg.seed_set) cfg.pos_sample_seed = (unsigned)yaml_cfg.seed;
    if (!yaml_get_bool_v2(doc, "train.quiet", cfg.quiet)) return false;

    const YamlScalarV2* rope_position_mode = yaml_find_v2(doc, "experimental.rope_position_mode");
    if (rope_position_mode && !parse_rope_position_mode_v2(rope_position_mode->value.c_str(), cfg.rope_position_mode)) {
        std::fprintf(stderr, "agpt_train_v2: unsupported YAML experimental.rope_position_mode: %s\n",
                     rope_position_mode->value.c_str());
        return false;
    }
    bool has_rope_position_offset = false;
    if (!yaml_get_int_v2(doc, "experimental.rope_position_offset", cfg.rope_position_offset,
                         &has_rope_position_offset)) return false;
    if (has_rope_position_offset && cfg.rope_position_offset < 0) {
        std::fprintf(stderr, "agpt_train_v2: YAML experimental.rope_position_offset must be non-negative\n");
        return false;
    }
    const YamlScalarV2* phase_order = yaml_find_v2(doc, "experimental.phase_order");
    if (phase_order) {
        if (phase_order->value == "sequential" || phase_order->value == "sweep") {
            cfg.rope_phase_shuffle = false;
        } else if (phase_order->value == "shuffle" || phase_order->value == "shuffled") {
            cfg.rope_phase_shuffle = true;
        } else {
            std::fprintf(stderr, "agpt_train_v2: unsupported YAML experimental.phase_order: %s\n",
                         phase_order->value.c_str());
            return false;
        }
    }
    int phase_order_seed = 0;
    bool has_phase_order_seed = false;
    if (!yaml_get_int_v2(doc, "experimental.phase_order_seed", phase_order_seed, &has_phase_order_seed)) return false;
    if (has_phase_order_seed) {
        if (phase_order_seed < 0) {
            std::fprintf(stderr, "agpt_train_v2: YAML experimental.phase_order_seed must be non-negative\n");
            return false;
        }
        cfg.rope_phase_shuffle_seed = (unsigned)phase_order_seed;
    }
    if (!yaml_expect_string_v2(doc, "experimental.position_data_dir", yaml_cfg.position_data_dir, false)) return false;
    int pos_sample_seed = 0;
    bool has_pos_sample_seed = false;
    if (!yaml_get_int_v2(doc, "experimental.pos_sample_seed", pos_sample_seed, &has_pos_sample_seed)) return false;
    if (has_pos_sample_seed) {
        if (pos_sample_seed < 0) {
            std::fprintf(stderr, "agpt_train_v2: YAML experimental.pos_sample_seed must be non-negative\n");
            return false;
        }
        cfg.pos_sample_seed = (unsigned)pos_sample_seed;
    }

    const YamlScalarV2* optimizer_name = yaml_find_v2(doc, "train.optimizer.name");
    if (!optimizer_name) {
        std::fprintf(stderr, "agpt_train_v2: YAML field train.optimizer.name is required\n");
        return false;
    }
    if (!parse_optimizer_kind(optimizer_name->value.c_str(), cfg.optimizer)) {
        std::fprintf(stderr, "agpt_train_v2: unsupported YAML train.optimizer.name: %s\n",
                     optimizer_name->value.c_str());
        return false;
    }
    const YamlScalarV2* lr = yaml_find_v2(doc, "train.optimizer.lr");
    if (!lr) {
        std::fprintf(stderr, "agpt_train_v2: YAML field train.optimizer.lr is required\n");
        return false;
    }
    if (!yaml_parse_float_v2("train.optimizer.lr", *lr, cfg.lr)) return false;
    if (!yaml_get_float_v2(doc, "train.optimizer.beta", cfg.rmsprop_beta)) return false;
    if (!yaml_get_float_v2(doc, "train.optimizer.momentum_beta", cfg.momentum_beta)) return false;
    if (!yaml_get_float_v2(doc, "train.optimizer.eps", cfg.optimizer_eps)) return false;
    if (!yaml_get_float_v2(doc, "train.optimizer.weight_decay", cfg.weight_decay)) return false;
    if (!yaml_get_float_v2(doc, "train.optimizer.grad_clip_norm", cfg.grad_clip_norm)) return false;
    if (!yaml_reject_non_default_float_v2(doc, "train.optimizer.grad_clip_norm", 0.0f,
                                          "gradient clipping is not wired in CUDAX yet")) return false;

    const YamlScalarV2* lr_schedule = yaml_find_v2(doc, "train.lr_schedule.name");
    if (lr_schedule) {
        if (!parse_lr_schedule(lr_schedule->value.c_str(), cfg.lr_schedule)) {
            std::fprintf(stderr, "agpt_train_v2: unsupported YAML train.lr_schedule.name: %s\n",
                         lr_schedule->value.c_str());
            return false;
        }
    }
    if (!yaml_get_int_v2(doc, "train.lr_schedule.warmup_epochs", cfg.warmup_epochs)) return false;

    if (!yaml_get_int_v2(doc, "train.partition_depth", cfg.partition_depth)) return false;
    if (!yaml_get_int_v2(doc, "train.chunk_queries", cfg.chunk_queries)) return false;
    if (!yaml_get_int_sequence_v2(doc, "train.checkpoint_epochs", yaml_cfg.checkpoint_epochs)) return false;
    if (!yaml_get_bool_v2(doc, "train.anc_grad", cfg.anc_grad)) return false;
    const YamlScalarV2* mass_weight = yaml_find_v2(doc, "train.mass_weight");
    if (mass_weight && !parse_mass_weight_v2(mass_weight->value.c_str(), cfg.mass_weight)) {
        std::fprintf(stderr, "agpt_train_v2: unsupported YAML train.mass_weight: %s\n",
                     mass_weight->value.c_str());
        return false;
    }

    if (!yaml_get_int_v2(doc, "train.max_depth", yaml_cfg.max_depth, &yaml_cfg.has_max_depth)) return false;
    if (!yaml_get_int_v2(doc, "train.seq_len", yaml_cfg.seq_len, &yaml_cfg.has_seq_len)) return false;
    if (yaml_cfg.has_max_depth) growth_max_depth = yaml_cfg.max_depth;
    const YamlScalarV2* trie_max_depth = yaml_find_v2(doc, "trie.max_depth");
    if (trie_max_depth) {
        int trie_depth = 0;
        if (!yaml_parse_int_v2("trie.max_depth", *trie_max_depth, trie_depth)) return false;
        if (yaml_cfg.has_max_depth && trie_depth != yaml_cfg.max_depth) {
            std::fprintf(stderr,
                         "agpt_train_v2: trie.max_depth (%d) must match train.max_depth (%d)\n",
                         trie_depth, yaml_cfg.max_depth);
            return false;
        }
        if (!yaml_cfg.has_max_depth) {
            yaml_cfg.max_depth = trie_depth;
            yaml_cfg.has_max_depth = true;
            growth_max_depth = trie_depth;
        }
    }
    if (yaml_cfg.has_seq_len && yaml_cfg.has_max_depth && yaml_cfg.seq_len != yaml_cfg.max_depth) {
        std::fprintf(stderr,
                     "agpt_train_v2: train.seq_len (%d) must match train.max_depth (%d)\n",
                     yaml_cfg.seq_len, yaml_cfg.max_depth);
        return false;
    }
    if (!yaml_cfg.has_max_depth) {
        std::fprintf(stderr, "agpt_train_v2: YAML field train.max_depth is required for v2 configs\n");
        return false;
    }

    if (!yaml_get_int_v2(doc, "model.d_model", yaml_cfg.model_d_model, &yaml_cfg.has_model_d_model)) return false;
    if (!yaml_get_int_v2(doc, "model.n_layers", yaml_cfg.model_n_layers, &yaml_cfg.has_model_n_layers)) return false;
    if (!yaml_get_int_v2(doc, "model.n_heads", yaml_cfg.model_n_heads, &yaml_cfg.has_model_n_heads)) return false;
    if (!yaml_get_int_v2(doc, "model.d_ff", yaml_cfg.model_d_ff, &yaml_cfg.has_model_d_ff)) return false;
    if (!yaml_get_int_v2(doc, "model.head_dim", yaml_cfg.model_head_dim, &yaml_cfg.has_model_head_dim)) return false;

    if (yaml_cfg.has_growth) {
        bool have_divisions = false;
        bool have_min_epochs = false;
        if (!yaml_get_int_v2(doc, "train.growth.divisions", growth_divisions, &have_divisions)) return false;
        if (!yaml_get_int_v2(doc, "train.growth.min_epochs", growth_min_epochs, &have_min_epochs)) return false;
        if (!have_divisions || !have_min_epochs) {
            std::fprintf(stderr,
                         "agpt_train_v2: train.growth.divisions and train.growth.min_epochs are required when train.growth is present\n");
            return false;
        }
        const YamlScalarV2* ramp = yaml_find_v2(doc, "train.growth.epoch_ramp");
        if (!ramp) {
            std::fprintf(stderr, "agpt_train_v2: YAML field train.growth.epoch_ramp is required when train.growth is present\n");
            return false;
        }
        if (!parse_growth_epoch_schedule_v2(ramp->value.c_str(), growth_epoch_schedule)) {
            std::fprintf(stderr, "agpt_train_v2: unsupported YAML train.growth.epoch_ramp: %s\n",
                         ramp->value.c_str());
            return false;
        }
    }

    bool virtual_tree = false;
    if (!yaml_get_bool_v2(doc, "trie.virtual_tree", virtual_tree)) return false;
    if (virtual_tree) {
        std::fprintf(stderr, "agpt_train_v2: trie.virtual_tree=true is not currently wired in CUDAX\n");
        return false;
    }
    int unused_int = 0;
    if (!yaml_get_int_v2(doc, "trie.prune_min_mass", unused_int)) return false;
    if (!yaml_get_int_v2(doc, "trie.prune_min_depth", unused_int)) return false;
    if (!yaml_reject_non_default_str_v2(doc, "train.fire_norm", "mass",
                                        "CUDAX currently normalizes by weighted event mass")) return false;
    if (!yaml_reject_non_default_float_v2(doc, "train.entropy_lambda", 0.0f,
                                          "entropy regularization is not wired in CUDAX yet")) return false;
    if (!yaml_reject_non_default_bool_v2(doc, "train.ce_only", false,
                                         "CE-only endpoint mode is not wired in CUDAX yet")) return false;

    if (cfg.epochs <= 0) {
        std::fprintf(stderr, "agpt_train_v2: train.budget.value must be positive for epoch budgets\n");
        return false;
    }
    if (yaml_cfg.has_growth && !yaml_cfg.checkpoint_epochs.empty()) {
        std::fprintf(stderr,
                     "agpt_train_v2: train.checkpoint_epochs is currently supported only for static v2 training (no train.growth block)\n");
        return false;
    }
    for (int epoch : yaml_cfg.checkpoint_epochs) {
        if (epoch <= 0 || epoch > cfg.epochs) {
            std::fprintf(stderr,
                         "agpt_train_v2: train.checkpoint_epochs values must be in [1, train.budget.value] (got %d, budget=%d)\n",
                         epoch, cfg.epochs);
            return false;
        }
    }
    if (growth_divisions < 0 || growth_min_epochs < 0 || unit_limit < 0) {
        std::fprintf(stderr, "agpt_train_v2: YAML numeric fields must be non-negative where applicable\n");
        return false;
    }
    return true;
}

#endif
