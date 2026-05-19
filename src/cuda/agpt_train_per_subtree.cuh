// ============================================================================
// agpt_train_per_subtree.cuh
// ============================================================================
//
// Per-subtree training path (format=2 manifest). For deep tries (d=32+)
// the global KV cache won't fit in RAM. The per-subtree radix format
// splits the trie into one file per root-child, each with a self-contained
// local index space. This function loads one subtree at a time, sizes the
// KV cache to just that subtree's character count, runs one Adam/RMSProp
// step on it, frees the KV cache, and moves on.
//
// Optimizer state (Adam m/v, RMSProp s, step counter) persists in host
// buffers across subtree calls so the running averages don't reset each
// subtree. Weights persist via the caller's h_weights buffer which
// run_radix_training updates in-place.
//
// Originally task #43. Calls run_radix_training (from agpt_train.cu) for
// each subtree; this file is included AFTER run_radix_training is defined.
// ============================================================================

// ============================================================================
// Per-subtree training path (task #43)
// ============================================================================
//
// For deep tries (d=32+) the global KV cache won't fit in RAM. The per-subtree
// radix format splits the trie into one file per root-child, each with a
// self-contained local index space. This function loads one subtree at a
// time, sizes the KV cache to just that subtree's character count, runs one
// Adam/RMSProp step on it, frees the KV cache, and moves on.
//
// Optimizer state (Adam m/v, RMSProp s, step counter) persists in host buffers
// across subtree calls so the running averages don't reset each subtree. Weights
// persist via the caller's h_weights buffer which run_radix_training updates
// in-place.
int run_per_subtree_training(const Config& cfg_in, const WeightOffsets& wo,
                              float* h_weights,
                              const SubtreeManifest& manifest,
                              int super_epochs, float entropy_lambda, MassWeightMode mass_weight,
                              int subtree_splits, int partition_depth, bool accumulate,
                              bool single_subtree, float intermediate_weight,
                              OptimizerKind optimizer, float momentum_beta, float rmsprop_beta,
                              LRSchedule lr_schedule, int warmup_super_epochs,
                              float weight_decay, float grad_clip_norm, int save_every,
                              CurriculumMode curriculum, const char* save_path,
                              bool lr_scale_by_steps = false,
                              LightningConfig lightning = LightningConfig{})
{
    // Auto-LR scaling: the optimal LR depends on total gradient-movement per pass
    // (lr × steps_per_super_epoch ≈ constant for a fixed depth). The winning d=16
    // unigram recipe was calibrated at 65 steps/super-epoch, lr=3e-3. When the
    // user changes subtree granularity (bigram: 1465 steps, trigram later) we
    // rescale lr so the same base_lr knob keeps working. Reference step count is
    // hardcoded at 65 because that's what the shipped recipe in memory was
    // calibrated against — don't change without a fresh calibration.
    Config cfg = cfg_in;
    const int LR_SCALE_REFERENCE_STEPS = 65;
    const bool lightning_active = (lightning.steps > 0);
    int steps_per_super_epoch = lightning_active
        ? lightning.steps
        : manifest.n_subtrees * subtree_splits;
    if (lr_scale_by_steps && steps_per_super_epoch > 0) {
        float scale = (float)LR_SCALE_REFERENCE_STEPS / (float)steps_per_super_epoch;
        float scaled_lr = cfg_in.lr * scale;
        printf("Per-subtree training: %d subtrees, %d super-epochs (lr auto-scaled %.4g → %.4g × %d/%d)\n",
               manifest.n_subtrees, super_epochs, cfg_in.lr, scaled_lr,
               LR_SCALE_REFERENCE_STEPS, steps_per_super_epoch);
        cfg.lr = scaled_lr;
    } else {
        printf("Per-subtree training: %d subtrees, %d super-epochs%s\n",
               manifest.n_subtrees, super_epochs,
               lightning_active ? " (Lightning stochastic sampling)" : "");
    }
    if (lightning_active) {
        const char* sname = (lightning.sampler == LightningSampler::L1_Uniform) ? "l1-uniform"
                          : (lightning.sampler == LightningSampler::L2_RcDepth) ? "l2-rc-depth"
                          : (lightning.sampler == LightningSampler::L4_Path)    ? "l4-path (SGD-equivalent)"
                          :                                                        "l3-mass-walk";
        printf("  lightning: %s, %d samples/SE total, p_stop=%.2f, seed=0x%x\n",
               sname, lightning.steps, lightning.p_stop, lightning.seed);
    }

    const char* opt_name = (optimizer == OptimizerKind::Adam)     ? "adam"
                         : (optimizer == OptimizerKind::SGD)      ? "sgd"
                         : (optimizer == OptimizerKind::Momentum) ? "momentum"
                         : (optimizer == OptimizerKind::LBFGS)    ? "lbfgs"
                         :                                          "rmsprop";
    printf("  optimizer: %s (lr=%.4g)\n", opt_name, cfg.lr);
    const char* sched_name = (lr_schedule == LRSchedule::Constant)    ? "constant"
                           : (lr_schedule == LRSchedule::Cosine)       ? "cosine"
                           :                                             "warmup-cosine";
    printf("  lr-schedule: %s (warmup %d super-epochs = %d steps)\n",
           sched_name, warmup_super_epochs, warmup_super_epochs * manifest.n_subtrees * subtree_splits);
    if (entropy_lambda > 0.0f) printf("  entropy lambda: %.3f\n", entropy_lambda);
    if (mass_weight != MassWeightMode::Off) {
        const char* mode_name = (mass_weight == MassWeightMode::Log)    ? "log"
                              : (mass_weight == MassWeightMode::Sqrt)   ? "sqrt"
                              : (mass_weight == MassWeightMode::Linear) ? "linear"
                              :                                            "?";
        printf("  mass weighting: %s\n", mode_name);
    }
    if (single_subtree) printf("  single-subtree (per file): 1 Adam step per subtree per super-epoch\n");

    // Allocate optimizer-state host buffers so state persists across the many
    // run_radix_training invocations below.
    float* h_adam_m = (float*)calloc(wo.total_floats, sizeof(float));
    float* h_adam_v = (float*)calloc(wo.total_floats, sizeof(float));
    int adam_t = 0;

    // Total optimizer steps across the whole training (for cosine horizon).
    int total_opt_steps = super_epochs * steps_per_super_epoch;
    int warmup_steps    = warmup_super_epochs * steps_per_super_epoch;

    printf("  total optimizer steps: %d (%d per super-epoch)\n",
           total_opt_steps, steps_per_super_epoch);

    // Largest subtree (by char count) paced first to surface OOM early.
    int largest_idx = 0;
    long long largest_chars = manifest.entries[0].total_edge_chars;
    for (int i = 1; i < manifest.n_subtrees; i++) {
        if (manifest.entries[i].total_edge_chars > largest_chars) {
            largest_chars = manifest.entries[i].total_edge_chars;
            largest_idx = i;
        }
    }
    printf("  largest subtree: rc=%d, %lld chars (peak per-subtree KV ≈ %.1f MB)\n",
           manifest.entries[largest_idx].root_child_id, largest_chars,
           largest_chars * cfg.d_model * 4.0 * 2 * cfg.n_layers / 1e6);

    // Lightning over per-subtree files: pre-sample root-child indices per SE,
    // weighted by each subtree's total_edge_chars (a proxy for corpus mass
    // flowing through that root-child; more principled would be to read each
    // subtree's edge_mass[0] but that requires a one-time scan). Each
    // root-child with ≥1 bucketed sample gets loaded once per SE and runs
    // that many Lightning samples within its local view.
    unsigned lightning_outer_rng = lightning.seed;
    double* lightning_rc_weights = NULL;
    double lightning_rc_total_weight = 0.0;
    if (lightning_active) {
        lightning_rc_weights = (double*)malloc(manifest.n_subtrees * sizeof(double));
        for (int i = 0; i < manifest.n_subtrees; i++) {
            lightning_rc_weights[i] = (double)manifest.entries[i].total_edge_chars;
            lightning_rc_total_weight += lightning_rc_weights[i];
        }
        if (lightning_rc_total_weight <= 0.0) lightning_rc_total_weight = 1.0;
    }

    for (int ep = 0; ep < super_epochs; ep++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        double super_loss_sum = 0.0;
        long long super_nodes_trained = 0;
        int subtrees_done = 0;

        // ---- Lightning path ----
        if (lightning_active) {
            // Bucket this SE's lightning.steps samples by root-child index,
            // weighted by total_edge_chars.
            int* bucket = (int*)calloc(manifest.n_subtrees, sizeof(int));
            for (int s = 0; s < lightning.steps; s++) {
                double u = (double)xorshift_float01(&lightning_outer_rng) * lightning_rc_total_weight;
                double acc = 0.0;
                int pick = manifest.n_subtrees - 1;
                for (int i = 0; i < manifest.n_subtrees; i++) {
                    acc += lightning_rc_weights[i];
                    if (u <= acc) { pick = i; break; }
                }
                bucket[pick]++;
            }
            int touched = 0, max_bucket = 0;
            for (int i = 0; i < manifest.n_subtrees; i++) {
                if (bucket[i] > 0) touched++;
                if (bucket[i] > max_bucket) max_bucket = bucket[i];
            }
            printf("  SE %d lightning buckets: %d root-children touched, max %d samples/rc\n",
                   ep + 1, touched, max_bucket);

            for (int i = 0; i < manifest.n_subtrees; i++) {
                if (bucket[i] == 0) continue;
                SubtreeData s = load_subtree(manifest, i);
                RadixView view = subtree_to_radix_view(s);

                TrainPersistence persist;
                persist.h_adam_m_io = h_adam_m;
                persist.h_adam_v_io = h_adam_v;
                persist.adam_t_io = &adam_t;
                persist.quiet = true;
                persist.total_opt_steps_override = total_opt_steps;
                persist.warmup_steps_override = warmup_steps;

                // Build a per-call Lightning config with steps scoped to this
                // root-child's bucket. Reuse the same sampler/p_stop/mass-lr
                // settings but derive the inner RNG seed from seed+ep+rc so
                // runs are reproducible and bucket sequences diverge across SEs.
                LightningConfig inner = lightning;
                inner.steps = bucket[i];
                inner.seed  = lightning.seed ^ (unsigned)(0x9E3779B9u * (ep + 1)) ^ (unsigned)(0x85EBCA6Bu * (i + 1));

                run_radix_training(cfg, wo, h_weights, view.t,
                                   /*epochs=*/1, entropy_lambda, mass_weight, subtree_splits, partition_depth, accumulate,
                                   /*single_subtree=*/true, intermediate_weight,
                                   optimizer, momentum_beta, rmsprop_beta,
                                   lr_schedule, warmup_super_epochs,
                                   weight_decay, grad_clip_norm, /*save_every=*/0,
                                   curriculum, /*save_path=*/NULL,
                                   inner, &persist);

                super_nodes_trained += s.n_nodes;
                subtrees_done++;

                free_radix_view(view);
                free_subtree(s);
            }
            free(bucket);
        } else {
        // ---- Deterministic per-root-child path (existing) ----
        for (int ii = 0; ii < manifest.n_subtrees; ii++) {
            int i = ii;
            if (ep == 0 && ii == 0) i = largest_idx;
            else if (ep == 0 && ii == largest_idx) i = 0;

            SubtreeData s = load_subtree(manifest, i);
            RadixView view = subtree_to_radix_view(s);

            TrainPersistence persist;
            persist.h_adam_m_io = h_adam_m;
            persist.h_adam_v_io = h_adam_v;
            persist.adam_t_io = &adam_t;
            persist.quiet = true;
            persist.total_opt_steps_override = total_opt_steps;
            persist.warmup_steps_override = warmup_steps;

            // One subtree, one Adam/RMSProp step (single_subtree semantics per file).
            // Save path is deferred to the super-epoch level below.
            run_radix_training(cfg, wo, h_weights, view.t,
                               /*epochs=*/1, entropy_lambda, mass_weight, subtree_splits, partition_depth, accumulate,
                               /*single_subtree=*/true, intermediate_weight,
                               optimizer, momentum_beta, rmsprop_beta,
                               lr_schedule, warmup_super_epochs,
                               weight_decay, grad_clip_norm, /*save_every=*/0,
                               curriculum, /*save_path=*/NULL,
                               /*lightning=*/LightningConfig{},
                               &persist);

            super_nodes_trained += s.n_nodes;
            subtrees_done++;

            free_radix_view(view);
            free_subtree(s);
        }
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("Super-epoch %d: %d subtrees, %lld radix nodes  (%.1f sec, adam_t=%d)\n",
               ep + 1, subtrees_done, super_nodes_trained, elapsed, adam_t);
        (void)super_loss_sum;  // loss is printed by run_radix_training when !quiet

        if (save_every > 0 && save_path && (ep + 1) % save_every == 0) {
            char ck_path[2048];
            snprintf(ck_path, sizeof(ck_path), "%s.ep%d", save_path, ep + 1);
            save_model_weights(ck_path, cfg, h_weights, wo);
            printf("  checkpoint: %s\n", ck_path);
        }
    }

    if (save_path) {
        save_model_weights(save_path, cfg, h_weights, wo);
        printf("Saved to %s\n", save_path);
    }
    free(h_adam_m); free(h_adam_v);
    if (lightning_rc_weights) free(lightning_rc_weights);
    printf("Done.\n");
    return 0;
}
