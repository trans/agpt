#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_i32(FILE* f) {
    int v = 0;
    if (fread(&v, 4, 1, f) != 1) return -1;
    return v;
}

static unsigned read_u32(FILE* f) {
    unsigned v = 0;
    if (fread(&v, 4, 1, f) != 1) return 0;
    return v;
}

typedef struct {
    int d_model, n_heads, n_layers, d_ff, vocab_size, seq_len;
} Header;

typedef struct {
    char name[32];
    long data_off;
    int rows;
    int cols;
    int count;
} Mat;

typedef struct {
    double mean;
    double stddev;
    float minv;
    float maxv;
    double l2;
    int zeros;
} Stats;

static void die(const char* msg, const char* path) {
    if (path) fprintf(stderr, "%s: %s\n", msg, path);
    else fprintf(stderr, "%s\n", msg);
    exit(2);
}

static Header read_header(FILE* f, const char* path) {
    Header h;
    unsigned magic = read_u32(f);
    if (magic != 0x4D475054u) die("bad MGPT magic", path);
    h.d_model = read_i32(f);
    h.n_heads = read_i32(f);
    h.n_layers = read_i32(f);
    h.d_ff = read_i32(f);
    h.vocab_size = read_i32(f);
    h.seq_len = read_i32(f);
    if (h.d_model <= 0 || h.n_heads <= 0 || h.n_layers <= 0 || h.d_ff <= 0 ||
        h.vocab_size <= 0 || h.seq_len <= 0) {
        die("invalid header", path);
    }
    return h;
}

static int build_mat_index(FILE* f, const char* path, Mat* mats, int cap, Header* out_h) {
    Header h = read_header(f, path);
    *out_h = h;
    int nm = 0;

    #define RECORD(name_) do { \
        if (nm >= cap) die("too many matrices", path); \
        int rows = read_i32(f); \
        int cols = read_i32(f); \
        if (rows <= 0 || cols <= 0) die("bad matrix shape", path); \
        snprintf(mats[nm].name, sizeof(mats[nm].name), "%s", name_); \
        mats[nm].rows = rows; \
        mats[nm].cols = cols; \
        mats[nm].count = rows * cols; \
        mats[nm].data_off = ftell(f); \
        if (fseek(f, (long)mats[nm].count * 4L, SEEK_CUR) != 0) die("bad matrix seek", path); \
        nm++; \
    } while (0)

    RECORD("token_emb");
    for (int i = 0; i < h.n_layers; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "L%d.wq_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wq_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wk_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wk_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wv_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wv_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wo_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wo_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.ln1_g", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.ln1_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.l1_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.l1_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.l2_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.l2_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.ln2_g", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.ln2_b", i); RECORD(buf);
    }
    RECORD("final_gamma");
    RECORD("final_beta");
    RECORD("out_w");
    RECORD("out_b");
    return nm;
}

static void read_mat_values(FILE* f, const Mat* m, float* out, const char* path) {
    if (fseek(f, m->data_off, SEEK_SET) != 0) die("seek failed", path);
    if ((int)fread(out, sizeof(float), (size_t)m->count, f) != m->count) {
        die("read failed", path);
    }
}

static Stats compute_stats(const float* a, int n) {
    Stats s;
    double sum = 0.0, sumsq = 0.0;
    s.minv = a[0];
    s.maxv = a[0];
    s.zeros = 0;
    for (int i = 0; i < n; i++) {
        double v = (double)a[i];
        sum += v;
        sumsq += v * v;
        if (a[i] < s.minv) s.minv = a[i];
        if (a[i] > s.maxv) s.maxv = a[i];
        if (a[i] == 0.0f) s.zeros++;
    }
    s.mean = sum / (double)n;
    double var = sumsq / (double)n - s.mean * s.mean;
    if (var < 0.0) var = 0.0;
    s.stddev = sqrt(var);
    s.l2 = sqrt(sumsq);
    return s;
}

static void print_header(const char* label, Header h) {
    printf("%s: d=%d heads=%d layers=%d ff=%d vocab=%d seq=%d\n",
           label, h.d_model, h.n_heads, h.n_layers, h.d_ff, h.vocab_size, h.seq_len);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <a.model> <b.model>\n", argv[0]);
        return 2;
    }

    const char* path_a = argv[1];
    const char* path_b = argv[2];
    FILE* fa = fopen(path_a, "rb");
    FILE* fb = fopen(path_b, "rb");
    if (!fa) die("cannot open", path_a);
    if (!fb) die("cannot open", path_b);

    Mat mats_a[256], mats_b[256];
    Header ha, hb;
    int na = build_mat_index(fa, path_a, mats_a, 256, &ha);
    int nb = build_mat_index(fb, path_b, mats_b, 256, &hb);
    if (na != nb) die("matrix count mismatch", NULL);

    print_header("A", ha);
    print_header("B", hb);
    printf("header_match: %s\n\n",
           (ha.d_model == hb.d_model && ha.n_heads == hb.n_heads &&
            ha.n_layers == hb.n_layers && ha.d_ff == hb.d_ff &&
            ha.vocab_size == hb.vocab_size && ha.seq_len == hb.seq_len) ? "yes" : "no");

    printf("%-12s %-10s %-10s %-11s %-11s %-11s %-11s %-11s %-11s\n",
           "matrix", "shape", "changed%", "a.std", "b.std", "meanΔ", "stdΔ", "max|Δ|", "rmse");

    for (int i = 0; i < na; i++) {
        const Mat* ma = &mats_a[i];
        const Mat* mb = &mats_b[i];
        if (strcmp(ma->name, mb->name) != 0 || ma->rows != mb->rows || ma->cols != mb->cols) {
            fprintf(stderr, "matrix mismatch at %d: %s vs %s\n", i, ma->name, mb->name);
            return 2;
        }

        float* a = (float*)malloc((size_t)ma->count * sizeof(float));
        float* b = (float*)malloc((size_t)mb->count * sizeof(float));
        if (!a || !b) die("alloc failed", NULL);
        read_mat_values(fa, ma, a, path_a);
        read_mat_values(fb, mb, b, path_b);

        Stats sa = compute_stats(a, ma->count);
        Stats sb = compute_stats(b, mb->count);
        double diff_sumsq = 0.0;
        float max_abs = 0.0f;
        int changed = 0;
        for (int j = 0; j < ma->count; j++) {
            float d = a[j] - b[j];
            float ad = d < 0 ? -d : d;
            if (ad != 0.0f) changed++;
            if (ad > max_abs) max_abs = ad;
            diff_sumsq += (double)d * (double)d;
        }
        double rmse = sqrt(diff_sumsq / (double)ma->count);
        double changed_pct = 100.0 * (double)changed / (double)ma->count;

        printf("%-12s %4dx%-5d %9.3f%% %11.6f %11.6f %11.6f %11.6f %11.6g %11.6g\n",
               ma->name, ma->rows, ma->cols, changed_pct,
               sa.stddev, sb.stddev, sb.mean - sa.mean, sb.stddev - sa.stddev,
               max_abs, rmse);

        free(a);
        free(b);
    }

    fclose(fa);
    fclose(fb);
    return 0;
}
