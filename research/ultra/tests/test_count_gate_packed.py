import unittest

import numpy as np

from agpt_ultra.count_gate import CountModel, PackedCountModel, expand_extra_features
from agpt_ultra.vectorized_count_prior import build_depth_prior_tables, compile_prior_rows_numpy


class PackedCountModelTest(unittest.TestCase):
    def test_matches_python_count_model_features_and_distribution(self) -> None:
        text = (b"to be or not to be, that is the question. " * 20)
        extra_features = expand_extra_features("entropy_delta,suffix_stats")
        depth = 8
        python_model = CountModel(text, 256, depth, extra_features)
        packed_model = PackedCountModel(text, 256, depth, extra_features)
        python_model.build()
        packed_model.build()
        theta = [0.2, -0.3, 0.1, 0.4, -0.05, 0.07, -0.11, 0.13, -0.17, 0.19, 0.01]

        for pos in range(1, len(text)):
            ctx = text[max(0, pos - depth) : pos]
            python_dist = python_model.gated_distribution(ctx, theta)
            packed_dist = packed_model.gated_distribution(ctx, theta)
            self.assertLess(
                max(abs(left - right) for left, right in zip(python_dist, packed_dist)),
                1.0e-12,
            )
            for d in range(1, min(depth, len(ctx)) + 1):
                python_features = python_model.features(ctx[-d:])
                packed_features = packed_model.features(ctx[-d:])
                self.assertEqual(python_features is None, packed_features is None)
                if python_features is not None and packed_features is not None:
                    self.assertLess(
                        max(abs(left - right) for left, right in zip(python_features, packed_features)),
                        1.0e-12,
                    )

    def test_vectorized_prior_compiler_matches_packed_model(self) -> None:
        text = (b"the rain in spain falls mainly on the plain. " * 10)
        extra_features = expand_extra_features("entropy_delta,suffix_stats")
        depth = 8
        model = PackedCountModel(text, 256, depth, extra_features)
        model.build()
        theta = [0.2, -0.3, 0.1, 0.4, -0.05, 0.07, -0.11, 0.13, -0.17, 0.19, 0.01]

        depth_tables = build_depth_prior_tables(model, theta)
        log_rows, feature_rows = compile_prior_rows_numpy(list(text), model, depth_tables, chunk_size=17)

        for pos in range(len(text) - 1):
            target_pos = pos + 1
            ctx = text[max(0, target_pos - depth) : target_pos]
            expected_log = np.log(np.asarray(model.gated_distribution(ctx, theta), dtype=np.float32))
            np.testing.assert_allclose(log_rows[pos], expected_log, rtol=1.0e-6, atol=1.0e-6)

            expected_features = None
            for d in range(min(depth, len(ctx)), 0, -1):
                expected_features = model.features(ctx[-d:])
                if expected_features is not None:
                    break
            if expected_features is None:
                expected_features = [0.0] * len(model.feature_names())
                expected_features[-1] = 1.0
            np.testing.assert_allclose(
                feature_rows[pos],
                np.asarray(expected_features, dtype=np.float32),
                rtol=1.0e-6,
                atol=1.0e-6,
            )


if __name__ == "__main__":
    unittest.main()
