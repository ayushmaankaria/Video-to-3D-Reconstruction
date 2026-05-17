from __future__ import annotations

import unittest

import numpy as np

from spatial_recon.fusion import fuse_predictions


def _toy_predictions() -> dict:
    points = np.zeros((1, 2, 3, 3), dtype=np.float32)
    points[0, :, :, 0] = np.array([[0.001, 0.002, 0.021], [0.003, 0.004, 0.022]])
    points[0, :, :, 1] = np.array([[0.001, 0.002, 0.021], [0.003, 0.004, 0.022]])
    images = np.ones((1, 3, 2, 3), dtype=np.float32)
    conf = np.array([[[0.2, 0.9, 0.8], [0.7, 0.6, 0.5]]], dtype=np.float32)
    return {
        "world_points_from_depth": points,
        "depth_conf": conf,
        "images": images,
        "source_image_hw": np.asarray([[20, 30]], dtype=np.int32),
    }


class FusionTest(unittest.TestCase):
    def test_voxel_reduce_uses_majority_label(self) -> None:
        label_map = np.array([[1, 1, 2], [2, 2, 3]], dtype=np.int32)
        cloud = fuse_predictions(
            _toy_predictions(),
            label_maps=[label_map],
            labels={1: "desk", 2: "chair", 3: "monitor"},
            conf_percentile=0,
            sample_stride=1,
            voxel_size=0.01,
            max_points=100,
        )

        self.assertEqual(len(cloud.points), 2)
        self.assertEqual(sorted(cloud.semantic_ids.tolist()), [1, 2])

    def test_source_indices_survive_fusion(self) -> None:
        cloud = fuse_predictions(
            _toy_predictions(),
            label_maps=[np.ones((2, 3), dtype=np.int32)],
            labels={1: "desk"},
            conf_percentile=0,
            sample_stride=1,
            voxel_size=0.01,
            max_points=100,
        )

        self.assertEqual(cloud.source_frame.shape, (len(cloud.points),))
        self.assertEqual(cloud.source_y.shape, (len(cloud.points),))
        self.assertEqual(cloud.source_x.shape, (len(cloud.points),))
        self.assertEqual(cloud.source_y_vggt.shape, (len(cloud.points),))
        self.assertEqual(cloud.source_x_vggt.shape, (len(cloud.points),))
        self.assertTrue(np.all(cloud.source_frame == 0))
        self.assertTrue(np.all((0 <= cloud.source_y) & (cloud.source_y < 20)))
        self.assertTrue(np.all((0 <= cloud.source_x) & (cloud.source_x < 30)))
        self.assertTrue(np.all((0 <= cloud.source_y_vggt) & (cloud.source_y_vggt < 2)))
        self.assertTrue(np.all((0 <= cloud.source_x_vggt) & (cloud.source_x_vggt < 3)))

    def test_nonfinite_points_are_dropped(self) -> None:
        predictions = _toy_predictions()
        predictions["world_points_from_depth"] = predictions["world_points_from_depth"].copy()
        predictions["world_points_from_depth"][0, 0, 0, 0] = np.nan

        cloud = fuse_predictions(
            predictions,
            label_maps=[np.ones((2, 3), dtype=np.int32)],
            labels={1: "desk"},
            conf_percentile=0,
            sample_stride=1,
            voxel_size=0,
            max_points=100,
        )

        self.assertEqual(len(cloud.points), 5)
        self.assertTrue(np.isfinite(cloud.points).all())

    def test_voxel_reduce_collapses_identical_points(self) -> None:
        predictions = {
            "world_points_from_depth": np.zeros((1, 1, 2, 3), dtype=np.float32),
            "depth_conf": np.ones((1, 1, 2), dtype=np.float32),
            "images": np.ones((1, 3, 1, 2), dtype=np.float32),
            "source_image_hw": np.asarray([[10, 20]], dtype=np.int32),
        }
        cloud = fuse_predictions(
            predictions,
            label_maps=[np.asarray([[1, 1]], dtype=np.int32)],
            labels={1: "desk"},
            conf_percentile=0,
            sample_stride=1,
            voxel_size=0.01,
            max_points=100,
        )

        self.assertEqual(len(cloud.points), 1)

    def test_numpy_unique_inverse_is_flattened_before_voting(self) -> None:
        predictions = {
            "world_points_from_depth": np.asarray(
                [[[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.04, 0.0, 0.0]]]],
                dtype=np.float32,
            ),
            "depth_conf": np.ones((1, 1, 3), dtype=np.float32),
            "images": np.ones((1, 3, 1, 3), dtype=np.float32),
            "source_image_hw": np.asarray([[10, 30]], dtype=np.int32),
        }
        cloud = fuse_predictions(
            predictions,
            label_maps=[np.asarray([[5, 1000, 5]], dtype=np.int32)],
            labels={5: "desk", 1000: "rare_large_label"},
            conf_percentile=0,
            sample_stride=1,
            voxel_size=0.005,
            max_points=100,
        )

        self.assertEqual(len(cloud.points), 3)
        self.assertEqual(sorted(cloud.semantic_ids.tolist()), [5, 5, 1000])


if __name__ == "__main__":
    unittest.main()
