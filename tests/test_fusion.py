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
        self.assertTrue(np.all(cloud.source_frame == 0))


if __name__ == "__main__":
    unittest.main()
