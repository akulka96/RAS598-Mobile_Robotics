#!/usr/bin/env python3

"""
Assignment 1 - Perception Pipeline
ROS 2 node for cylinder detection and color classification from /oakd/points

Pipeline:
1. Box filter
2. Voxel downsample
3. Normal estimation using SVD
4. Iterative plane removal using RANSAC
5. Euclidean clustering
6. Cylinder RANSAC on each cluster
7. HSV color classification
8. RViz visualization with point clouds and markers
"""

from collections import deque
import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray


class PipelineConfig:
    def __init__(self):
        self.input_topic = '/oakd/points'

        # Box filter bounds in meters
        self.box_min = np.array([-1.0, -0.6, 0.2], dtype=np.float32)
        self.box_max = np.array([1.0, 0.6, 2.0], dtype=np.float32)

        # Downsampling
        self.voxel_size = 0.02

        # Normals
        self.normal_k = 15

        # Plane removal
        self.plane_dist_thresh = 0.02
        self.expected_vertical = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.plane_alignment_thresh = 0.85
        self.max_planes_to_remove = 3
        self.plane_ransac_iters = 100

        # Clustering
        self.cluster_radius = 0.06
        self.cluster_min_size = 70
        self.cluster_max_size = 5000

        # Cylinder detection
        self.expected_cylinder_radius = 0.055
        self.cylinder_radius_tol = 0.015
        self.cylinder_axis_alignment_thresh = 0.80
        self.cylinder_min_inliers = 30
        self.cylinder_ransac_iters = 300
        self.max_cylinders = 3


class CylinderVisualizer:
    def __init__(self, marker_pub):
        self.marker_pub = marker_pub

    def _make_delete_all_marker(self):
        marker = Marker()
        marker.action = Marker.DELETEALL
        return marker

    def _make_cylinder_marker(self, center, radius, display_rgb, marker_id, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = 'detected_cylinders'
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = float(center[0])
        marker.pose.position.y = 0.0
        marker.pose.position.z = float(center[2])

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = float(radius * 2.0)
        marker.scale.y = float(radius * 2.0)
        marker.scale.z = 0.4

        marker.color.r = float(display_rgb[0])
        marker.color.g = float(display_rgb[1])
        marker.color.b = float(display_rgb[2])
        marker.color.a = 0.8
        return marker

    def _make_text_marker(self, center, label, marker_id, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = 'detected_cylinder_labels'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(center[0])
        marker.pose.position.y = 0.0
        marker.pose.position.z = float(center[2] + 0.25)
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.08
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = label
        return marker

    def publish(self, detections, frame_id):
        marker_array = MarkerArray()
        marker_array.markers.append(self._make_delete_all_marker())

        for i, detection in enumerate(detections):
            model, display_rgb, label = detection
            center, _, radius = model

            marker_array.markers.append(
                self._make_cylinder_marker(center, radius, display_rgb, 1000 + i, frame_id)
            )
            marker_array.markers.append(
                self._make_text_marker(center, label, 2000 + i, frame_id)
            )

        self.marker_pub.publish(marker_array)


class CylinderPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def rgb_to_hsv(self, r, g, b):
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val

        if delta == 0.0:
            hue = 0.0
        elif max_val == r:
            hue = (60.0 * ((g - b) / delta) + 360.0) % 360.0
        elif max_val == g:
            hue = (60.0 * ((b - r) / delta) + 120.0) % 360.0
        else:
            hue = (60.0 * ((r - g) / delta) + 240.0) % 360.0

        sat = 0.0 if max_val == 0.0 else delta / max_val
        val = max_val
        return hue, sat, val

    def classify_color(self, hue, sat, val):
        if val < 0.15:
            return 'unknown', np.array([0.5, 0.5, 0.5], dtype=np.float32)

        if sat < 0.08:
            return 'unknown', np.array([0.5, 0.5, 0.5], dtype=np.float32)

        if hue < 20.0 or hue >= 335.0:
            return 'red', np.array([1.0, 0.0, 0.0], dtype=np.float32)

        if 90.0 <= hue <= 150.0:
            return 'green', np.array([0.0, 1.0, 0.0], dtype=np.float32)

        if 200.0 <= hue <= 260.0:
            return 'blue', np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Pink range
        if 285.0 <= hue < 335.0 and sat >= 0.12:
            return 'pink', np.array([1.0, 0.4, 0.7], dtype=np.float32)

        return 'unknown', np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def box_filter(self, points, colors):
        mask = (
            (points[:, 0] >= self.cfg.box_min[0]) & (points[:, 0] <= self.cfg.box_max[0]) &
            (points[:, 1] >= self.cfg.box_min[1]) & (points[:, 1] <= self.cfg.box_max[1]) &
            (points[:, 2] >= self.cfg.box_min[2]) & (points[:, 2] <= self.cfg.box_max[2])
        )
        return points[mask], colors[mask]

    def downsample(self, points, colors):
        if len(points) == 0:
            return points, colors

        voxel_indices = np.floor(points / self.cfg.voxel_size).astype(np.int32)
        _, first_indices = np.unique(voxel_indices, axis=0, return_index=True)
        return points[first_indices], colors[first_indices]

    def get_knn_indices(self, points, query_points, k):
        if len(points) < k:
            return None
        tree = cKDTree(points)
        _, indices = tree.query(query_points, k=k)
        return indices

    def estimate_normals(self, points):
        normals = np.zeros((len(points), 3), dtype=np.float64)
        knn_indices = self.get_knn_indices(points, points, self.cfg.normal_k)
        if knn_indices is None:
            return normals

        for i in range(len(points)):
            neighbors = points[knn_indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            normals[i] = vt[-1]

        return normals

    def find_plane_ransac(self, points):
        best_count = 0
        best_normal = None
        best_d = None
        best_mask = None

        if len(points) < 3:
            return None, None, None

        for _ in range(self.cfg.plane_ransac_iters):
            sample_idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[sample_idx]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-6:
                continue

            normal = normal / normal_norm

            if abs(np.dot(normal, self.cfg.expected_vertical)) < self.cfg.plane_alignment_thresh:
                continue

            d = -np.dot(normal, p1)
            distances = np.abs(points @ normal + d)
            inlier_mask = distances < self.cfg.plane_dist_thresh
            count = int(np.sum(inlier_mask))

            if count > best_count:
                best_count = count
                best_normal = normal
                best_d = d
                best_mask = inlier_mask

        return best_normal, best_d, best_mask

    def euclidean_clustering(self, points):
        if len(points) == 0:
            return []

        visited = np.zeros(len(points), dtype=bool)
        tree = cKDTree(points)
        clusters = []

        for i in range(len(points)):
            if visited[i]:
                continue

            queue = deque([i])
            visited[i] = True
            cluster_indices = []

            while queue:
                current = queue.popleft()
                cluster_indices.append(current)

                neighbors = tree.query_ball_point(points[current], r=self.cfg.cluster_radius)
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)

            cluster_size = len(cluster_indices)
            if self.cfg.cluster_min_size <= cluster_size <= self.cfg.cluster_max_size:
                clusters.append(np.array(cluster_indices, dtype=np.int32))

        return clusters

    def find_single_cylinder(self, points, normals):
        if len(points) < 20:
            return None

        best_count = 0
        best_result = None
        vertical = self.cfg.expected_vertical

        for _ in range(self.cfg.cylinder_ransac_iters):
            sample_idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[sample_idx[0]], points[sample_idx[1]]
            n1, n2 = normals[sample_idx[0]], normals[sample_idx[1]]

            axis = np.cross(n1, n2)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                continue

            axis = axis / axis_norm

            if np.dot(axis, vertical) < 0:
                axis = -axis

            if abs(np.dot(axis, vertical)) < self.cfg.cylinder_axis_alignment_thresh:
                continue

            vectors = points - p1
            proj_lengths = vectors @ axis
            proj_vectors = np.outer(proj_lengths, axis)
            perp_vectors = vectors - proj_vectors
            perp_distances = np.linalg.norm(perp_vectors, axis=1)

            inlier_mask = np.abs(
                perp_distances - self.cfg.expected_cylinder_radius
            ) < self.cfg.cylinder_radius_tol
            count = int(np.sum(inlier_mask))

            if count > best_count:
                inlier_points = points[inlier_mask]
                if len(inlier_points) == 0:
                    continue

                best_count = count
                center = inlier_points.mean(axis=0)
                radius = float(perp_distances[inlier_mask].mean())
                best_result = (center, axis, radius, inlier_mask)

        if best_count < self.cfg.cylinder_min_inliers:
            return None

        return best_result


class CylinderProcessorNode(Node):
    def __init__(self):
        super().__init__('cylinder_processor_node')

        self.cfg = PipelineConfig()
        self.pipeline = CylinderPipeline(self.cfg)

        self.stage0_pub = self.create_publisher(PointCloud2, '/assignment1/stage0_box', 10)
        self.stage1_pub = self.create_publisher(PointCloud2, '/assignment1/stage1_no_planes', 10)
        self.stage2_pub = self.create_publisher(PointCloud2, '/assignment1/stage2_clusters', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/assignment1/cylinder_markers', 10)

        self.visualizer = CylinderVisualizer(self.marker_pub)

        self.subscription = self.create_subscription(
            PointCloud2,
            self.cfg.input_topic,
            self.listener_callback,
            10
        )

        self.get_logger().info('CylinderProcessorNode ready.')

    def decode_rgb_column(self, packed_rgb_float_column):
        rgb_u32 = packed_rgb_float_column.view(np.uint32)
        r = ((rgb_u32 >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((rgb_u32 >> 8) & 0xFF).astype(np.float32) / 255.0
        b = (rgb_u32 & 0xFF).astype(np.float32) / 255.0
        return np.stack([r, g, b], axis=1)

    def numpy_to_pc2_rgb(self, points, colors, frame_id):
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(points)
        msg.is_dense = True

        clipped = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint32)
        packed_rgb = ((clipped[:, 0] << 16) | (clipped[:, 1] << 8) | clipped[:, 2]).view(np.float32)

        cloud_array = np.hstack([points.astype(np.float32), packed_rgb.reshape(-1, 1)])
        msg.data = cloud_array.tobytes()
        return msg

    def publish_colored_clusters(self, clusters, points, frame_id):
        palette = np.array([
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.2, 1.0],
            [1.0, 1.0, 0.2],
            [1.0, 0.2, 1.0],
            [0.2, 1.0, 1.0],
        ], dtype=np.float32)

        all_points = []
        all_colors = []

        for i, cluster_indices in enumerate(clusters):
            color = palette[i % len(palette)]
            all_points.append(points[cluster_indices])
            all_colors.append(np.tile(color, (len(cluster_indices), 1)))

        if all_points:
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors)
            self.stage2_pub.publish(self.numpy_to_pc2_rgb(merged_points, merged_colors, frame_id))

    def listener_callback(self, msg):
        frame_id = msg.header.frame_id

        stride = msg.point_step // 4
        raw = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, stride)

        points = raw[:, :3].copy()
        finite_mask = np.all(np.isfinite(points), axis=1)
        points = points[finite_mask]

        colors = self.decode_rgb_column(raw[finite_mask, 4].copy())

        if len(points) < 100:
            return

        box_points, box_colors = self.pipeline.box_filter(points, colors)
        if len(box_points) < 100:
            self.get_logger().warn(f'Box filter left only {len(box_points)} points.')
            return

        ds_points, ds_colors = self.pipeline.downsample(box_points, box_colors)
        if len(ds_points) < 50:
            self.get_logger().warn('Downsample left fewer than 50 points.')
            return

        self.stage0_pub.publish(self.numpy_to_pc2_rgb(ds_points, ds_colors, frame_id))

        normals = self.pipeline.estimate_normals(ds_points)

        work_points = ds_points.copy()
        work_colors = ds_colors.copy()
        work_normals = normals.copy()

        for _ in range(self.cfg.max_planes_to_remove):
            if len(work_points) < 50:
                break

            _, _, plane_inlier_mask = self.pipeline.find_plane_ransac(work_points)
            if plane_inlier_mask is None or int(np.sum(plane_inlier_mask)) < 20:
                break

            keep_mask = ~plane_inlier_mask
            work_points = work_points[keep_mask]
            work_colors = work_colors[keep_mask]
            work_normals = work_normals[keep_mask]

        self.stage1_pub.publish(self.numpy_to_pc2_rgb(work_points, work_colors, frame_id))

        if len(work_points) < 20:
            self.visualizer.publish([], frame_id)
            return

        clusters = self.pipeline.euclidean_clustering(work_points)
        if not clusters:
            self.visualizer.publish([], frame_id)
            return

        clusters = sorted(clusters, key=lambda c: len(c), reverse=True)

        self.publish_colored_clusters(clusters, work_points, frame_id)

        detections = []

        for cluster_indices in clusters:
            if len(detections) >= self.cfg.max_cylinders:
                break

            cluster_points = work_points[cluster_indices]
            cluster_colors = work_colors[cluster_indices]
            cluster_normals = work_normals[cluster_indices]

            cylinder_result = self.pipeline.find_single_cylinder(cluster_points, cluster_normals)
            if cylinder_result is None:
                continue

            center, axis, radius, inlier_mask = cylinder_result

            avg_rgb = cluster_colors[inlier_mask].mean(axis=0)
            hue, sat, val = self.pipeline.rgb_to_hsv(
                float(avg_rgb[0]),
                float(avg_rgb[1]),
                float(avg_rgb[2])
            )
            label, display_rgb = self.pipeline.classify_color(hue, sat, val)
            model = (center, axis, radius)

            detections.append((model, display_rgb, label))

            self.get_logger().info(
                f'Cylinder: label={label}  '
                f'center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})  '
                f'r={radius:.3f} m  inliers={int(np.sum(inlier_mask))}'
            )

        self.visualizer.publish(detections, frame_id)


def main():
    rclpy.init()
    node = CylinderProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()