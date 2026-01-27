# Pose Graph Optimization (PGO) Pipeline

This document describes the standalone PGO pipeline test executable added in
`test/test_vpgo_pipeline.cc`. It is designed to make the PGO flow easy to run,
export/import edges, and re-run PGO with manually provided loop edges.

## Overview

The pipeline performs the following steps:

1. Read a COLMAP sparse reconstruction.
2. Build odometry edges (sequential edges per camera).
3. Build loop edges from two-view geometries in the database.
4. Optionally add manual loop edges from a file.
5. Save all edges to a text file (stream-friendly format).
6. Optimize the pose graph (SE3) with Ceres.
7. Write the optimized reconstruction back to disk.
8. Optionally check mean reprojection error against a threshold.

## Executable

The test target is built as:

- `test_vpgo_pipeline`

Build via the usual COLMAP build process (this target is added under
`test/CMakeLists.txt`).

## Usage

Basic run:

```
./bin/test_vpgo_pipeline <sparse_path> <database_path> <output_path>
```

Manual recall (read edges from file instead of rebuilding):

```
./bin/test_vpgo_pipeline --edge-input <edge_file> <sparse_path> <database_path> <output_path>
```

Add manual loop edges while still building edges:

```
./bin/test_vpgo_pipeline --manual-loop <loop_edges_file> <sparse_path> <database_path> <output_path>
```

Validate error threshold (12px example):

```
./bin/test_vpgo_pipeline --eval-threshold 12 <sparse_path> <database_path> <output_path>
```

Run IO self-test:

```
./bin/test_vpgo_pipeline --self-test
```

## Edge File Format

The edge file is line-based and stream-friendly. Lines starting with `#` are
comments. Each non-comment line is:

```
<type> <id1> <id2> <qw> <qx> <qy> <qz> <tx> <ty> <tz> <t_is_unit> <rot_w> <trans_w>
```

- `type` is `odom` or `loop`.
- `t_is_unit` is `1` if translation is a direction (unit vector), `0` for full
  translation.
- `rot_w` and `trans_w` are the rotation/translation weights.

Legacy format is also accepted (no type prefix):

```
<id1> <id2> <qw> <qx> <qy> <qz> <tx> <ty> <tz>
```

These lines are interpreted as loop edges with unit translation direction.

## Manual Recall Workflow

1. Run once to generate an edge file:
   `./bin/test_vpgo_pipeline <sparse> <db> <out>`
2. Edit or replace the edge file as needed.
3. Re-run PGO using `--edge-input`:
   `./bin/test_vpgo_pipeline --edge-input <edge_file> <sparse> <db> <out>`

This supports manual loop edge injection and repeated PGO runs without
rebuilding edges from the database.

## Error Constraint for 12px Target

The executable prints the mean reprojection error before and after PGO using
`Reconstruction::ComputeMeanReprojectionError()`.

To validate datasets that require the optimized result to be within 12 pixels,
run:

```
./bin/test_vpgo_pipeline --eval-threshold 12 <sparse> <db> <out>
```

Repeat for at least 3 datasets that exhibit cumulative drift to verify the
constraint. The pipeline exits non-zero if the threshold is violated.

## Notes

- Loop edges are filtered by inlier count, inlier ratio, triangulation angle,
  and geometric error (Sampson or homography residuals).
- Odometry edges are built per camera by ordering `image_id` and connecting
  adjacent frames.
- Loop edges use a Huber loss to reduce the impact of false positives.
