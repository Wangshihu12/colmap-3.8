// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "glomap/observation_manager.h"

#include "base/reconstruction.h"

namespace colmap {

ObservationManager::ObservationManager(class Reconstruction& reconstruction)
    : reconstruction_(reconstruction) {}

void ObservationManager::DeleteAllPoints3D() {
  // 这里会清空全部 2D-3D 关联，因此只应在“重三角化重建结构”前使用。
  reconstruction_.DeleteAllPoints2DAndPoints3D();
}

size_t ObservationManager::FilterAllPoints3D(const double max_reproj_error,
                                             const double min_tri_angle) {
  return reconstruction_.FilterAllPoints3D(max_reproj_error, min_tri_angle);
}

size_t ObservationManager::FilterPoints3DInImages(
    const double max_reproj_error,
    const double min_tri_angle,
    const std::unordered_set<image_t>& image_ids) {
  return reconstruction_.FilterPoints3DInImages(
      max_reproj_error, min_tri_angle, image_ids);
}

size_t ObservationManager::FilterPoints3DWithShortTracks(
    const size_t min_track_length) {
  size_t num_filtered_observations = 0;
  const std::unordered_set<point3D_t> point3D_ids = reconstruction_.Point3DIds();
  for (const point3D_t point3D_id : point3D_ids) {
    if (!reconstruction_.ExistsPoint3D(point3D_id)) {
      continue;
    }
    const Point3D& point3D = reconstruction_.Point3D(point3D_id);
    if (point3D.Track().Length() < min_track_length) {
      // 这里统计的是被删除的观测数，不是点数，方便和其他过滤接口保持一致。
      num_filtered_observations += point3D.Track().Length();
      reconstruction_.DeletePoint3D(point3D_id);
    }
  }
  return num_filtered_observations;
}

size_t ObservationManager::FilterObservationsWithNegativeDepth() {
  return reconstruction_.FilterObservationsWithNegativeDepth();
}

}  // namespace colmap
