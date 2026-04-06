// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "glomap/pose_graph.h"

#include <queue>

#include "base/reconstruction.h"

namespace colmap {

bool PoseGraph::Load(const Database& database,
                     const Reconstruction& reconstruction,
                     const size_t min_num_matches,
                     const bool ignore_watermarks) {
  // 从数据库批量读取两视图几何。这里仍以旧版 COLMAP 的
  // `two_view_geometries` 表作为位姿图来源。
  std::vector<image_pair_t> pair_ids;
  std::vector<TwoViewGeometry> geometries;
  database.ReadTwoViewGeometries(&pair_ids, &geometries);

  edges_.clear();
  pair_id_to_index_.clear();
  edges_.reserve(pair_ids.size());

  for (size_t i = 0; i < pair_ids.size(); ++i) {
    const TwoViewGeometry& geometry = geometries[i];
    // 先做最基本的边筛选，避免极弱边一开始就进入全局图。
    if (geometry.inlier_matches.size() < min_num_matches) {
      continue;
    }
    if (ignore_watermarks && geometry.config == TwoViewGeometry::WATERMARK) {
      continue;
    }

    image_t image_id1;
    image_t image_id2;
    Database::PairIdToImagePair(pair_ids[i], &image_id1, &image_id2);
    if (!reconstruction.ExistsImage(image_id1) ||
        !reconstruction.ExistsImage(image_id2)) {
      continue;
    }

    ViewEdge edge;
    edge.image_id1 = image_id1;
    edge.image_id2 = image_id2;
    edge.pair_id = pair_ids[i];
    edge.weight = static_cast<double>(geometry.inlier_matches.size());
    edge.valid = true;
    edge.geometry = geometry;

    pair_id_to_index_[edge.pair_id] = edges_.size();
    edges_.push_back(std::move(edge));
  }

  return !edges_.empty();
}

const std::vector<ViewEdge>& PoseGraph::Edges() const { return edges_; }

std::vector<ViewEdge>& PoseGraph::MutableEdges() { return edges_; }

std::vector<ViewEdge> PoseGraph::ValidEdges() const {
  std::vector<ViewEdge> valid_edges;
  valid_edges.reserve(edges_.size());
  for (const ViewEdge& edge : edges_) {
    // 当前实现约定：只有 `valid=true` 且 `weight>0` 的边才算真正参与优化。
    if (edge.valid && edge.weight > 0.0) {
      valid_edges.push_back(edge);
    }
  }
  return valid_edges;
}

bool PoseGraph::Empty() const { return NumValidEdges() == 0; }

size_t PoseGraph::NumValidEdges() const {
  size_t num_valid_edges = 0;
  for (const ViewEdge& edge : edges_) {
    if (edge.valid && edge.weight > 0.0) {
      ++num_valid_edges;
    }
  }
  return num_valid_edges;
}

std::unordered_set<image_t> PoseGraph::ComputeLargestConnectedImageComponent()
    const {
  // 用有效边建立无向邻接表，再做 BFS 找最大连通分量。
  std::unordered_map<image_t, std::vector<size_t>> adjacency;
  for (size_t edge_idx = 0; edge_idx < edges_.size(); ++edge_idx) {
    const ViewEdge& edge = edges_[edge_idx];
    if (!edge.valid || edge.weight <= 0.0) {
      continue;
    }
    adjacency[edge.image_id1].push_back(edge_idx);
    adjacency[edge.image_id2].push_back(edge_idx);
  }

  std::unordered_set<image_t> visited;
  std::unordered_set<image_t> best_component;
  for (const auto& item : adjacency) {
    if (visited.count(item.first) > 0) {
      continue;
    }

    std::queue<image_t> queue;
    std::unordered_set<image_t> component;
    queue.push(item.first);
    visited.insert(item.first);
    component.insert(item.first);

    while (!queue.empty()) {
      const image_t image_id = queue.front();
      queue.pop();
      for (const size_t edge_idx : adjacency.at(image_id)) {
        const ViewEdge& edge = edges_[edge_idx];
        const image_t next_image_id =
            edge.image_id1 == image_id ? edge.image_id2 : edge.image_id1;
        if (visited.insert(next_image_id).second) {
          component.insert(next_image_id);
          queue.push(next_image_id);
        }
      }
    }

    if (component.size() > best_component.size()) {
      best_component = std::move(component);
    }
  }

  return best_component;
}

void PoseGraph::InvalidatePairsOutsideActiveImageIds(
    const std::unordered_set<image_t>& active_image_ids) {
  for (ViewEdge& edge : edges_) {
    // 只保留主连通分量中的边，避免离群子图干扰全局求解。
    if (active_image_ids.count(edge.image_id1) == 0 ||
        active_image_ids.count(edge.image_id2) == 0) {
      edge.valid = false;
    }
  }
}

}  // namespace colmap
