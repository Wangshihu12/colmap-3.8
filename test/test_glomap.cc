// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include <cstdio>
#include <iostream>
#include <memory>
#include <fstream>
#include <string>
#include <vector>

#include "base/camera.h"
#include "base/camera_models.h"
#include "base/database.h"
#include "base/pose.h"
#include "base/projection.h"
#include "base/reconstruction.h"
#include "base/reconstruction_manager.h"
#include "feature/types.h"
#include "glomap/global_pipeline.h"
#include "glomap/global_mapper.h"
#include "util/misc.h"

namespace colmap {
namespace {

struct SyntheticImageData {
  // synthetic 测试里每张图像的位姿与关键点缓存。
  image_t image_id = kInvalidImageId;
  Eigen::Vector4d qvec = Eigen::Vector4d(1, 0, 0, 0);
  Eigen::Vector3d tvec = Eigen::Vector3d::Zero();
  // keypoints 形状为 [N, 2]，存每个三维点在该图上的投影像素坐标。
  FeatureKeypoints keypoints;
};

/**
 * @brief 功能描述：将世界坐标中的 3D 点投影到指定相机图像上。
 * @param camera 相机内参对象。
 * @param qvec 世界到相机的旋转四元数。
 * @param tvec 世界到相机的平移向量。
 * @param xyz 世界坐标系下的三维点。
 * @return 返回值说明：返回投影后的像素坐标。
 */
Eigen::Vector2d ProjectPoint(const Camera& camera,
                             const Eigen::Vector4d& qvec,
                             const Eigen::Vector3d& tvec,
                             const Eigen::Vector3d& xyz) {
  const Eigen::Vector3d xyz_cam = QuaternionToRotationMatrix(qvec) * xyz + tvec;
  return camera.WorldToImage(xyz_cam.hnormalized());
}

/**
 * @brief 功能描述：构造一个最小可运行的三视图合成数据库，用于验证全局 SfM 接口。
 * @param database_path 输出 SQLite 数据库路径。
 * @return 返回值说明：数据库构造成功返回 true，否则返回 false。
 */
bool CreateSyntheticDatabase(const std::string& database_path) {
  std::remove(database_path.c_str());

  Database database(database_path);

  // 构造一个简单 pinhole 相机，便于稳定验证主流程是否贯通。
  Camera camera;
  camera.SetCameraId(1);
  camera.SetModelId(SimplePinholeCameraModel::kModelId);
  camera.SetWidth(1280);
  camera.SetHeight(960);
  camera.SetParams({900.0, 640.0, 480.0});
  camera.SetPriorFocalLength(true);
  database.WriteCamera(camera, true);

  std::vector<Eigen::Vector3d> points3D;
  // 在相机前方生成一个稠密但很小的规则点云，保证三视图之间都有稳定共视。
  for (int y = -1; y <= 1; ++y) {
    for (int x = -2; x <= 2; ++x) {
      points3D.emplace_back(0.3 * x, 0.2 * y, 4.0 + 0.15 * (x + 2));
    }
  }

  std::vector<SyntheticImageData> images(3);
  // 三张图像沿 x 方向平移，形成一个最小可三角化基线。
  images[0].image_id = 1;
  images[0].tvec = Eigen::Vector3d(0.0, 0.0, 0.0);
  images[1].image_id = 2;
  images[1].tvec = Eigen::Vector3d(-1.0, 0.0, 0.0);
  images[2].image_id = 3;
  images[2].tvec = Eigen::Vector3d(-2.0, -0.1, 0.0);

  for (size_t i = 0; i < images.size(); ++i) {
    Image image;
    image.SetImageId(images[i].image_id);
    image.SetCameraId(camera.CameraId());
    image.SetName("synthetic_" + std::to_string(i) + ".jpg");
    database.WriteImage(image, true);
  }

  for (SyntheticImageData& image_data : images) {
    image_data.keypoints.reserve(points3D.size());
    for (const Eigen::Vector3d& point3D : points3D) {
      const Eigen::Vector2d xy =
          ProjectPoint(camera, image_data.qvec, image_data.tvec, point3D);
      image_data.keypoints.emplace_back(
          static_cast<float>(xy(0)), static_cast<float>(xy(1)));
    }
    database.WriteKeypoints(image_data.image_id, image_data.keypoints);
  }

  for (size_t i = 0; i < images.size(); ++i) {
    for (size_t j = i + 1; j < images.size(); ++j) {
      TwoViewGeometry geometry;
      geometry.config = TwoViewGeometry::CALIBRATED;
      // 合成数据里每个点在每对图像上都一一对应，因此直接生成完美匹配。
      geometry.inlier_matches.reserve(points3D.size());
      for (point2D_t point_idx = 0; point_idx < points3D.size(); ++point_idx) {
        geometry.inlier_matches.emplace_back(point_idx, point_idx);
      }

      ComputeRelativePose(images[i].qvec,
                          images[i].tvec,
                          images[j].qvec,
                          images[j].tvec,
                          &geometry.qvec,
                          &geometry.tvec);
      database.WriteTwoViewGeometry(
          images[i].image_id, images[j].image_id, geometry);
    }
  }

  return true;
}

/**
 * @brief 功能描述：执行一次 GLOMAP 风格全局重建，并将结果输出到模型目录。
 * @param database_path 输入数据库路径。
 * @param output_path 输出重建目录。
 * @return 返回值说明：执行成功返回 true，否则返回 false。
 */
bool RunGlomapPipeline(const std::string& database_path,
                       const std::string& output_path,
                       const bool is_synthetic_test) {
  {
    // 在真正进入 COLMAP Database 之前先做一次文件可读检查，
    // 这样路径错误时可以更直接地给出中文提示。
    std::ifstream db_file(database_path, std::ios::binary);
    if (!db_file.good()) {
      std::cerr << "无法读取数据库文件: " << database_path << std::endl;
      return false;
    }
  }

  GlomapPipelineOptions options;
  options.min_num_matches = 6;
  options.mapper.min_num_matches = 6;
  options.mapper.min_track_length = 3;
  options.mapper.max_reproj_error = 4.0;
  options.mapper.min_tri_angle = 2.0;
  options.mapper.use_three_stage_optimization = true;
  options.mapper.skip_retriangulation = false;
  options.mapper.bundle_adjustment_options.solver_options.max_num_iterations = 50;

  // 参考 test_vpgo_pipeline.cc 的三阶段粗到细优化参数。
  options.mapper.stage1_options.name = "阶段1：固定内参，粗优化外参";
  options.mapper.stage1_options.max_iterations = 2;
  options.mapper.stage1_options.tri_max_project_error = 16.0;
  options.mapper.stage1_options.filter_max_reproj_error = 16.0;
  options.mapper.stage1_options.refine_focal_length = false;
  options.mapper.stage1_options.refine_principal_point = false;
  options.mapper.stage1_options.refine_extra_params = false;

  options.mapper.stage2_options.name = "阶段2：固定内参，细化外参";
  options.mapper.stage2_options.max_iterations = 2;
  options.mapper.stage2_options.tri_max_project_error = 4.0;
  options.mapper.stage2_options.filter_max_reproj_error = 4.0;
  options.mapper.stage2_options.refine_focal_length = false;
  options.mapper.stage2_options.refine_principal_point = false;
  options.mapper.stage2_options.refine_extra_params = false;

  options.mapper.stage3_options.name = "阶段3：优化内参与外参";
  options.mapper.stage3_options.max_iterations = 3;
  options.mapper.stage3_options.tri_max_project_error = 4.0;
  options.mapper.stage3_options.filter_max_reproj_error = 4.0;
  options.mapper.stage3_options.refine_focal_length = true;
  options.mapper.stage3_options.refine_principal_point = true;
  options.mapper.stage3_options.refine_extra_params = true;

  // 合成三视图样例太小，第三阶段放开内参会导致问题退化，真实数据仍保持完整三阶段。
  if (is_synthetic_test) {
    options.mapper.skip_retriangulation = true;
    options.mapper.stage3_options.refine_focal_length = false;
    options.mapper.stage3_options.refine_principal_point = false;
    options.mapper.stage3_options.refine_extra_params = false;
  }

  std::shared_ptr<Database> database =
      std::make_shared<Database>(database_path);
  std::shared_ptr<ReconstructionManager> reconstruction_manager =
      std::make_shared<ReconstructionManager>();
  // test_glomap 的外部接口只依赖顶层 pipeline，便于后续与正式 CLI 保持一致。
  GlomapPipeline pipeline(options, database, reconstruction_manager);
  if (!pipeline.Run() || reconstruction_manager->Size() == 0) {
    std::cerr << "GLOMAP 管线执行失败，请检查前面的中文阶段日志" << std::endl;
    return false;
  }

  Reconstruction& reconstruction = reconstruction_manager->Get(0);
  reconstruction.WriteText(output_path);
  std::cout << "Registered images: " << reconstruction.NumRegImages()
            << std::endl;
  std::cout << "Points3D: " << reconstruction.NumPoints3D() << std::endl;
  return reconstruction.NumRegImages() >= 3 && reconstruction.NumPoints3D() >= 6;
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage:\n"
      << "  " << argv0
      << " --database_path path/to/database.db --output_path path/to/model\n"
      << "  " << argv0 << " --synthetic_test\n";
}

}  // namespace
}  // namespace colmap

int main(int argc, char** argv) {
  using namespace colmap;

  std::string database_path;
  std::string output_path;
  bool synthetic_test = argc == 1;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--database_path" && i + 1 < argc) {
      database_path = argv[++i];
    } else if (arg == "--output_path" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--synthetic_test") {
      synthetic_test = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (synthetic_test) {
    // 无参数时默认运行合成烟测，便于快速验证编译后的主流程是否可执行。
    database_path = "/tmp/test_glomap_synthetic.db";
    output_path = "/tmp/test_glomap_synthetic_model";
    if (!CreateSyntheticDatabase(database_path)) {
      std::cerr << "Failed to create synthetic database" << std::endl;
      return 1;
    }
  } else if (database_path.empty() || output_path.empty()) {
    PrintUsage(argv[0]);
    return 1;
  }

  CreateDirIfNotExists(output_path, true);
  const bool success =
      RunGlomapPipeline(database_path, output_path, synthetic_test);
  return success ? 0 : 1;
}
