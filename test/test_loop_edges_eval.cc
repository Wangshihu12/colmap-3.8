#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "base/pose.h"
#include "base/reconstruction.h"
#include "util/misc.h"
#include "util/string.h"

using namespace colmap;

namespace {

struct LoopEdgeMeasurement {
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Eigen::Quaterniond q_ij = Eigen::Quaterniond::Identity();
  Eigen::Vector3d t_ij = Eigen::Vector3d::Zero();
  bool t_is_unit = true;
};

bool IsNumericToken(const std::string& token) {
  char* end_ptr = nullptr;
  std::strtod(token.c_str(), &end_ptr);
  return end_ptr != token.c_str() && *end_ptr == '\0';
}

bool ParseLoopEdgeLine(const std::string& line,
                       LoopEdgeMeasurement* measurement) {
  std::string content = line;
  const auto comment_pos = content.find('#');
  if (comment_pos != std::string::npos) {
    content = content.substr(0, comment_pos);
  }
  StringTrim(&content);
  if (content.empty()) {
    return false;
  }

  std::istringstream iss(content);
  std::string first;
  if (!(iss >> first)) {
    return false;
  }

  bool has_type = !IsNumericToken(first);
  std::string type_token;
  int64_t image_id1 = -1;
  int64_t image_id2 = -1;
  double qw = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
  if (has_type) {
    type_token = first;
    StringToLower(&type_token);
    if (!(iss >> image_id1 >> image_id2 >> qw >> qx >> qy >> qz >> tx >> ty >>
          tz)) {
      return false;
    }
  } else {
    image_id1 = std::stoll(first);
    if (!(iss >> image_id2 >> qw >> qx >> qy >> qz >> tx >> ty >> tz)) {
      return false;
    }
    type_token = "loop";
  }

  if (type_token == "odom" || type_token == "rig") {
    return false;
  }

  double t_is_unit_value = 1.0;
  iss >> t_is_unit_value;  // optional in new format

  measurement->image_id1 = static_cast<image_t>(image_id1);
  measurement->image_id2 = static_cast<image_t>(image_id2);
  measurement->q_ij = Eigen::Quaterniond(qw, qx, qy, qz);
  measurement->q_ij.normalize();
  measurement->t_is_unit = std::abs(t_is_unit_value) > 0.5;
  measurement->t_ij = Eigen::Vector3d(tx, ty, tz);
  if (measurement->t_is_unit) {
    const double t_norm = measurement->t_ij.norm();
    if (t_norm > 1e-12) {
      measurement->t_ij /= t_norm;
    }
  }
  return true;
}

double ClampDot(const double value) {
  return std::max(-1.0, std::min(1.0, value));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "Usage: test_loop_edges_eval sparse_path loop_edges_path "
                 "output_dir\n";
    return -1;
  }

  const std::string sparse_path = argv[1];
  const std::string loop_edges_path = argv[2];
  const std::string output_dir = argv[3];

  CreateDirIfNotExists(output_dir);
  const std::string gt_path = JoinPaths(output_dir, "loop_edges_gt.txt");
  const std::string err_path = JoinPaths(output_dir, "loop_edges_error.txt");
  const std::string large_err_path = JoinPaths(output_dir, "loop_edges_large_error.txt");

  Reconstruction reconstruction;
  reconstruction.Read(sparse_path);

  std::ifstream loop_edges_file(loop_edges_path);
  if (!loop_edges_file.is_open()) {
    std::cout << "ERROR: Could not open loop_edges file at "
              << loop_edges_path << std::endl;
    return -1;
  }

  std::ofstream gt_file(gt_path);
  if (!gt_file.is_open()) {
    std::cout << "ERROR: Could not open ground truth output at " << gt_path
              << std::endl;
    return -1;
  }

  std::ofstream err_file(err_path);
  if (!err_file.is_open()) {
    std::cout << "ERROR: Could not open error output at " << err_path
              << std::endl;
    return -1;
  }

  std::ofstream large_err_file(large_err_path);
  if (!large_err_file.is_open()) {
    std::cout << "ERROR: Could not open large error output at " << large_err_path
              << std::endl;
    return -1;
  }

  std::string line;
  size_t num_written = 0;
  size_t num_skipped = 0;
  while (std::getline(loop_edges_file, line)) {
    LoopEdgeMeasurement measurement;
    if (!ParseLoopEdgeLine(line, &measurement)) {
      continue;
    }

    if (!reconstruction.ExistsImage(measurement.image_id1) ||
        !reconstruction.ExistsImage(measurement.image_id2)) {
      num_skipped += 1;
      continue;
    }

    const Image& image1 = reconstruction.Image(measurement.image_id1);
    const Image& image2 = reconstruction.Image(measurement.image_id2);
    if (!image1.IsRegistered() || !image2.IsRegistered()) {
      num_skipped += 1;
      continue;
    }

    Eigen::Vector4d qvec_ij;
    Eigen::Vector3d tvec_ij;
    ComputeRelativePose(image1.Qvec(), image1.Tvec(), image2.Qvec(),
                        image2.Tvec(), &qvec_ij, &tvec_ij);
    const Eigen::Vector4d qvec_ij_norm = NormalizeQuaternion(qvec_ij);
    Eigen::Quaterniond q_ij_gt(qvec_ij_norm(0), qvec_ij_norm(1),
                               qvec_ij_norm(2), qvec_ij_norm(3));

    // // 增加一个简单的过滤,如果两张图像的相机中心距离超过10m,则跳过该边
    // // 注意: Tvec()不是相机中心位置,相机中心应使用ProjectionCenter()
    // const double dist = (image1.ProjectionCenter() - image2.ProjectionCenter()).norm();
    // if (dist > 10.0) {
    //   num_skipped += 1;
    //   continue;
    // }

    gt_file << measurement.image_id1 << " " << measurement.image_id2 << " "
            << q_ij_gt.w() << " " << q_ij_gt.x() << " " << q_ij_gt.y() << " "
            << q_ij_gt.z() << " " << tvec_ij(0) << " " << tvec_ij(1)
            << " " << tvec_ij(2) << "\n";

    double rot_err_deg = -1.0;
    double trans_err = -1.0;

    const double q_norm = measurement.q_ij.norm();
    if (q_norm > 1e-12) {
      Eigen::Quaterniond q_meas = measurement.q_ij;
      q_meas.normalize();
      Eigen::Quaterniond q_err = q_meas.conjugate() * q_ij_gt;
      const double w = std::abs(q_err.w());
      const double w_clamped = ClampDot(w);
      rot_err_deg = 2.0 * std::acos(w_clamped) * 180.0 / M_PI;
    }

    if (!measurement.t_is_unit) {
      if (tvec_ij.squaredNorm() > 1e-24 &&
          measurement.t_ij.squaredNorm() > 1e-24) {
        trans_err = (measurement.t_ij - tvec_ij).norm();
      }
    }

    err_file << std::fixed << std::setprecision(6) << measurement.image_id1 << " " << measurement.image_id2 << " "
             << rot_err_deg << " " << trans_err << "\n";

    // 如果旋转角度误差大于5度或平移误差大于1米,则记录到大误差文件中
    if (rot_err_deg > 5.0 || trans_err > 1.0) {
      large_err_file << measurement.image_id1 << " " << measurement.image_id2 << " "
                     << measurement.q_ij.w() << " " << measurement.q_ij.x() << " " << measurement.q_ij.y() << " "
                     << measurement.q_ij.z() << " "
                     << measurement.t_ij(0) << " " << measurement.t_ij(1) << " " << measurement.t_ij(2) << "\n";
    }

    num_written += 1;
  }

  std::cout << "Wrote ground truth edges: " << num_written << "\n";
  if (num_skipped > 0) {
    std::cout << "Skipped edges (missing/unregistered images): "
              << num_skipped << "\n";
  }
  std::cout << "Ground truth file: " << gt_path << "\n";
  std::cout << "Error file: " << err_path << "\n";
  return 0;
}
