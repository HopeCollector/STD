#include <nav_msgs/Odometry.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <yaml-cpp/yaml.h>

#include <expr-utils/data_loader.hh>
#include <filesystem>

#include "cli11.hh"
#include "include/STDesc.h"

namespace fs = std::filesystem;

std::string dataset_type{""};
std::string lidar_path{""};
std::string pose_path{""};
std::string calib_path{""};
std::string config_path{""};
std::string output_path{""};

int parse_param(int argc, char **argv) {
  CLI::App app{"STDesc expriement"};
  app.add_option("-c,--config", config_path, "Path to the config file")
      ->required();
  CLI11_PARSE(app, argc, argv);
  return 0;
}

ConfigSetting load_config() {
  ConfigSetting setting;
  YAML::Node config = YAML::LoadFile(config_path);

  // dataset
  dataset_type = config["dataset"].as<std::string>();
  lidar_path = config["lidar"].as<std::string>();
  pose_path = config["pose"].as<std::string>();
  calib_path = config["calib"].as<std::string>();
  output_path = config["result"].as<std::string>();

  // verify the output path
  fs::path output_parent = fs::path(output_path).parent_path();
  if (!fs::exists(output_parent)) {
    // check if the parent directory exists
    throw std::runtime_error(output_parent.string() + " does not exist");
  }
  if (fs::is_directory(output_path)) {
    // check if the output path is a directory
    if (output_path.back() != '/') {
      output_path += "/";
    }
    // append the current time to the output path
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << output_path << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    output_path = oss.str();
  }

  // pre-process
  setting.ds_size_ = config["ds_size"].as<double>(0.5);
  setting.maximum_corner_num_ = config["maximum_corner_num"].as<int>(100);
  std::cout << "pre-process" << std::endl;

  // key points
  setting.plane_merge_normal_thre_ =
      config["plane_merge_normal_thre"].as<double>(0.1);
  setting.plane_detection_thre_ =
      config["plane_detection_thre"].as<double>(0.01);
  setting.voxel_size_ = config["voxel_size"].as<double>(2.0);
  setting.voxel_init_num_ = config["voxel_init_num"].as<int>(10);
  setting.proj_image_resolution_ =
      config["proj_image_resolution"].as<double>(0.5);
  setting.proj_dis_min_ = config["proj_dis_min"].as<double>(0);
  setting.proj_dis_max_ = config["proj_dis_max"].as<double>(2);
  setting.corner_thre_ = config["corner_thre"].as<double>(10);
  std::cout << "key points" << std::endl;

  // std descriptor
  setting.descriptor_near_num_ = config["descriptor_near_num"].as<int>(10);
  setting.descriptor_min_len_ = config["descriptor_min_len"].as<double>(2);
  setting.descriptor_max_len_ = config["descriptor_max_len"].as<double>(50);
  setting.non_max_suppression_radius_ =
      config["non_max_suppression_radius"].as<double>(2.0);
  setting.std_side_resolution_ = config["std_side_resolution"].as<double>(0.2);
  std::cout << "std descriptor" << std::endl;

  // candidate search
  setting.skip_near_num_ = config["skip_near_num"].as<int>(50);
  setting.candidate_num_ = config["candidate_num"].as<int>(50);
  setting.sub_frame_num_ = config["sub_frame_num"].as<int>(10);
  setting.rough_dis_threshold_ = config["rough_dis_threshold"].as<double>(0.01);
  setting.vertex_diff_threshold_ =
      config["vertex_diff_threshold"].as<double>(0.5);
  setting.icp_threshold_ = config["icp_threshold"].as<double>(0.5);
  setting.normal_threshold_ = config["normal_threshold"].as<double>(0.2);
  setting.dis_threshold_ = config["dis_threshold"].as<double>(0.5);
  std::cout << "candidate search" << std::endl;

  std::cout << "Successfully loaded parameters:" << std::endl;
  std::cout << "----------------Main Parameters-------------------"
            << std::endl;
  std::cout << "voxel size: " << setting.voxel_size_ << std::endl;
  std::cout << "loop detection threshold: " << setting.icp_threshold_
            << std::endl;
  std::cout << "sub-frame number: " << setting.sub_frame_num_ << std::endl;
  std::cout << "candidate number: " << setting.candidate_num_ << std::endl;
  std::cout << "maximum corners size: " << setting.maximum_corner_num_
            << std::endl;

  return setting;
}

struct LoopResult {
  size_t key_frame_id;
  size_t loop_frame_id;
  double score;
  double iou;
  Eigen::Vector3d center;

  LoopResult(size_t key_frame_id, size_t loop_frame_id, double score)
      : key_frame_id(key_frame_id),
        loop_frame_id(loop_frame_id),
        score(score),
        center() {}

  friend std::ostream &operator<<(std::ostream &os, const LoopResult &res) {
    os << res.key_frame_id << "," << res.loop_frame_id << "," << res.score
       << "," << res.iou << "," << res.center.x() << "," << res.center.y()
       << "," << res.center.z();
    return os;
  }
};

double iou(pcl::PointCloud<pcl::PointXYZI>::ConstPtr cld1,
           pcl::PointCloud<pcl::PointXYZI>::ConstPtr cld2) {
  pcl::KdTreeFLANN<pcl::PointXYZI> tree;
  tree.setInputCloud(cld1);
  std::vector<bool> marks1(cld1->size(), false);
  std::vector<bool> marks2(cld2->size(), false);

  // search the near points in 0.5m
  for (size_t i = 0; i < cld2->size(); i++) {
    const auto &p = cld2->at(i);
    std::vector<int> indices;
    std::vector<float> distances;
    tree.radiusSearch(p, 0.5, indices, distances);
    if (indices.empty()) {
      continue;
    }
    // mark the indices
    for (const auto &idx : indices) {
      marks1[idx] = true;
    }
    marks2[i] = true;
  }

  // calculate the iou, no need to minus the intersection, because this is point
  // num, not volume
  size_t inter = std::count(marks1.begin(), marks1.end(), true) +
                 std::count(marks2.begin(), marks2.end(), true);
  return double(inter) / double(cld1->size() + cld2->size());
}

int main(int argc, char **argv) {
  // parse parameters
  parse_param(argc, argv);

  // create std manager
  auto setting = load_config();
  STDescManager *std_manager = new STDescManager(setting);

  // load data
  auto loader =
      utils::create_loader(dataset_type, lidar_path, pose_path, calib_path);

  size_t cloudInd = 0;
  size_t keyCloudInd = 0;
  pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
      new pcl::PointCloud<pcl::PointXYZI>());

  std::vector<double> descriptor_time;
  std::vector<double> querying_time;
  std::vector<double> update_time;
  std::vector<LoopResult> reses;
  int triggle_loop_num = 0;
  while (true) {
    auto current_cloud = loader->next(true);
    if (current_cloud) {
      down_sampling_voxel(*current_cloud, setting.ds_size_);
      *temp_cloud += *current_cloud;
    }

    // check if keyframe
    if ((cloudInd % setting.sub_frame_num_ == 0 && cloudInd != 0) ||
        (!current_cloud && temp_cloud->size() > 0)) {
      std::cout << "Key Frame id:" << keyCloudInd
                << ", cloud size: " << temp_cloud->size() << std::endl;
      // step1. Descriptor Extraction
      auto t_descriptor_begin = std::chrono::high_resolution_clock::now();
      std::vector<STDesc> stds_vec;
      std_manager->GenerateSTDescs(temp_cloud, stds_vec);
      auto t_descriptor_end = std::chrono::high_resolution_clock::now();
      descriptor_time.push_back(time_inc(t_descriptor_end, t_descriptor_begin));
      // step2. Searching Loop
      auto t_query_begin = std::chrono::high_resolution_clock::now();
      std::pair<int, double> search_result(-1, 0);
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
      loop_transform.first << 0, 0, 0;
      loop_transform.second = Eigen::Matrix3d::Identity();
      std::vector<std::pair<STDesc, STDesc>> loop_std_pair;
      if (keyCloudInd > setting.skip_near_num_) {
        std_manager->SearchLoop(stds_vec, search_result, loop_transform,
                                loop_std_pair);
        if (search_result.second > 1e-3) {
          reses.emplace_back(keyCloudInd, size_t(search_result.first),
                             search_result.second);
          auto &res = reses.back();
          auto cld = std_manager->key_cloud_vec_[search_result.first];
          for (const auto &p : cld->points) {
            res.center += p.getVector3fMap().cast<double>();
          }
          for (const auto &p : temp_cloud->points) {
            res.center += p.getVector3fMap().cast<double>();
          }
          res.center /= cld->size() + temp_cloud->size();
          res.iou = iou(cld, temp_cloud);
        }
      }
      auto t_query_end = std::chrono::high_resolution_clock::now();
      querying_time.push_back(time_inc(t_query_end, t_query_begin));

      // step3. Add descriptors to the database
      auto t_map_update_begin = std::chrono::high_resolution_clock::now();
      std_manager->AddSTDescs(stds_vec);
      auto t_map_update_end = std::chrono::high_resolution_clock::now();
      update_time.push_back(time_inc(t_map_update_end, t_map_update_begin));
      std::cout << "[Time] descriptor extraction: "
                << time_inc(t_descriptor_end, t_descriptor_begin) << "ms, "
                << "query: " << time_inc(t_query_end, t_query_begin) << "ms, "
                << "update map:"
                << time_inc(t_map_update_end, t_map_update_begin) << "ms"
                << std::endl;
      std::cout << std::endl;

      pcl::PointCloud<pcl::PointXYZI> save_key_cloud;
      save_key_cloud = *temp_cloud;

      std_manager->key_cloud_vec_.push_back(save_key_cloud.makeShared());

      if (search_result.first > 0) {
        triggle_loop_num++;
      }
      temp_cloud->clear();
      keyCloudInd++;
    }
    cloudInd++;

    if (!current_cloud) {
      break;
    }
  }
  double mean_descriptor_time =
      std::accumulate(descriptor_time.begin(), descriptor_time.end(), 0) * 1.0 /
      descriptor_time.size();
  double mean_query_time =
      std::accumulate(querying_time.begin(), querying_time.end(), 0) * 1.0 /
      querying_time.size();
  double mean_update_time =
      std::accumulate(update_time.begin(), update_time.end(), 0) * 1.0 /
      update_time.size();
  std::cout << "Total key frame number:" << keyCloudInd
            << ", loop number:" << triggle_loop_num << std::endl;
  std::cout << "Mean time for descriptor extraction: " << mean_descriptor_time
            << "ms, query: " << mean_query_time
            << "ms, update: " << mean_update_time << "ms, total: "
            << mean_descriptor_time + mean_query_time + mean_update_time << "ms"
            << std::endl;

  std::ofstream resf(output_path);
  if (resf.is_open()) {
    std::sort(reses.begin(), reses.end(),
              [](const LoopResult &a, const LoopResult &b) {
                return a.score > b.score;
              });
    for (const auto &res : reses) {
      resf << res << std::endl;
    }
    resf.close();
  } else {
    std::cerr << "Unable to open file to save pairs" << std::endl;
  }
  return 0;
}