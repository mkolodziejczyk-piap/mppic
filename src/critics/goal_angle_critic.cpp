// Copyright 2022 FastSense, Samsung Research
#include "mppic/critics/goal_angle_critic.hpp"

namespace mppi::critics
{

void GoalAngleCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(power_, "goal_angle_cost_power", 1);
  getParam(weight_, "goal_angle_cost_weight", 5.0);
  getParam(
    threshold_to_consider_goal_angle_,
    "threshold_to_consider_goal_angle", 0.20);
  RCLCPP_INFO(
    logger_,
    "GoalAngleCritic instantiated with %d power, %f weight, and %f "
    "angular threshold.",
    power_, weight_, threshold_to_consider_goal_angle_);
}

void GoalAngleCritic::score(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const models::State & /*state*/,
  const torch::Tensor & trajectories,
  const torch::Tensor & path,
  torch::Tensor & costs,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  torch::Tensor tensor_pose = torch::Tensor({
    static_cast<double>(robot_pose.pose.position.x),
    static_cast<double>(robot_pose.pose.position.y)});

  auto path_points = path.index({-1, Slice(0, 2)});

  double points_to_goal_dists = xt::norm_l2(tensor_pose - path_points, {0})();

  if (points_to_goal_dists < threshold_to_consider_goal_angle_) {
    auto yaws = trajectories.index({"...", 2});
    auto goal_yaw = path.index({-1, 2});

    costs += torch::pow(
      torch::mean(torch::abs(utils::shortest_angular_distance(yaws, goal_yaw)), {1}) *
      weight_, power_);
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::GoalAngleCritic,
  mppi::critics::CriticFunction)
