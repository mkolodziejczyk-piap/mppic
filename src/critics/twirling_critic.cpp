// Copyright 2022 FastSense, Samsung Research
#include "mppic/critics/twirling_critic.hpp"

namespace mppi::critics
{

void TwirlingCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(power_, "twirling_cost_power", 1);
  getParam(weight_, "twirling_cost_weight", 10.0);

  RCLCPP_INFO(
    logger_, "TwirlingCritic instantiated with %d power and %f weight.", power_, weight_);
}

void TwirlingCritic::score(
  const geometry_msgs::msg::PoseStamped & /*robot_pose*/, const models::State & state,
  const torch::Tensor & /*trajectories*/,
  const torch::Tensor & /*path*/, torch::Tensor & costs,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  auto wz = torch::abs(state.getVelocitiesWZ());
  costs += torch::pow(torch::mean(wz, {1}) * weight_, power_);
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mppi::critics::TwirlingCritic, mppi::critics::CriticFunction)
