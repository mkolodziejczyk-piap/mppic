// Copyright 2022 FastSense, Samsung Research
#include "mppic/critics/goal_critic.hpp"

namespace mppi::critics
{

void GoalCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(power_, "goal_cost_power", 1);
  getParam(weight_, "goal_cost_weight", 5.0);
  RCLCPP_INFO(
    logger_, "GoalCritic instantiated with %d power and %f weight.",
    power_, weight_);
}

void GoalCritic::score(
  const geometry_msgs::msg::PoseStamped & /*robot_pose*/,
  const models::State & /*state*/,
  const af::array & trajectories, // (num_trajectories, num_states, 3)
  const af::array & path,
  af::array & costs,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  const auto goal_points = path(-1, af::seq(0, 2));

  auto trajectories_end =
    trajectories(af::span, -1, af::seq(0, 2));

  auto dim = trajectories_end.dimension() - 1;

  af::array dists_trajectories_end_to_goal(trajectories.dims(0), 2);
  gfor (seq i, dists_trajectories_end_to_goal.dims(0)) {
    dists_trajectories_end_to_goal(i, 2) = af::norm(trajectories_end.row(i) - goal_points);
  } 

  costs += af::pow(std::move(dists_trajectories_end_to_goal) * weight_, power_);
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mppi::critics::GoalCritic, mppi::critics::CriticFunction)
