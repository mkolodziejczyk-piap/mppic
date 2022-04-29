// Copyright 2022 FastSense, Samsung Research
#pragma once

#include "mppic/critic_function.hpp"
#include "mppic/models/state.hpp"
#include "mppic/utils.hpp"

namespace mppi::critics
{

class GoalAngleCritic : public CriticFunction
{
public:
  void initialize() override;

  /**
   * @brief Evaluate cost related to robot orientation at goal pose
   * (considered only if robot near last goal in current plan)
   *
   * @param costs [out] add goal angle cost values to this tensor
   */
  void score(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const models::State & state, const torch::Tensor & trajectories,
    const torch::Tensor & path, torch::Tensor & costs,
    nav2_core::GoalChecker * goal_checker) override;

protected:
  double threshold_to_consider_goal_angle_{0};
  unsigned int power_{0};
  double weight_{0};
};

}  // namespace mppi::critics
