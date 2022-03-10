#pragma once

#include <xtensor/xtensor.hpp>

#include "mppic/critics/critic_function.hpp"
#include "mppic/utils.hpp"

namespace mppi::optimization {

template <typename T>
class ReferenceTrajectoryCritic : public CriticFunction<T>
{
public:
  using CriticFunction<T>::parent_;
  using CriticFunction<T>::node_name_;

  void getParams() override
  {
    auto node = parent_.lock();
    auto getParam = utils::getParamGetter(node, node_name_);
    getParam(power_, "reference_cost_power", 1);
    getParam(weight_, "reference_cost_weight", 15);
  }

  /**
   * @brief Evaluate cost related to trajectories path alignment
   *
   * @param costs [out] add reference cost values to this tensor
   */
  virtual void score(
    const geometry_msgs::msg::PoseStamped & robot_pose, const xt::xtensor<T, 3> & trajectories,
    const xt::xtensor<T, 2> & path, xt::xtensor<T, 1> & costs) override
  {
    (void)robot_pose;

    using xt::evaluation_strategy::immediate;

    xt::xtensor<T, 3> dists_path_to_trajectories =
      utils::distPointsToLineSegments2D(path, trajectories);

    xt::xtensor<T, 1> cost =
      xt::mean(xt::amin(std::move(dists_path_to_trajectories), 1, immediate), 1, immediate);

    costs += xt::pow(std::move(cost) * weight_, power_);
  }

protected:
  unsigned int power_{0};
  double weight_{0};
};

} // namespace mppi::optimization