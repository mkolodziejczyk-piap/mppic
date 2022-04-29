// Copyright 2022 FastSense, Samsung Research
#ifndef MPPIC__MOTION_MODELS_HPP_
#define MPPIC__MOTION_MODELS_HPP_

#include <cstdint>

#include "mppic/models/state.hpp"

namespace mppi
{

class MotionModel
{
public:
  MotionModel() = default;
  virtual ~MotionModel() = default;

  /**
   * @brief Predict velocities for given trajectories the next time step
   *
   * @param state for given time_step, tensor of shape
   * [batch_size, ...] where last dim could be 5 or 7 depending on motion model used
   *
   * @return predicted velocities of the robot: tensor of shape [batch_size, ... ]
   * where last dim could be 2 or 3 depending on motion model used
   */
  virtual torch::Tensor predict(
    const torch::Tensor & state, const models::StateIdxes & idx)
  {
    return state.index({"...", Slice(idx.cbegin(), idx.cend())});
  }

  virtual bool isHolonomic() const = 0;
};

class DiffDriveMotionModel : public MotionModel
{
public:
  bool isHolonomic() const override {return false;}
};

class OmniMotionModel : public MotionModel
{
public:
  bool isHolonomic() const override {return true;}
};

class AckermannMotionModel : public MotionModel
{
public:
  torch::Tensor predict(
    const torch::Tensor & /*state*/, const models::StateIdxes & /*idx*/) override
  {
    throw std::runtime_error("Ackermann motion model not yet implemented");
  }

  bool isHolonomic() const override {return false;}
};

}  // namespace mppi

#endif  // MPPIC__MOTION_MODELS_HPP_
