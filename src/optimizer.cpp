// Copyright 2022 FastSense, Samsung Research
#include "mppic/optimizer.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nav2_core/exceptions.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"

using namespace torch::indexing;

namespace mppi
{

void Optimizer::initialize(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
  ParametersHandler * param_handler)
{
  parent_ = parent;
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  parameters_handler_ = param_handler;

  auto node = parent_.lock();
  logger_ = node->get_logger();

  getParams();
  setOffset();

  critic_manager_.on_configure(parent_, name_, costmap_ros_, parameters_handler_);

  reset();
}

void Optimizer::getParams()
{
  std::string motion_model_name;

  auto & s = settings_;
  auto getParam = parameters_handler_->getParamGetter(name_);
  auto getParentParam = parameters_handler_->getParamGetter("");
  getParam(s.model_dt_, "model_dt", 0.1);
  getParam(s.time_steps_, "time_steps", 15);
  getParam(s.batch_size_, "batch_size", 400);
  getParam(s.iteration_count_, "iteration_count", 2);
  getParam(s.temperature_, "temperature", 0.25);
  getParam(s.base_constraints_.vx, "vx_max", 0.5);
  getParam(s.base_constraints_.vy, "vy_max", 0.5);
  getParam(s.base_constraints_.wz, "wz_max", 1.3);
  getParam(s.sampling_std_.vx, "vx_std", 0.2);
  getParam(s.sampling_std_.vy, "vy_std", 0.2);
  getParam(s.sampling_std_.wz, "wz_std", 1.0);
  getParam(motion_model_name, "motion_model", std::string("DiffDrive"));
  getParentParam(controller_frequency_, "controller_frequency", 0.0);

  s.constraints_ = s.base_constraints_;
  setMotionModel(motion_model_name);
  parameters_handler_->addPostCallback([this]() {reset();});
}

void Optimizer::setOffset()
{
  const double controller_period = 1.0 / controller_frequency_;
  constexpr double eps = 1e-6;

  if (controller_period < settings_.model_dt_) {
    RCLCPP_WARN(
      logger_,
      "Controller period is less then model dt, consider setting it equal");
    settings_.control_sequence_shift_offset_ = 0;
  } else if (abs(controller_period - settings_.model_dt_) < eps) {
    RCLCPP_INFO(
      logger_,
      "Controller period is equal to model dt. Control seuqence "
      "shifting is ON");
    settings_.control_sequence_shift_offset_ = 1;
  } else {
    throw std::runtime_error(
            "Controller period more then model dt, set it equal to model dt");
  }
}

void Optimizer::setSpeedLimit(double speed_limit, bool percentage)
{
  auto & s = settings_;
  if (speed_limit == nav2_costmap_2d::NO_SPEED_LIMIT) {
    s.constraints_.vx = s.base_constraints_.vx;
    s.constraints_.vy = s.base_constraints_.vy;
    s.constraints_.wz = s.base_constraints_.wz;
  } else {
    if (percentage) {
      // Speed limit is expressed in % from maximum speed of robot
      double ratio = speed_limit / 100.0;
      s.constraints_.vx = s.base_constraints_.vx * ratio;
      s.constraints_.vy = s.base_constraints_.vy * ratio;
      s.constraints_.wz = s.base_constraints_.wz * ratio;
    } else {
      // Speed limit is expressed in absolute value
      double ratio = speed_limit / s.base_constraints_.vx;
      s.constraints_.vx = speed_limit;
      s.constraints_.vy = s.base_constraints_.vx * ratio;
      s.constraints_.wz = s.base_constraints_.wz * ratio;
    }
  }
}

void Optimizer::reset()
{
  state_.reset(settings_.batch_size_, settings_.time_steps_);
  state_.getTimeIntervals() = settings_.model_dt_;
  control_sequence_.reset(settings_.time_steps_);

  RCLCPP_INFO(logger_, "Optimizer reset");
}

geometry_msgs::msg::TwistStamped Optimizer::evalControl(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan, nav2_core::GoalChecker * goal_checker)
{
  for (size_t i = 0; i < settings_.iteration_count_; ++i) {
    generated_trajectories_ =
      generateNoisedTrajectories(robot_pose, robot_speed);
    auto && costs = critic_manager_.evalTrajectoriesScores(
      state_, generated_trajectories_, plan, robot_pose, goal_checker);
    updateControlSequence(costs);
  }

  auto control = getControlFromSequenceAsTwist(
    settings_.control_sequence_shift_offset_,
    plan.header.stamp);

  shiftControlSequence();
  return control;
}

void Optimizer::shiftControlSequence()
{
  if (settings_.control_sequence_shift_offset_ == 0) {
    return;
  }

  using namespace xt::placeholders;  // NOLINT
  xt::view(
    control_sequence_.data,
    xt::range(_, -settings_.control_sequence_shift_offset_), xt::all()) =
    xt::view(
    control_sequence_.data,
    xt::range(settings_.control_sequence_shift_offset_, _),
    xt::all());
}

torch::Tensor Optimizer::generateNoisedTrajectories(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed)
{
  state_.getControls() = generateNoisedControls();
  applyControlConstraints();
  updateStateVelocities(state_, robot_speed);
  return integrateStateVelocities(state_, robot_pose);
}

torch::Tensor Optimizer::generateNoisedControls() const
{
  auto & s = settings_;
  auto vx_noises = torch::randn(
    {s.batch_size_, s.time_steps_, 1U}) * s.sampling_std_.vx;
  auto wz_noises = torch::randn(
    {s.batch_size_, s.time_steps_, 1U}) * s.sampling_std_.wz;

  if (isHolonomic()) {
    auto vy_noises = torch::randn(
      {s.batch_size_, s.time_steps_, 1U}, 0.0, s.sampling_std_.vy);
    return control_sequence_.data +
           torch::cat({vx_noises, vy_noises, wz_noises}, -1);
  }

  return control_sequence_.data +
         torch::cat({vx_noises, wz_noises}, -1);

  //TODO allocate control_sequence_ at once
  /*


  */
}

bool Optimizer::isHolonomic() const {return motion_model_->isHolonomic();}

void Optimizer::applyControlConstraints()
{
  auto vx = state_.getControlVelocitiesVX();
  auto wz = state_.getControlVelocitiesWZ();
  auto & s = settings_;

  if (isHolonomic()) {
    auto vy = state_.getControlVelocitiesVY();
    vy = torch::clip(vy, -s.constraints_.vy, s.constraints_.vy);
  }

  vx = torch::clip(vx, -s.constraints_.vx, s.constraints_.vx);
  wz = torch::clip(wz, -s.constraints_.wz, s.constraints_.wz);
  //TODO one clip on whole state_ ?
}

void Optimizer::updateStateVelocities(
  models::State & state, const geometry_msgs::msg::Twist & robot_speed) const
{
  updateInitialStateVelocities(state, robot_speed);
  propagateStateVelocitiesFromInitials(state);
}

void Optimizer::updateInitialStateVelocities(
  models::State & state, const geometry_msgs::msg::Twist & robot_speed) const
{
  state.getVelocitiesVX().index_put_({0}, robot_speed.linear.x);
  state.getVelocitiesWZ().index_put_({0}, robot_speed.angular.z);

  if (isHolonomic()) {
    state.getVelocitiesVY().index_put_({0}, robot_speed.linear.y);
  }
}

// this is trajectory rollout
void Optimizer::propagateStateVelocitiesFromInitials(
  models::State & state) const
{
  for (size_t i = 0; i < settings_.time_steps_ - 1; i++) {
    auto curr_state = state.data.index({"...", i});
    auto next_velocities = state.data.index({"...", i + 1, torch::Slice(state.idx.vbegin(), state.idx.vend())});

    next_velocities = motion_model_->predict(curr_state, state.idx); //TODO where is real predict?
  }
}

torch::Tensor Optimizer::evalTrajectoryFromControlSequence(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed) const
{
  models::State state;
  state.idx.setLayout(motion_model_->isHolonomic());
  state.reset(1U, settings_.time_steps_);
  state.getControls() = control_sequence_.data;
  state.getTimeIntervals() = settings_.model_dt_;

  updateStateVelocities(state, robot_speed);
  return xt::squeeze(integrateStateVelocities(state, robot_pose));
}

torch::Tensor Optimizer::integrateStateVelocities(
  const models::State & state,
  const geometry_msgs::msg::PoseStamped & pose) const
{

  auto w = state.getVelocitiesWZ();
  double initial_yaw = tf2::getYaw(pose.pose.orientation);
  torch::Tensor yaw =
    torch::cumsum(w * settings_.model_dt_, 1) + initial_yaw;
  torch::Tensor yaw_offseted = yaw;

  yaw_offseted.index_put_({"...", Slice(1, None)}) =
    yaw.index({"...", Slice(None, -1)}); //TODO roll() ?
  yaw_offseted.index_put_({"...", 0}) = initial_yaw;

  auto yaw_cos = torch::cos(yaw_offseted);
  auto yaw_sin = torch::sin(yaw_offseted);

  auto vx = state.getVelocitiesVX();
  auto dx = vx * yaw_cos;
  auto dy = vx * yaw_sin;

  if (isHolonomic()) {
    auto vy = state.getVelocitiesVY();
    dx = dx - vy * yaw_sin;
    dy = dy + vy * yaw_cos;
  }

  auto x = pose.pose.position.x + torch::cumsum(dx * settings_.model_dt_, 1);
  auto y = pose.pose.position.y + torch::cumsum(dy * settings_.model_dt_, 1);

  return torch::cat(
    {x, y, yaw}, //TODO newaxis?
    }); //TODO axis?
}

void Optimizer::updateControlSequence(const xt::xtensor<double, 1> & costs)
{
  using xt::evaluation_strategy::immediate;

  auto && costs_normalized = costs - xt::amin(costs, immediate);
  auto exponents =
    xt::eval(xt::exp(-1 / settings_.temperature_ * costs_normalized));
  auto softmaxes = exponents / xt::sum(exponents, immediate);
  auto softmaxes_expanded =
    xt::view(softmaxes, xt::all(), xt::newaxis(), xt::newaxis());

  control_sequence_.data =
    xt::sum(state_.getControls() * softmaxes_expanded, 0);
}

auto Optimizer::getControlFromSequence(const unsigned int offset)
{
  return xt::view(control_sequence_.data, offset);
}

geometry_msgs::msg::TwistStamped Optimizer::getControlFromSequenceAsTwist(
  const unsigned int offset, const builtin_interfaces::msg::Time & stamp)
{
  return utils::toTwistStamped(
    getControlFromSequence(offset),
    control_sequence_.idx, isHolonomic(), stamp,
    costmap_ros_->getBaseFrameID());
}

void Optimizer::setMotionModel(const std::string & model)
{
  if (model == "DiffDrive") {
    motion_model_ = std::make_unique<DiffDriveMotionModel>();
  } else if (model == "Omni") {
    motion_model_ = std::make_unique<OmniMotionModel>();
  } else if (model == "Ackermann") {
    motion_model_ = std::make_unique<AckermannMotionModel>();
  } else {
    throw std::runtime_error(
            std::string(
              "Model %s is not valid! Valid options are DiffDrive, Omni, "
              "or Ackermann",
              model.c_str()));
  }

  state_.idx.setLayout(motion_model_->isHolonomic());
  control_sequence_.idx.setLayout(motion_model_->isHolonomic());
}

torch::Tensor & Optimizer::getGeneratedTrajectories()
{
  return generated_trajectories_;
}

}  // namespace mppi
