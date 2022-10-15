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
  state_.dts = af::constant(settings_.model_dt_, settings_.batch_size_, settings_.time_steps_);
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

  control_sequence_.data = af::shift(control_sequence_.data, 0, 0, -settings_.control_sequence_shift_offset_);
}

af::array Optimizer::generateNoisedTrajectories(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed)
{
  state_.controls = generateNoisedControls();
  applyControlConstraints();
  updateStateVelocities(state_, robot_speed);
  return integrateStateVelocities(state_, robot_pose);
}

af::array Optimizer::generateNoisedControls() const
{
  auto & s = settings_;

  af::array vw_noises;
  
  if (isHolonomic()) {
    //TODO class variable
    double sampling_std_data[] = {s.sampling_std_.vx, s.sampling_std_.vy, s.sampling_std_.wz};
    af::array sampling_std = af::array(3, sampling_std_data);
    vw_noises = af::randn(
      3, s.batch_size_ * s.time_steps_);
    gfor (af::seq i, vw_noises.dims(1)) {
      // element-wise multiplication, https://arrayfire.org/docs/group__arith__func__mul.htm#ga8317504ec8b9c15d29b27cc77039cb69
      vw_noises(af::span, i) *= sampling_std;
    }
  } else {
    double sampling_std_data[] = {s.sampling_std_.vx, s.sampling_std_.wz};
    af::array sampling_std = af::array(2, sampling_std_data);
    vw_noises = af::randn(
      2, s.batch_size_ * s.time_steps_);
    gfor (af::seq i, vw_noises.dims(1)) {
      // element-wise multiplication, https://arrayfire.org/docs/group__arith__func__mul.htm#ga8317504ec8b9c15d29b27cc77039cb69
      vw_noises(af::span, i) *= sampling_std;
    }
  }

  return control_sequence_.data + vw_noises;
}

bool Optimizer::isHolonomic() const {return motion_model_->isHolonomic();}

void Optimizer::applyControlConstraints()
{

  auto & s = settings_;

  double hi_data[] = {s.constraints_.vx, s.constraints_.vy, s.constraints_.wz};
  af::array hi = af::array(3, hi_data);
  af::array lo = -hi;

  // af::clamp() https://arrayfire.org/docs/namespaceaf.htm#a5d4d2a41fad7d816b70be0e92270dc5f

  // if (isHolonomic()) {

  gfor (af::seq i, state_.controls.dims(1)) {
    state_.controls(af::span, i) = af::clamp(state_.controls(af::span, i), lo, hi);
  }

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
  //TODO
  // if (isHolonomic()) {
  // }
  
  double robot_speed_data[] = {robot_speed.linear.x, robot_speed.linear.y, robot_speed.angular.z};
  
  state.states(af::span, af::span, 0) = af::array(3, robot_speed_data);
}

// this is trajectory rollout
void Optimizer::propagateStateVelocitiesFromInitials(
  models::State & state) const
{
  for (size_t i = 0; i < settings_.time_steps_ - 1; i++) {
    auto curr_state = state.controls(af::span, af::span, i);
    state.states(af::span, af::span, i + 1) = motion_model_->predict(curr_state, state.idx);
    //TODO this is not real predict
  }
}

af::array Optimizer::evalTrajectoryFromControlSequence(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed) const
{
  models::State state;
  state.idx.setLayout(motion_model_->isHolonomic());
  state.reset(1U, settings_.time_steps_);
  state.dts = af::constant(settings_.model_dt_, 1U, settings_.time_steps_);
  state.controls = control_sequence_.data;  

  updateStateVelocities(state, robot_speed);
  // return xt::squeeze(integrateStateVelocities(state, robot_pose));
  return integrateStateVelocities(state, robot_pose);
}

// kinematic part of dynamics and returned in trajectories
// "dynamic" part of dynamics is calculated in propagateStateVelocitiesFromInitials and stored in states_.states
af::array Optimizer::calcKinematics(
  af::array se2,
  af::array state)
{
    af::array state_dt = state * settings_.model_dt_;
    // https://arrayfire.org/docs/group__manip__func__join.htm
    af::array mult1 = af::join(0, state_dt(0), state_dt(0), af::constant(1.0, 1));
    af::array mult2 = af::join(0, af::cos(state_dt(1)), af::sin(state_dt(1)), af::constant(1.0, 1));
    af::array next_se2 = se2 * mult1 * mult2;
    return next_se2;
}

af::array Optimizer::integrateStateVelocities(
  const models::State & state,
  const geometry_msgs::msg::PoseStamped & pose) const
{

  double initial_yaw = tf2::getYaw(pose.pose.orientation);
  double trajecories_0_data[] = {pose.pose.position.x, pose.pose.position.y, initial_yaw};
  af::array trajectories_0 = af::array(3, 1, trajecories_0_data);

  af::array trajectories = af::constant(0, 3, settings_.batch_size_, settings_.time_steps_);

  trajectories(af::span, af::span, 0) = af::tile(trajectories_0, 1, settings_.batch_size_);
  
  for (size_t j = 0; j < settings_.time_steps_ - 1; j++) {
    auto state_j = state.states(af::span, af::span, j);
    auto trajectories_j = trajectories(af::span, af::span, j);
    auto trajectories_j1 = trajectories(af::span, af::span, j+1);
    gfor(af::seq i, settings_.batch_size_)
    {
      trajectories_j1(af::span, i) = calcKinematics(trajectories_j(af::span, i), state_j(af::span, i));
    }
  }

  return trajectories;

}



void Optimizer::updateControlSequence(const af::array & costs)
{
  auto && costs_normalized = costs - af::min(costs);
  auto exponents =
    af::exp(-1 / settings_.temperature_ * costs_normalized);
  auto softmaxes = exponents / af::sum(exponents);
  auto softmaxes_expanded =
    af::tile(softmaxes, 3, 1, settings_.time_steps_);

  control_sequence_.data =
    af::sum(state_.controls * softmaxes_expanded, 0);
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

af::array & Optimizer::getGeneratedTrajectories()
{
  return generated_trajectories_;
}

}  // namespace mppi
