#include <memory>
#include <thread>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

int main(int argc, char * argv[])
{
  // Initialize ROS 2 and create the node
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
    "hello_moveit",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );
  auto logger = rclcpp::get_logger("hello_moveit");

  // Start a SingleThreadedExecutor in a separate thread
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  // Instantiate MoveItVisualTools correctly using the node
  moveit_visual_tools::MoveItVisualTools visual_tools(node, "panda_link0", "rviz_visual_markers");
  visual_tools.deleteAllMarkers();
  visual_tools.loadRemoteControl();  // Optional: allows remote control via RViz

  // Create the MoveGroupInterface (using the proper namespace)
  using moveit::planning_interface::MoveGroupInterface;
  MoveGroupInterface move_group_interface(node, "panda_arm");

  // Define lambdas for visualization (after instantiating visual_tools and move_group_interface)
  auto draw_title = [&visual_tools](const std::string & text) {
    // Create a pose for the text (e.g. 1 m above the base frame)
    Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
    text_pose.translation().z() = 1.0;
    visual_tools.publishText(text_pose, text, rviz_visual_tools::WHITE,
                               rviz_visual_tools::XLARGE);
    visual_tools.trigger();
  };

  auto prompt = [&visual_tools](const std::string & text) {
    visual_tools.prompt(text);
  };

  // Get the JointModelGroup for the planning group "manipulator"
  auto jmg = move_group_interface.getRobotModel()->getJointModelGroup("panda_arm");

  // Lambda to publish the trajectory (ensure the type matches the overload)
  auto draw_trajectory_tool_path = [&visual_tools, jmg](const moveit_msgs::msg::RobotTrajectory & trajectory) {
    visual_tools.publishTrajectoryLine(trajectory, jmg);
    visual_tools.trigger();
  };

  // Set a target pose
  geometry_msgs::msg::Pose target_pose;
  target_pose.orientation.w = 1.0;
  target_pose.position.x = 0.28;
  target_pose.position.y = -0.2;
  target_pose.position.z = 0.5;
  move_group_interface.setPoseTarget(target_pose);

  // Inform the user and plan a trajectory
  prompt("Press 'Next' in the RvizVisualToolsGui window to plan");
  draw_title("Planning");

  // Create a plan – note: ensure you are using the correct member (usually `trajectory_`)
  MoveGroupInterface::Plan plan;
  bool success = static_cast<bool>(move_group_interface.plan(plan));

  if (success)
  {
    // Use the correct trajectory member (e.g., plan.trajectory_) when publishing the trajectory
    draw_trajectory_tool_path(plan.trajectory_);
    prompt("Press 'Next' in the RvizVisualToolsGui window to execute");
    draw_title("Executing");
    move_group_interface.execute(plan);
  }
  else
  {
    draw_title("Planning Failed!");
    RCLCPP_ERROR(logger, "Planning failed!");
  }

  // Shutdown ROS 2
  rclcpp::shutdown();
  spinner.join();
  return 0;
}
