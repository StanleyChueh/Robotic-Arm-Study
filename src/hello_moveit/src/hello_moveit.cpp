#include <memory>
#include <thread>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

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

  // Create collision object for the robot to avoid (define before using it)
  auto const collision_object = [frame_id = move_group_interface.getPlanningFrame()]() {
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = frame_id;
    collision_object.id = "box1";

    shape_msgs::msg::SolidPrimitive primitive;
    // Define the size of the box in meters
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[primitive.BOX_X] = 0.5;
    primitive.dimensions[primitive.BOX_Y] = 0.1;
    primitive.dimensions[primitive.BOX_Z] = 0.5;

    // Define the pose of the box (relative to the frame_id)
    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;  // Default orientation (x, y, z = 0)
    box_pose.position.x = 0.2;
    box_pose.position.y = 0.2;
    box_pose.position.z = 0.25;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;

    return collision_object;
  }();

  // Add the collision object to the scene (now that it is defined)
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  planning_scene_interface.applyCollisionObject(collision_object);

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

  // Get the JointModelGroup for the planning group "panda_arm"
  auto jmg = move_group_interface.getRobotModel()->getJointModelGroup("panda_arm");

  // Lambda to publish the trajectory (ensure the type matches the overload)
  auto draw_trajectory_tool_path = [&visual_tools, jmg](const moveit_msgs::msg::RobotTrajectory & trajectory) {
    visual_tools.publishTrajectoryLine(trajectory, jmg);
    visual_tools.trigger();
  };

  // Set a target pose
  geometry_msgs::msg::Pose target_pose;
  target_pose.orientation.y = 0.8;
  target_pose.orientation.w = 0.6;
  target_pose.position.x = 0.1;
  target_pose.position.y = 0.4;
  target_pose.position.z = 0.4;
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

// below is the official code from moveit2
// #include <moveit/move_group_interface/move_group_interface.h>
// #include <moveit/planning_scene_interface/planning_scene_interface.h>
// #include <moveit_visual_tools/moveit_visual_tools.h>

// #include <memory>
// #include <rclcpp/rclcpp.hpp>
// #include <thread>

// int main(int argc, char* argv[])
// {
//   // Initialize ROS and create the Node
//   rclcpp::init(argc, argv);
//   auto const node = std::make_shared<rclcpp::Node>(
//       "hello_moveit", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

//   // Create a ROS logger
//   auto const logger = rclcpp::get_logger("hello_moveit");

//   // We spin up a SingleThreadedExecutor for the current state monitor to get
//   // information about the robot's state.
//   rclcpp::executors::SingleThreadedExecutor executor;
//   executor.add_node(node);
//   auto spinner = std::thread([&executor]() { executor.spin(); });

//   // Create the MoveIt MoveGroup Interface
//   using moveit::planning_interface::MoveGroupInterface;
//   auto move_group_interface = MoveGroupInterface(node, "panda_arm");

//   // Construct and initialize MoveItVisualTools
//   auto moveit_visual_tools =
//       moveit_visual_tools::MoveItVisualTools{ node, "base_link", rviz_visual_tools::RVIZ_MARKER_TOPIC,
//                                               move_group_interface.getRobotModel() };
//   moveit_visual_tools.deleteAllMarkers();
//   moveit_visual_tools.loadRemoteControl();

//   // Create a closure for updating the text in rviz
//   auto const draw_title = [&moveit_visual_tools](auto text) {
//     auto const text_pose = [] {
//       auto msg = Eigen::Isometry3d::Identity();
//       msg.translation().z() = 1.0;
//       return msg;
//     }();
//     moveit_visual_tools.publishText(text_pose, text, rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
//   };
//   auto const prompt = [&moveit_visual_tools](auto text) { moveit_visual_tools.prompt(text); };
//   auto const draw_trajectory_tool_path =
//       [&moveit_visual_tools, jmg = move_group_interface.getRobotModel()->getJointModelGroup("panda_arm")](
//           auto const trajectory) { moveit_visual_tools.publishTrajectoryLine(trajectory, jmg); };

//   // Set a target Pose with updated values !!!
//   auto const target_pose = [] {
//     geometry_msgs::msg::Pose msg;
//     msg.orientation.y = 0.8;
//     msg.orientation.w = 0.6;
//     msg.position.x = 0.1;
//     msg.position.y = 0.4;
//     msg.position.z = 0.4;
//     return msg;
//   }();
//   move_group_interface.setPoseTarget(target_pose);

//   // Create collision object for the robot to avoid
//   auto const collision_object = [frame_id = move_group_interface.getPlanningFrame()] {
//     moveit_msgs::msg::CollisionObject collision_object;
//     collision_object.header.frame_id = frame_id;
//     collision_object.id = "box1";
//     shape_msgs::msg::SolidPrimitive primitive;

//     // Define the size of the box in meters
//     primitive.type = primitive.BOX;
//     primitive.dimensions.resize(3);
//     primitive.dimensions[primitive.BOX_X] = 0.5;
//     primitive.dimensions[primitive.BOX_Y] = 0.1;
//     primitive.dimensions[primitive.BOX_Z] = 0.5;

//     // Define the pose of the box (relative to the frame_id)
//     geometry_msgs::msg::Pose box_pose;
//     box_pose.orientation.w = 1.0;
//     box_pose.position.x = 0.2;
//     box_pose.position.y = 0.2;
//     box_pose.position.z = 0.25;

//     collision_object.primitives.push_back(primitive);
//     collision_object.primitive_poses.push_back(box_pose);
//     collision_object.operation = collision_object.ADD;

//     return collision_object;
//   }();

//   // Add the collision object to the scene
//   moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
//   planning_scene_interface.applyCollisionObject(collision_object);

//   // Create a plan to that target pose
//   prompt("Press 'next' in the RvizVisualToolsGui window to plan");
//   draw_title("Planning");
//   moveit_visual_tools.trigger();
//   auto const [success, plan] = [&move_group_interface] {
//     moveit::planning_interface::MoveGroupInterface::Plan msg;
//     auto const ok = static_cast<bool>(move_group_interface.plan(msg));
//     return std::make_pair(ok, msg);
//   }();

//   // Execute the plan
//   if (success)
//   {
//     draw_trajectory_tool_path(plan.trajectory_);
//     moveit_visual_tools.trigger();
//     prompt("Press 'next' in the RvizVisualToolsGui window to execute");
//     draw_title("Executing");
//     moveit_visual_tools.trigger();
//     move_group_interface.execute(plan);
//   }
//   else
//   {
//     draw_title("Planning Failed!");
//     moveit_visual_tools.trigger();
//     RCLCPP_ERROR(logger, "Planning failed!");
//   }

//   // Shutdown ROS
//   rclcpp::shutdown();
//   spinner.join();
//   return 0;
// }