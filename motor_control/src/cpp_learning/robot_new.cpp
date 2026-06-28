#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class RobotNew : public rclcpp::Node
{
public:
    RobotNew() : Node("robot_new"), robot_name_("R2D2")
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("robot_news", 10);
        timer_ = this->create_wall_timer(0.5s,std::bind(&RobotNew::publishNews,this));
        RCLCPP_INFO(this->get_logger(), "robot news stantion has bên started");
    }
private:
    void publishNews(){
        auto msg = std_msgs::msg::String();
        msg.data = std::string("hi, this is ") + robot_name_ + std::string("from the robot news station");
        publisher_->publish(msg);
    }
    std::string robot_name_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

};
int main(int argc, char *argv[])
{
    rclcpp::init(argc,argv);
    auto node = std::make_shared<RobotNew>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

