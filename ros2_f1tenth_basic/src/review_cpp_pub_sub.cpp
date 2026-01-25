#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"



class RobotTuhanh : public rclcpp::Node{
    public:
        RobotTuhanh() : Node("node_robot_tu_hanh"){
            publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
            subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
                "/scan",
                10,
                [this](const sensor_msgs::msg::LaserScan & msg){
                    this->xu_ly_lidar (msg);
                }
            );
            RCLCPP_INFO(this->get_logger(), "Robot da khoi dong che do tu lai!");
        }
    private:
        void xu_ly_lidar(const sensor_msgs::msg::LaserScan & msg){
            int mid_idx = msg.ranges.size()/2;
            float dis = msg.ranges[mid_idx];

            this->xu_ly_lai_xe(dis); 
        }
        void xu_ly_lai_xe(float khoang_cach) {

        auto lenh = ackermann_msgs::msg::AckermannDriveStamped();
        
        if (khoang_cach < 1.0) { // Nếu vật gần nhất < 1m
            RCLCPP_WARN(this->get_logger(), "Nguy hiem! Cach %.2f m", khoang_cach);
            lenh.drive.speed = 0.0;
            // Quay trái né
        } else {
            lenh.drive.speed  = 1.0;
            
        }
        publisher_->publish(lenh);
    }

        rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_;
        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
};
int main(int argc, char *argv[]){
    rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<RobotTuhanh>());
    rclcpp::shutdown();
    return 0;
}