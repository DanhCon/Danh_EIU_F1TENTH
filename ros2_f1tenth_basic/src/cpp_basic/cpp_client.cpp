#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

#include <memory>

void tinh_tong(const std::shared_ptr<example_interfaces::srv::AddTwoInts::Request> request, std::shared_ptr<example_interfaces::srv::AddTwoInts::Response> response)
{
    response->sum = request->a + request->b;

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "nhan yeu cau: %ld + %ld", request->a, request->b);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Dang gui phan hoi: %ld ", response->sum);


}
int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("server_tinh_tong");
    rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service = 
      node ->create_service<example_interfaces::srv::AddTwoInts>("tinh_tong_2_so", &tinh_tong);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"),"Server da sang cong 2 so" );
    rclcpp::spin(node);
    rclcpp::shutdown();


}