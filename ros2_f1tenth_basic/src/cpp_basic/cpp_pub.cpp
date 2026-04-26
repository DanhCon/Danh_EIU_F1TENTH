class NguoiDieuKhien :public rclcpp::Node{
    public:
         NguoiDieuKhien() : Node("node_dieu_khien_tu_xa"){
            client_ = this-> create_client<std_srvs::srv::SetBool>("/enabale_auto_drive");

            timer_ = this->create_wall_timer(
                5s,[this](){this->gui_lenh_cong_tac();}
            );

         }
    private:
         rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr client_;
         rclcpp::TimerBase::SharedPtr timer_;

         void gui_lenh_cong_tac(){
            
         }
}