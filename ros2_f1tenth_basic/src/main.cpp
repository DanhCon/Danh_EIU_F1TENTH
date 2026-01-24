#include "motor_control/dynamixel_sdk_wrapper.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace robotis::turtlebot3;

int main()
{
  // 1️⃣ Cấu hình thông tin động cơ
  DynamixelSDKWrapper::Device dxl_device;
  dxl_device.usb_port = "/dev/ttyACM0"; // thay vì /dev/ttyUSB0
  dxl_device.baud_rate = 57600;
  dxl_device.protocol_version = 2.0;

  // dxl_device.protocol_version = 2.0;       // Dynamixel XL430, XM430,... dùng 2.0

  // 2️⃣ Khởi tạo SDK wrapper
  DynamixelSDKWrapper dxl(dxl_device);

  if (!dxl.is_connected_to_device()) {
    std::cerr << "❌ Không kết nối được đến động cơ!" << std::endl;
    return -1;
  }
  std::cout << "✅ Kết nối thành công đến động cơ ID " << (int)dxl_device.id << std::endl;

  // 3️⃣ Địa chỉ thanh ghi bạn muốn truy cập
  const uint16_t ADDR_TORQUE_ENABLE = 64;
  const uint16_t ADDR_GOAL_VELOCITY = 104;
  const uint16_t ADDR_PRESENT_POSITION = 132;
  const uint16_t LEN_4BYTE = 4;

  // 4️⃣ Gửi dữ liệu: bật Torque (enable)
  uint8_t torque_enable = 1;
  std::string msg;
  dxl.set_data_to_device(ADDR_TORQUE_ENABLE, 1, &torque_enable, &msg);
  std::cout << "⚙️ " << msg << std::endl;

  // 5️⃣ Gửi dữ liệu: đặt vận tốc mục tiêu
  int32_t goal_velocity = 100;  // đơn vị: 0.229 [rev/min]
  uint8_t* vel_ptr = reinterpret_cast<uint8_t*>(&goal_velocity);
  dxl.set_data_to_device(ADDR_GOAL_VELOCITY, LEN_4BYTE, vel_ptr, &msg);
  std::cout << "🚀 Gửi tốc độ: " << goal_velocity << " -> " << msg << std::endl;

  // 6️⃣ Đọc dữ liệu: vị trí hiện tại
  dxl.init_read_memory(ADDR_PRESENT_POSITION, LEN_4BYTE);
  for (int i = 0; i < 5; i++)
  {
    dxl.read_data_set();  // đọc bộ nhớ
    int32_t pos = dxl.get_data_from_device<int32_t>(ADDR_PRESENT_POSITION, LEN_4BYTE);
    std::cout << "📍 Lần " << i+1 << " | Vị trí hiện tại: " << pos << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  // 7️⃣ Dừng động cơ
  int32_t stop_velocity = 0;
  vel_ptr = reinterpret_cast<uint8_t*>(&stop_velocity);
  dxl.set_data_to_device(ADDR_GOAL_VELOCITY, LEN_4BYTE, vel_ptr, &msg);
  std::cout << "🛑 Dừng động cơ -> " << msg << std::endl;

  return 0;
}

