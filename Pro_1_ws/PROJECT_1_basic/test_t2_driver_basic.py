import time
# Import module lớp điều khiển từ file zlac8015d_driver.py
from zlac8015d_driver import ZLAC8015D_Driver


def run_sequence(driver: ZLAC8015D_Driver, linear_speed=0.2, angular_speed=0.5, pause_time=2.0):
    """
    Kịch bản điều khiển chuyển động theo chuỗi thời gian:
    1. Tiến trong 5s
    2. Nghỉ 2s -> Lùi trong 5s
    3. Nghỉ 2s -> Quay sang trái trong 2s
    4. Nghỉ 2s -> Quay lại thẳng (xoay ngược lại góc cũ trong 2s)
    5. Nghỉ 2s -> Quay sang phải trong 2s
    """
    def execute_step(step_name: str, linear_x: float, angular_z: float, duration: float):
        print(f"\n[THỰC HIỆN] {step_name} | v = {linear_x} m/s, w = {angular_z} rad/s trong {duration}s")
        start_time = time.time()
        
        # Gửi lệnh liên tục ở chu kỳ 20Hz (0.05s) để tránh Timeout Watchdog của Modbus
        while time.time() - start_time < duration:
            driver.set_twist(linear_x, angular_z)
            time.sleep(0.05)
            
        # Dừng robot giữa các bước thực thi
        print(f"[TẠM DỪNG] Dừng robot trong {pause_time}s...")
        driver.set_twist(0.0, 0.0)
        time.sleep(pause_time)

    try:
        print("=== BẮT ĐẦU CHUỖI ĐIỀU KHIỂN DÙNG THƯ VIỆN ZLAC8015D ===")

        # 1. Tiến thẳng trong 5 giây
        execute_step("1. Tiến thẳng", linear_x=linear_speed, angular_z=0.0, duration=5.0)

        # 2. Lùi lại trong 5 giây
        execute_step("2. Lùi lại", linear_x=-linear_speed, angular_z=0.0, duration=5.0)

        # 3. Quay sang trái tại chỗ (2 giây)
        execute_step("3. Quay sang trái", linear_x=0.0, angular_z=angular_speed, duration=3.0)

        # 4. Quay lại thẳng (xoay góc ngược chiều trái để trả về hướng ban đầu)
        execute_step("4. Quay lại thẳng", linear_x=0.0, angular_z=-angular_speed, duration=3.0)

        # 5. Quay sang phải tại chỗ (2 giây)
        execute_step("5. Quay sang phải", linear_x=0.0, angular_z=-angular_speed, duration=3.0)

        execute_step("6. Quay lai", linear_x=0.0, angular_z=angular_speed, duration=3.0)



    except KeyboardInterrupt:
        print("\n[CẢNH BÁO] Phát hiện tín hiệu dừng khẩn cấp từ phím bấm (Ctrl+C)!")
    finally:
        print("[HỆ THỐNG] Ngắt mô-men động cơ và đóng cổng kết nối RS485...")
        driver.set_twist(0.0, 0.0)
        driver.disable_motor()
        driver.close_connect()


if __name__ == "__main__":
    # 1. Khởi tạo đối tượng driver từ module zlac8015d_driver
    PORT = '/dev/ttyUSB0'
    driver = ZLAC8015D_Driver(port=PORT, baudrate=115200)

    # 2. Kết nối và chuyển động cơ sang chế độ Velocity Mode
    if not driver.init_motor():
        print(f"[LỖI] Không thể kết nối hoặc khởi tạo Driver tại cổng {PORT}")
    else:
        print("[THÀNH CÔNG] Driver đã sẵn sàng nhận lệnh!")
        # 3. Chạy kịch bản điều khiển
        run_sequence(driver, linear_speed=0.2, angular_speed=0.5, pause_time=2.0)