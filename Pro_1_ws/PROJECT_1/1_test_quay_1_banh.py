import time
import serial


class ZLAC8015DController:

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, slave_id: int = 1):
        self.slave_id = slave_id
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
        except serial.SerialException as e:
            print(f"Không thể mở cổng kết nối {port}: {e}")
            raise

    @staticmethod
    def calculate_crc(data: bytes) -> bytes:
        """Thuật toán tính toán CRC16-Modbus tiêu chuẩn sử dụng bitwise."""
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for _ in range(8):
                if (crc & 1) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        # Trả về định dạng Low byte trước, High byte sau
        return bytes([crc & 0xFF, (crc >> 8) & 0xFF])

    def _send_command(self, func_code: int, register: int, value_bytes: bytes) -> bytes:
        """Xây dựng cấu trúc khung truyền ADU và gửi qua RS485."""
        # Ghép khung: ID + Func Code + Reg High + Reg Low + Data
        packet = (
            bytes([self.slave_id, func_code])+
            bytes([(register >> 8) & 0xFF, register & 0xFF])+
            value_bytes
        )
        packet += self.calculate_crc(packet)

        self.ser.reset_input_buffer()
        self.ser.write(packet)
        self.ser.flush()

        # Đọc phản hồi (Cơ bản cho hàm ghi Single Register là 8 bytes)
        response = self.ser.read(8)
        return response

    def clear_fault(self):
        """Xóa trạng thái lỗi hiện tại của Driver (Thanh ghi 200Eh -> giá trị 0x0006)."""
        return self._send_command(0x06, 0x200E, bytes([0x00, 0x06]))

    def set_mode(self, mode: int = 3):
        """Đặt chế độ điều khiển. Mặc định là 3 (Profile Velocity Mode)."""
        return self._send_command(0x06, 0x200D, bytes([0x00, mode]))

    def enable_motor(self):
        """Kích hoạt trục động cơ để sẵn sàng nhận lệnh tốc độ."""
        return self._send_command(0x06, 0x200E, bytes([0x00, 0x08]))

    def set_wheel_speed(self, is_left: bool, speed_rpm: int):
        """Điều khiển vận tốc của một bánh xe đơn lẻ.

        Kiểu dữ liệu I16: Xử lý số bù 2 (Two's Complement) cho vận tốc âm (đảo chiều).
        """
        # Giới hạn dải tốc độ an toàn theo thông số phần cứng
        speed_rpm = max(min(speed_rpm, 1000), -1000)

        # Chuyển đổi sang định dạng 2 bytes số bù 2
        value_bytes = (speed_rpm).to_bytes(2, byteorder="big", signed=True)

        # Xác định địa chỉ thanh ghi: Bánh trái = 2088h, Bánh phải = 2089h
        target_register = 0x2088 if is_left else 0x2089

        return self._send_command(0x06, target_register, value_bytes)

    def emergency_stop(self):
        """Lệnh dừng khẩn cấp lập tức khóa trục hoặc dừng tự do."""
        return self._send_command(0x06, 0x200E, bytes([0x00, 0x05]))

    def close(self):
        if self.ser.is_open:
            self.ser.close()


# --- Kịch bản vận hành thử nghiệm (Kiểm thử thực tế) ---
if __name__ == "__main__":
    # Khởi tạo bộ điều khiển kết nối với Driver
    wheel = ZLAC8015DController(port="/dev/ttyUSB0", slave_id=1)

    print("Bước 1: Khởi tạo hệ thống và xóa lỗi...")
    wheel.clear_fault()
    time.sleep(0.1)

    print("Bước 2: Cấu hình chế độ vận tốc thanh ghi 200Dh...")
    wheel.set_mode(3)
    time.sleep(0.1)

    print("Bước 3: Enable động cơ...")
    wheel.enable_motor()
    time.sleep(0.1)

    print("Bước 4: Điều khiển bánh trái quay thuận với tốc độ 100 RPM...")
    wheel.set_wheel_speed(is_left=False, speed_rpm=50)

    time.sleep(3)  # Chạy thử trong 3 giây

    print("Bước 5: Dừng bánh xe...")
    wheel.set_wheel_speed(is_left=False, speed_rpm=0)

    wheel.close()