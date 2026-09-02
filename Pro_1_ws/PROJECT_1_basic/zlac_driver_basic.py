import time
import logging

logging.getLogger('pymodbus').setLevel(logging.ERROR)

try:
    from pymodbus.client import ModbusSerialClient as ModbusClient
except ImportError:
    from pymodbus.client.sync import ModbusSerialClient as ModbusClient


class ModbusRegister:
    CONTROL_REG = 0x200E  # Thanh ghi điều khiển (Enable / Disable / Clear Alarm)
    OPR_MODE    = 0x200D  # Thanh ghi cài đặt chế độ hoạt động (Velocity / Position Mode)
    CLR_FB_POS  = 0x2005  # Thanh ghi reset vị trí encoder về 0

    L_CMD_RPM   = 0x2088  # Thanh ghi gửi tốc độ bánh trái (RPM)
    R_CMD_RPM   = 0x2089  # Thanh ghi gửi tốc độ bánh phải (RPM)

    L_ACL_TIME  = 0x2080  # Thời gian tăng tốc bánh trái (ms)
    R_ACL_TIME  = 0x2081  # Thời gian tăng tốc bánh phải (ms)
    L_DCL_TIME  = 0x2082  # Thời gian giảm tốc bánh trái (ms)
    R_DCL_TIME  = 0x2083  # Thời gian giảm tốc bánh phải (ms)

    L_FAULT     = 0x20A5  # Đọc lỗi driver động cơ

    ENABLE      = 0x08    # Kích hoạt động cơ
    ALRM_CLR    = 0x06    # Xóa lỗi (Alarm Clear)
    DOWN_TIME   = 0x07    # Vô hiệu hóa động cơ (Khóa trục nhả ra, quay tự do)

    VEL_CONTROL = 3       # Chế độ điều khiển tốc độ (Velocity Mode)


modbus_register = ModbusRegister()


class ZLAC8015D_Driver:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200,
                 wheel_base=0.45, travel_in_1_vong=0.336, max_rpm=150):
        self.port = port
        self.baudrate = baudrate
        self.wheel_base = wheel_base                # Khoảng cách giữa 2 bánh xe (m)
        self.travel_in_one_rev = travel_in_1_vong  # Chu vi bánh xe (m)
        self.max_rpm = max_rpm                      # Giới hạn RPM tối đa
        self.ID = 1                                 # Modbus Slave ID

        # Khởi tạo kết nối Modbus Serial (RS485)
        self.client = ModbusClient(
            port=self.port,
            baudrate=self.baudrate,
            stopbits=1,
            parity="N",
            bytesize=8,
            timeout=0.1
        )
        self.connected = self.client.connect()

    def is_connected(self):
        """Kiểm tra trạng thái kết nối tới Driver"""
        if not self.client:
            return False
        try:
            result = self.client.read_holding_registers(modbus_register.L_FAULT, 1, slave=self.ID)
            return result is not None and not result.isError()
        except Exception:
            return False

    def _int16_to_u16(self, v):
        """Chuyển đổi số nguyên 16-bit có dấu (Signed Int16) sang dạng không dấu (Unsigned UInt16) để nạp vào thanh ghi Modbus"""
        return int(v) & 0xFFFF

    def init_motor(self):
        """Kích hoạt và chuyển Driver sang Velocity Mode"""
        if not self.connected:
            self.connected = self.client.connect()
            if not self.connected:
                return False
        try:
            # 1. Xóa Alarm/Fault
            self.client.write_register(modbus_register.CONTROL_REG, modbus_register.ALRM_CLR, slave=self.ID)
            time.sleep(0.1)

            # 2. Chuyển sang Velocity Control Mode
            self.client.write_register(modbus_register.OPR_MODE, modbus_register.VEL_CONTROL, slave=self.ID)
            time.sleep(0.1)

            # 3. Cài đặt gia tốc/giảm tốc phần cứng (200ms)
            self.set_accel_time(200, 200)
            self.set_decel_time(200, 200)

            # 4. Kích hoạt động cơ
            self.client.write_register(modbus_register.CONTROL_REG, modbus_register.ENABLE, slave=self.ID)
            time.sleep(0.1)
            return True
        except Exception as e:
            print(f"Lỗi khởi tạo động cơ: {e}")
            return False

    def disable_motor(self):
        """Tắt động cơ (Ngắt dòng cấp)"""
        try:
            self.client.write_register(modbus_register.CONTROL_REG, modbus_register.DOWN_TIME, slave=self.ID)
        except Exception:
            pass

    def set_accel_time(self, L_ms, R_ms):
        """Cài đặt thời gian tăng tốc (ms)"""
        L_ms = max(0, min(32767, L_ms))
        R_ms = max(0, min(32767, R_ms))
        try:
            self.client.write_registers(modbus_register.L_ACL_TIME, [int(L_ms), int(R_ms)], slave=self.ID)
            time.sleep(0.01)
        except Exception as e:
            print(f"Lỗi gửi lệnh Accel: {e}")

    def set_decel_time(self, L_ms, R_ms):
        """Cài đặt thời gian giảm tốc (ms)"""
        L_ms = max(0, min(32767, L_ms))
        R_ms = max(0, min(32767, R_ms))
        try:
            self.client.write_registers(modbus_register.L_DCL_TIME, [int(L_ms), int(R_ms)], slave=self.ID)
            time.sleep(0.01)
        except Exception as e:
            print(f"Lỗi gửi lệnh Decel: {e}")

    def set_rpm(self, L_rpm, R_rpm):
        """Gửi lệnh vận tốc RPM trực tiếp xuống hai bánh qua Modbus RTU"""
        # Sát thực giới hạn Clamp vận tốc đầu vào
        L_rpm = max(min(L_rpm, self.max_rpm), -self.max_rpm)
        R_rpm = max(min(R_rpm, self.max_rpm), -self.max_rpm)

        # Chuyển đổi định dạng dữ liệu truyền thông
        # Lưu ý: Bánh bên phải được đảo chiều do đặc thù lắp đối xứng của hệ khung gầm Differential Drive
        left_u16 = self._int16_to_u16(L_rpm)
        right_u16 = self._int16_to_u16(-R_rpm)

        try:
            self.client.write_registers(modbus_register.L_CMD_RPM, [left_u16, right_u16], slave=self.ID)
            time.sleep(0.01)  # Delay để đảm bảo đường truyền RS485 không bị nghẽn đệm
        except Exception as e:
            print(f"Lỗi gửi lệnh RPM: {e}")

    def set_twist(self, linear_x, angular_z):
        """
        Mô hình Động học ngược (Inverse Kinematics) cho robot vi phân (Differential Drive)
        Chuyển đổi từ vận tốc dài v (m/s) và vận tốc góc w (rad/s) sang tốc độ vòng/phút (RPM) của từng bánh.
        """
        # 1. Tính vận tốc dài tuyến tính của từng bánh xe v_L, v_R (m/s)
        v_l = linear_x - angular_z * (self.wheel_base / 2.0)
        v_r = linear_x + angular_z * (self.wheel_base / 2.0)

        # 2. Quy đổi từ vận tốc dài (m/s) sang vòng/phút (RPM) dựa vào chu vi bánh xe C
        # RPM = (v_m/s * 60 s) / C_m
        l_rpm = (v_l * 60.0) / self.travel_in_one_rev
        r_rpm = (v_r * 60.0) / self.travel_in_one_rev

        # 3. Gửi lệnh RPM xuống driver
        self.set_rpm(l_rpm, r_rpm)

    def close_connect(self):
        """Đóng cổng nối tiếp RS485"""
        if self.client:
            self.client.close()