import numpy as np
import time
import math
import logging

logging.getLogger('pymodbus').setLevel(logging.ERROR)

try:
    from pymodbus.client import ModbusSerialClient as ModbusClient
except ImportError:
    from pymodbus.client.sync import ModbusSerialClient as ModbusClient

class ModbusRegister:
    CONTROL_REG                    = 0x200E # thanh ghi dieu khien
    # enable/disable/ clear Alarm
    OPR_MODE                       = 0x200D # thanh ghi cai dai che do chay 
    # Van toc / Vi Tri
    CLR_FB_POS                     = 0x2005 # thanh ghi reset vi tri encoder ve 0

    L_CMD_RPM                      = 0x2088 # thanh ghi gui toc do banh trai
    #(vong/phut)
    R_CMD_RPM                      = 0x2089 # thanh ghi gui toc do banh phai
    #(vong/phut)

    L_ACL_TIME                     = 0x2080 # Thoi gian tang toc banh trai (ms)
    R_ACL_TIME                     = 0x2081 # Thoi gian tang toc banh phai (ms)
    L_DCL_TIME                     = 0x2082 # Thoi gian giam toc banh trai (ms)
    R_DCL_TIME                     = 0x2083 # Thoi gian giam toc banh phai (ms)

    L_FB_POS_HI                    = 0x20A7 # thanh ghi doc phan hoi Encoder ( 4 thanh ghi lien tiep)
    

    L_FB_RPM                       = 0x20AB # Phan hoi toc do banh trai thuc

    R_FB_RPM                       = 0x20AC # Phan hoi toc do banh phai thuc 

    L_FAULT                        = 0x20A5 # doc loi driver dong co 

    ENABLE                         = 0x08 # kich hoat dong co 

    ALRM_CLR                       = 0x06 # Xoa loi/ (Alarm clear)

    DOWN_TIME                      = 0x07 # VO HIEU HOA DONG CO de dong co quay tu do 

    VEL_CONTROL                    = 3 #che do dieu khien toc do (Velocity mode)

modbus_register = ModbusRegister()

class ZLAC8015D_Driver:
    def __init__(self, port = '/dev/ttyUSB0', baudrate= 115200,
                 wheel_radius= 0.0535, wheel_base= 0.45, cpr=4096, max_rpm=150):
        self.port = port
        self.baudrate = baudrate
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.cpr = cpr                             # Số xung trên 1 vòng quay của encoder (4096)
        self.travel_in_one_rev = 2 * math.pi * self.wheel_radius # Chu vi bánh xe thực tế (mét)
        self.max_rpm = max_rpm                     # Giới hạn tốc độ RPM tối đa cho động cơ
        self.ID = 1                                # Modbus ID mặc định của driver ZLAC8015D
        # 2. Khởi tạo kết nối Modbus Serial
        self.client = ModbusClient(
           
            port=self.port, 
            baudrate=self.baudrate, 
            stopbits=1, 
            parity="N", 
            bytesize=8, 
            timeout=0.1 # Timeout ngắn (100ms) để không làm treo vòng lặp Odom khi mất kết nối
        )
        self.connected = self.client.connect()
        
        # 3. Các biến lưu trữ trạng thái Encoder phục vụ tính toán vi phân
        self.prev_l_pulse = None
        self.prev_r_pulse = None
        
        # Tọa độ vị trí robot tích lũy toàn cục (X, Y, Yaw)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        # Vận tốc thực tế của robot (m/s và rad/s) tính từ phản hồi encoder
        self.v_x = 0.0
        self.w_z = 0.0
        # Ngưỡng lọc nhiễu tín hiệu: 2000 xung (Nếu thay đổi xung lớn hơn ngưỡng này trong 0.05s
        # thì coi như dữ liệu lỗi do nhiễu Modbus hoặc driver đột ngột reset).
        self.max_pulse_diff_threshold = 2000 
    def is_connected(self):
        """Kiểm tra xem kết nối Modbus tới Driver còn hoạt động tốt hay không"""
        if not self.client:
            return False
        try:
            # Đọc thanh ghi lỗi động cơ, nếu không lỗi và có phản hồi là thành công
            result = self.client.read_holding_registers(modbus_register.L_FAULT, 1, slave=self.ID)
            return result is not None and not result.isError()
        except:
            return False
    def modbus_fail_read_handler(self, ADDR, WORD, max_retries=3, delay=0.01):
        """Hàm đọc Modbus an toàn, tự động thử lại (retry) khi gặp lỗi truyền thông RS485"""
        for attempt in range(max_retries):
            try:
                result = self.client.read_holding_registers(ADDR, WORD, slave=self.ID)
                if result and not result.isError():
                    time.sleep(0.005) # Trễ nhỏ trước khi return để mạch RS485 kịp chuyển đổi TX/RX
                    return result.registers
            except:
                pass
            time.sleep(delay)
            
        # Nếu thử 3 lần vẫn thất bại, in ra cảnh báo!
        print("\n[CẢNH BÁO] Lỗi đọc Odom: Mất kết nối RS485 tạm thời!")
        return None
    def _join_u16_to_s32(self, hi, lo):
        """Ghép hai thanh ghi 16-bit (High & Low) thành một số nguyên 32-bit có dấu (Signed Int32)"""
        # Phép dịch bit: Dịch High sang trái 16 bit rồi kết hợp với Low
        u32 = ((int(hi) & 0xFFFF) << 16) | (int(lo) & 0xFFFF)
        # Chuyển đổi từ không dấu (Unsigned) sang có dấu (Signed) sử dụng bù 2
        return u32 - 0x100000000 if (u32 & 0x80000000) else u32
    def _int16_to_u16(self, v):
        """Chuyển đổi số nguyên 16-bit có dấu sang định dạng 16-bit không dấu để gửi Modbus"""
        return int(v) & 0xFFFF

    def init_motor(self):
        """Kích hoạt và cài đặt động cơ chạy ở chế độ điều khiển tốc độ (Velocity Mode)"""
        if not self.connected:
            self.connected = self.client.connect()
            if not self.connected:
                return False
        try:
            # 1. Xóa tất cả các báo động/lỗi đang có trên driver
            self.client.write_register(modbus_register.CONTROL_REG, modbus_register.ALRM_CLR, slave=self.ID)
            time.sleep(0.1)
            # 2. Cài đặt chế độ chạy là Velocity Mode
            self.client.write_register(modbus_register.OPR_MODE, modbus_register.VEL_CONTROL, slave=self.ID)
            time.sleep(0.1)
            
            # Cài đặt gia tốc phần cứng (Hardware Acceleration) 200ms để làm mềm quỹ đạo
            self.set_accel_time(200, 200)
            self.set_decel_time(200, 200)

            # 3. Kích hoạt (Enable) động cơ cấp dòng khóa trục bánh xe
            self.client.write_register(modbus_register.CONTROL_REG, modbus_register.ENABLE, slave=self.ID)
            time.sleep(0.1)
            return True
        except Exception as e:
            print(f"Lỗi khởi tạo động cơ: {e}")
            return False
    def disable_motor(self):
        try:
            self.client.write_register(modbus_register.CONTROL_REG, modbus_register.DOWN_TIME, slave=self.ID)
        except:
            pass

    def set_accel_time(self, L_ms, R_ms):
        """🔧 Cai dat thoi gian tang toc (ms)"""
        L_ms = max(0, min(32767, L_ms))
        R_ms = max(0, min(32767, R_ms))
        try:
            self.client.write_registers(modbus_register.L_ACL_TIME, [int(L_ms), int(R_ms)], slave=self.ID)
            time.sleep(0.01)
        except Exception as e:
            print(f"loi gui lenh Accel: {e}")

    def set_decel_time(self, L_ms, R_ms):
        """🔧 Cai dat thoi gian giam toc (ms)"""
        L_ms = max(0, min(32767, L_ms))
        R_ms = max(0, min(32767, R_ms))
        try:
            self.client.write_registers(modbus_register.L_DCL_TIME, [int(L_ms), int(R_ms)], slave=self.ID)
            time.sleep(0.01)
        except Exception as e:
            print(f"loi gui lenh Decel: {e}")
            
    def get_wheels_travelled(self):
        """Đọc tổng quãng đường bánh xe đã quay được (mét) - tính tuyệt đối"""
        regs = self.modbus_fail_read_handler(modbus_register.L_FB_POS_HI, 4)
        if not regs or len(regs) < 4:
            return None, None
        l_pulse = self._join_u16_to_s32(regs[0], regs[1])
        r_pulse = self._join_u16_to_s32(regs[2], regs[3])
        r_pulse = - r_pulse # bánh phải bị đảo chiều
        
        l_travelled = (float(l_pulse) / self.cpr) * self.travel_in_one_rev
        r_travelled = (float(r_pulse) / self.cpr) * self.travel_in_one_rev
        return l_travelled, r_travelled
    
    def reset_odom(self):
        self.x = 0.0
        self.y= 0.0
        self.theta = 0.0
        self.prev_l_pulse = None
        self.prev_r_pulse = None
        self.v_x = 0.0
        self.w_z = 0.0

    def set_rpm(self, L_rpm, R_rpm):
        """ Gui lenh xuong qua Modbus RTU"""
        L_rpm = max(min(L_rpm, self.max_rpm), -self.max_rpm)
        R_rpm = max(min(R_rpm, self.max_rpm), -self.max_rpm)

        left_u16 = self._int16_to_u16(L_rpm)
        right_u16 = self._int16_to_u16(-R_rpm) 

        try:
            self.client.write_registers(modbus_register.L_CMD_RPM, [left_u16,right_u16], slave= self.ID)
            time.sleep(0.01) # <-- Tăng delay lên 10ms để RS485 xả đệm an toàn hơn
        except Exception as e:
            print(f"loi gui lenh RPM: {e}")
    def set_twist(self, linear_x, angular_z):
        """ Dong hoc nguoc (Inverse Kinematics): Quy doi tu toc do robot (m/s vaf rad/s) --> sang RPM cua banh xe trai phai"""

        v_l = linear_x - angular_z *(self.wheel_base/2.0)
        v_r =  linear_x + angular_z *(self.wheel_base/2.0)

        l_rpm = (v_l*60) /self.travel_in_one_rev 
        r_rpm = (v_r*60) / self.travel_in_one_rev

        self.set_rpm(l_rpm, r_rpm)

    def update_odometry(self, dt):
        """ Dong hoc thuan + Loc Nhieu Encoder :
        Doc encoder tu driver, xu ly tran so/ loi nhay vot xung va cap nhat vi tri (X, Y, Theta)"""

        regs = self.modbus_fail_read_handler(modbus_register.L_FB_POS_HI, 4)
        if not regs or len(regs) < 4:

            self.v_x = 0.0
            self.w_z = 0.0
            return self.x, self.y , self.theta, self.v_x , self.w_z
        
        curr_l_pulse = self._join_u16_to_s32(regs[0], regs[1])
        curr_r_pulse = self._join_u16_to_s32(regs[2], regs[3])
        curr_r_pulse =  - curr_r_pulse

        if self.prev_l_pulse is None or self.prev_r_pulse is None:
            self.prev_l_pulse = curr_l_pulse
            self.prev_r_pulse = curr_r_pulse

            return self.x, self.y, self.theta, self.v_x , self.w_z
        
        delta_l_pulse = curr_l_pulse - self.prev_l_pulse
        delta_r_pulse = curr_r_pulse - self.prev_r_pulse

        # Bo loc

        if abs(delta_l_pulse) > self.max_pulse_diff_threshold or abs(delta_r_pulse) > self.max_pulse_diff_threshold:
            self.prev_l_pulse = curr_l_pulse
            self.prev_r_pulse = curr_r_pulse

            self.v_x = 0.0
            self.w_z = 0.0

            return self.x , self.y, self.theta, self.v_x , self.w_z
        
        self.prev_l_pulse = curr_l_pulse
        self.prev_r_pulse = curr_r_pulse

        d_left = (float(delta_l_pulse) / self.cpr) * self.travel_in_one_rev
        d_right = (float(delta_r_pulse)/self.cpr) * self.travel_in_one_rev

        dS = (d_left + d_right)/2

        d_theta = (d_right - d_left)/self.wheel_base

        half_theta =self.theta + (d_theta/2.0)
        self.x += dS*math.cos(half_theta)
        self.y += dS*math.sin(half_theta)

        self.theta += d_theta
        
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        if dt > 0:
            self.v_x = dS /dt
            self.w_z = d_theta/dt
        else:
            self.v_x = 0.0
            self.w_z = 0.0
        return self.x, self.y, self.theta, self.v_x, self.w_z
    def close_connect(self):
        if self.client:
            self.client.close()