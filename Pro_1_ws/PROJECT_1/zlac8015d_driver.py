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
    CONTROL_REG                    = 0x200E 
    OPR_MODE                       = 0x200D 
    CLR_FB_POS                     = 0x2005 

    L_CMD_RPM                      = 0x2088 
    R_CMD_RPM                      = 0x2089 

    L_ACL_TIME                     = 0x2080 
    R_ACL_TIME                     = 0x2081 
    L_DCL_TIME                     = 0x2082 
    R_DCL_TIME                     = 0x2083 

    L_FB_POS_HI                    = 0x20A7 
    L_FB_RPM                       = 0x20AB 
    R_FB_RPM                       = 0x20AC 
    L_FAULT                        = 0x20A5 

    ENABLE                         = 0x08 
    ALRM_CLR                       = 0x06 
    DOWN_TIME                      = 0x07 
    VEL_CONTROL                    = 3 

modbus_register = ModbusRegister()

class ZLAC8015D_Driver:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200,
                 wheel_radius=0.0535, wheel_base=0.45, cpr=4096, max_rpm=150):
        self.port = port
        self.baudrate = baudrate
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.cpr = cpr                             
        self.travel_in_one_rev = 2 * math.pi * self.wheel_radius 
        self.max_rpm = max_rpm                     
        self.ID = 1                                

        self.client = ModbusClient(
            method='rtu',
            port=self.port, 
            baudrate=self.baudrate, 
            stopbits=1, 
            parity="N", 
            bytesize=8, 
            timeout=0.5
        )
        self.connected = self.client.connect()
        
        self.prev_l_pulse = None
        self.prev_r_pulse = None
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_x = 0.0
        self.w_z = 0.0
        self.max_pulse_diff_threshold = 2000 

    def is_connected(self):
        if not self.client:
            return False
        try:
            # Sửa sang unit=self.ID
            result = self.client.read_holding_registers(modbus_register.L_FAULT, 1, unit=self.ID)
            return result is not None and not result.isError()
        except:
            return False

    def modbus_fail_read_handler(self, ADDR, WORD, max_retries=3, delay=0.02):
        for attempt in range(max_retries):
            try:
                # Sửa sang unit=self.ID
                result = self.client.read_holding_registers(ADDR, WORD, unit=self.ID)
                if result and not result.isError():
                    time.sleep(0.005) # Trễ nhỏ xả đệm RS485
                    return result.registers
            except:
                pass
            time.sleep(delay)
        return None

    def _join_u16_to_s32(self, hi, lo):
        u32 = ((int(hi) & 0xFFFF) << 16) | (int(lo) & 0xFFFF)
        return u32 - 0x100000000 if (u32 & 0x80000000) else u32

    def _int16_to_u16(self, v):
        return int(v) & 0xFFFF

    def init_motor(self):
        if not self.connected:
            self.connected = self.client.connect()
            if not self.connected:
                return False
        try:
            # Sửa tất cả sang unit=self.ID + Thêm khoảng nghỉ nhịp nhàng (sleep 0.1-0.2s)
            self.client.write_register(modbus_register.CONTROL_REG, modbus_register.ALRM_CLR, unit=self.ID)
            time.sleep(0.2)
            
            self.client.write_register(modbus_register.OPR_MODE, modbus_register.VEL_CONTROL, unit=self.ID)
            time.sleep(0.2)
            
            self.set_accel_time(200, 200)
            self.set_decel_time(200, 200)
            time.sleep(0.1)

            self.client.write_register(modbus_register.CONTROL_REG, modbus_register.ENABLE, unit=self.ID)
            time.sleep(0.5) # Dừng 0.5s cho động cơ khởi động hoàn toàn giống file standalone
            return True
        except Exception as e:
            print(f"Lỗi khởi tạo động cơ: {e}")
            return False

    def disable_motor(self):
        try:
            self.client.write_register(modbus_register.CONTROL_REG, modbus_register.DOWN_TIME, unit=self.ID)
        except:
            pass

    def set_accel_time(self, L_ms, R_ms):
        L_ms = max(0, min(32767, L_ms))
        R_ms = max(0, min(32767, R_ms))
        try:
            self.client.write_registers(modbus_register.L_ACL_TIME, [int(L_ms), int(R_ms)], unit=self.ID)
            time.sleep(0.01)
        except Exception as e:
            print(f"loi gui lenh Accel: {e}")

    def set_decel_time(self, L_ms, R_ms):
        L_ms = max(0, min(32767, L_ms))
        R_ms = max(0, min(32767, R_ms))
        try:
            self.client.write_registers(modbus_register.L_DCL_TIME, [int(L_ms), int(R_ms)], unit=self.ID)
            time.sleep(0.01)
        except Exception as e:
            print(f"loi gui lenh Decel: {e}")

    def get_wheels_travelled(self):
        regs = self.modbus_fail_read_handler(modbus_register.L_FB_POS_HI, 4)
        if not regs or len(regs) < 4:
            return None, None
        l_pulse = self._join_u16_to_s32(regs[0], regs[1])
        r_pulse = self._join_u16_to_s32(regs[2], regs[3])
        r_pulse = - r_pulse 
        
        l_travelled = (float(l_pulse) / self.cpr) * self.travel_in_one_rev
        r_travelled = (float(r_pulse) / self.cpr) * self.travel_in_one_rev
        return l_travelled, r_travelled

    def reset_odom(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_l_pulse = None
        self.prev_r_pulse = None
        self.v_x = 0.0
        self.w_z = 0.0

    def set_rpm(self, L_rpm, R_rpm):
        L_rpm = max(min(L_rpm, self.max_rpm), -self.max_rpm)
        R_rpm = max(min(R_rpm, self.max_rpm), -self.max_rpm)

        left_u16 = self._int16_to_u16(L_rpm)
        right_u16 = self._int16_to_u16(-R_rpm) 

        try:
            # Sửa sang unit=self.ID
            self.client.write_registers(modbus_register.L_CMD_RPM, [left_u16, right_u16], unit=self.ID)
            time.sleep(0.01) 
        except Exception as e:
            print(f"loi gui lenh RPM: {e}")

    def update_odometry(self, dt):
        regs = self.modbus_fail_read_handler(modbus_register.L_FB_POS_HI, 4)
        if not regs or len(regs) < 4:
            self.v_x = 0.0
            self.w_z = 0.0
            return self.x, self.y, self.theta, self.v_x, self.w_z
        
        curr_l_pulse = self._join_u16_to_s32(regs[0], regs[1])
        curr_r_pulse = self._join_u16_to_s32(regs[2], regs[3])
        curr_r_pulse = - curr_r_pulse

        if self.prev_l_pulse is None or self.prev_r_pulse is None:
            self.prev_l_pulse = curr_l_pulse
            self.prev_r_pulse = curr_r_pulse
            return self.x, self.y, self.theta, self.v_x, self.w_z
        
        delta_l_pulse = curr_l_pulse - self.prev_l_pulse
        delta_r_pulse = curr_r_pulse - self.prev_r_pulse

        if abs(delta_l_pulse) > self.max_pulse_diff_threshold or abs(delta_r_pulse) > self.max_pulse_diff_threshold:
            self.prev_l_pulse = curr_l_pulse
            self.prev_r_pulse = curr_r_pulse
            self.v_x = 0.0
            self.w_z = 0.0
            return self.x, self.y, self.theta, self.v_x, self.w_z
        
        self.prev_l_pulse = curr_l_pulse
        self.prev_r_pulse = curr_r_pulse

        d_left = (float(delta_l_pulse) / self.cpr) * self.travel_in_one_rev
        d_right = (float(delta_r_pulse) / self.cpr) * self.travel_in_one_rev

        dS = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self.wheel_base

        half_theta = self.theta + (d_theta / 2.0)
        self.x += dS * math.cos(half_theta)
        self.y += dS * math.sin(half_theta)

        self.theta += d_theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        if dt > 0:
            self.v_x = dS / dt
            self.w_z = d_theta / dt
        else:
            self.v_x = 0.0
            self.w_z = 0.0
        return self.x, self.y, self.theta, self.v_x, self.w_z

    def close_connect(self):
        if self.client:
            self.client.close()