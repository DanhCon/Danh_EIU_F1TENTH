
from pymodbus.client.sync import ModbusSerialClient as ModbusClient # type: ignore
import numpy as np
import time 
import pymodbus
import modbus_register


class ZLAC8015D_API:
    def __init__(self, node, param):

        self.param = param ; self.node = node

        self.client = ModbusClient(method=self.param.connection_type, port=self.param.modbus_port, baudrate=self.param.modbus_baudrate, stopbits=1,parity="N",bytesize=8,timeout=1)
        
        self.was_connected = False 

        # ⏳ Last connection attempt time
        self.last_attempt_time = 0  
        
        # ⏳ Retry connection only after 5 seconds
        self.reconnect_interval = 5  
        
        self.connected = self.client.connect()
       
        if self.connected:
            print( str(self.connected) + " ==> " + "successfully")
        else:
            print(str(self.connected) + " " + "Failed connection")
            print("Please recheck parameters again!")

        self.ID = 1
        
        ## 4 inches wheel
        self.travel_in_one_rev = self.param.travel_in_one_rev
        self.cpr = self.param.cpr
        self.R_Wheel = self.param.wheel_radius

    def is_connected(self):
        """🔍 Check connection status and attempt to reconnect if necessary"""
        if not self.client:
            return False

        try:
            result = self.client.read_holding_registers(modbus_register.L_FAULT, 1, unit=self.ID)
            if result is not None and not result.isError():
                if not self.was_connected:
                    print("✅ Modbus connection has been restored!")
                    self.was_connected = True
                return True
        except:
            pass

        now = time.time()
        if self.was_connected and (now - self.last_attempt_time > self.reconnect_interval):
            print("⚠ Modbus connection lost! Attempting to reconnect...")
            self.last_attempt_time = now
            self.was_connected = False

        if self.client.connect():
            print("✅ Modbus reconnection successful!")
            self.was_connected = True
            return True

        return False

    def reset_motor_after_reconnect(self):
        """🔄 Reset motor status after reconnect"""
        try:
            if not self.is_connected():
                print("⚠ Modbus connection lost! Attempting to reconnect...")
                self.client.connect()
                time.sleep(1)

            if self.is_connected():
                print("✅ Modbus reconnection successful! Reset motor status...")

                for cmd, desc in [
                    (modbus_register.ALRM_CLR, "Clear Alarm"),
                    (modbus_register.ENABLE, "Enable Motor"),
                    ((0, 0), "Set RPM = 0")]:
                    try:
                        if isinstance(cmd, tuple):
                            self.client.write_registers(modbus_register.L_CMD_RPM, list(cmd), unit=self.ID)
                        else:
                            self.client.write_register(modbus_register.CONTROL_REG, cmd, unit=self.ID)
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"❌ Error sending command {desc}: {str(e)}")
                        return  # ❌ If error, Stop immediately

                print("✅ Motor is ready for control")
            else:
                print("❌ Could not reconnect. Please check connection wires  or power source!")
        except Exception as e:
            print(f"❌ Error when reset motor: {str(e)}")


    def modbus_fail_read_handler(self, ADDR, WORD, max_retries=3, delay=0.1):
        """Read Modbus data with a maximum of max_retries attempts"""
        reg = [None] * WORD  #  Array to store read data

        for attempt in range(max_retries):
            try:
                result = self.client.read_holding_registers(ADDR, WORD, unit=self.ID)

                # ✅ Check for errors before processing data
                if result and not result.isError():
                    reg = result.registers  # Assign data read
                    return reg  # ✅ successful, return the result

                print(f"⚠️ [Attempt {attempt+1}/{max_retries}] Error when read modus: {result}")

            except AttributeError as e:
                print(f"❌ [Attempt {attempt+1}/{max_retries}] AttributeError: {e}")

            except Exception as e:
                print(f"❌ [Attempt{attempt+1}/{max_retries}] Unidentified error: {e}")

            time.sleep(delay)  # ⏳ Wait before retrying

        print(f"❌ Unable read Modbus address {ADDR} after {max_retries} attempts!")
        return reg  # Return default value if failed 


    def _u16_to_s16(self, u):
        u = int(u) & 0xFFFF
        return u - 0x10000 if (u & 0x8000) else u

    def _join_u16_to_s32(self, hi, lo):
        u32 = ((int(hi) & 0xFFFF) << 16) | (int(lo) & 0xFFFF)
        return u32 - 0x100000000 if (u32 & 0x80000000) else u32

    def _int16_to_u16(self, v):
        # gửi xuống Modbus: signed int16 -> dạng u16 (two's complement)
        return int(v) & 0xFFFF
    

    def rpm_to_radPerSec(self, rpm):

        """Convert RPM to radians per second"""
        return rpm*2*np.pi/60.0
    
    def get_angular_velocity(self):
        """Calculate angular velocity from RPM of both values"""
        rpmL, rpmR = self.get_rpm()
        angular_L =  self.rpm_to_radPerSec(rpmL)
        angular_R = self.rpm_to_radPerSec(rpmR)
        return angular_L, -angular_R
        
    
    def rpm_to_linear(self, rpm):
        """Convert RPM to linear velocity."""
        W_Wheel = self.rpm_to_radPerSec(rpm)
        V = W_Wheel*self.R_Wheel
        return V
    
    def set_mode(self, MODE):
        """Set the operation mode for the motor (relative, absolute, or velocity mode)"""
        if MODE == 1:
            print("Set relative position mode")
        elif MODE == 2:
            print("Set absolute position mode")
        elif MODE == 3:
            print("Set velocity mode")
        else:
            print("set_mode ERROR: set only 1, 2, or 3")
            return 0
        
        result = self.client.write_register(modbus_register.OPR_MODE, MODE, unit=self.ID)
        return result

    def clear_position(self, pos):
        if pos not in [0, 1, 2, 3]:  # Check if position value is valid
            print("❌ set_mode POS: only 0, 1, 2, or 3")
            return False

        messages = {
            1: "✅ Reset encoder position (Left)",
            2: "✅ Reset encoder position (Right)",
            3: "✅ Reset encoder position (Both sides)"
        }

        print(messages.get(pos, "❌ Invalid pos value"))

        try:
            # Write value to Modbus register 
            self.client.write_register(modbus_register.CLR_FB_POS, pos, unit=self.ID)
            return True  # Return True if no errors
        except Exception as e:
            print(f"❌ Error sending reset encoder command: {str(e)}")
            return False  # Return False if error occurs


    # mode = registers[0]
    # return mode
    def enable_motor(self):
        """Enable the motor"""
        result = self.client.write_register(modbus_register.CONTROL_REG, modbus_register.ENABLE, unit=self.ID)       
        
    def disable_motor(self):
        "Disable to the motor"
        result = self.client.write_register(modbus_register.CONTROL_REG, modbus_register.DOWN_TIME, unit=self.ID)
    
    def emergency_stop(self):
        """Stop the motor immediately (emergency stop)."""
        result = self.client.write_register(modbus_register.CONTROL_REG, modbus_register.EMER_STOP, unit=self.ID)
    def get_default(self): # Clear Alarm
        """Clear alarm (reset default state)"""
        result = self.client.write_register(modbus_register.CONTROL_REG, modbus_register.ALRM_CLR, unit=self.ID)
        
    def get_fault_code(self):
        """🔍 Retrieve the fault codes of the motor, with retry mechanism if read fails."""
        try:
            fault_codes = self.modbus_fail_read_handler(modbus_register.L_FAULT, 2)  # Use the retry read function

            # ✅ Check if data is valid
            if fault_codes is None or len(fault_codes) < 2:
                print("⚠️ Error! Unable to read fault codes from Modbus!")
                return (False, None), (False, None)  # Return default error state

            # ✅ Read fault codes for both motors
            L_fault_code, R_fault_code = fault_codes[0], fault_codes[1]
            L_fault_flag, R_fault_flag = L_fault_code in modbus_register.FAULT_LIST, R_fault_code in modbus_register.FAULT_LIST

            return (L_fault_flag, L_fault_code), (R_fault_flag, R_fault_code)

        except Exception as e:
            print(f"❌ Undefined error when reading fault code: {str(e)}")
            return (False, None), (False, None)  # Return default error state

    def set_accel_time(self, L_ms, R_ms):
        """🔧 Set acceleration time for the motor (ms)"""
        if not self.is_connected():
            print("⚠️ Unable to set acceleration time: Modbus connection lost!")
            return
        L_ms = max(0, min(32767, L_ms))
        R_ms = max(0, min(32767, R_ms))

        try:
            self.client.write_registers(modbus_register.L_ACL_TIME, [int(L_ms), int(R_ms)], unit=self.ID)
            print(f"✅ Acceleration time set successfully:: L={L_ms} ms, R={R_ms} ms")
        except Exception as e:
            print(f"❌ Error setting acceleration time: {str(e)}")

    def set_decel_time(self, L_ms, R_ms):
        """Set deceleration time for the motor (ms)"""
        if not self.is_connected():
            print("⚠️ Unable to set deceleration time: Modbus connection lost!!")
            return
        L_ms = max(0, min(32767, L_ms))
        R_ms = max(0, min(32767, R_ms))

        try:
            self.client.write_registers(modbus_register.L_DCL_TIME, [int(L_ms), int(R_ms)], unit=self.ID)
            print(f"✅ Deceleration time set successfully: L={L_ms} ms, R={R_ms} ms")
        except Exception as e:
            print(f"❌ Error setting deceleration: {str(e)}")

    
    # def get_rpm(self):
    #     """Get the RPM (rotations per minute) of the motor"""
    #     if not self.is_connected():
    #         print("⚠️ Unable to get RPM speed: Modbus connection lost!")
    #         return 0.0, 0.0  # Return default value to avoid errors

    #     registers = self.modbus_fail_read_handler(modbus_register.L_FB_RPM, 2)

    #     # 🔍 Check if data reading fails
    #     if registers is None or len(registers) < 2:
    #         print("❌ Error: Unable to read RPM from the motor!")
    #         return 0.0, 0.0  # Return default value to avoid errors

    #     try:
    #         fb_L_rpm = np.int16(registers[0]) / 10.0
    #         fb_R_rpm = np.int16(registers[1]) / 10.0
    #         return fb_L_rpm, fb_R_rpm
    #     except (IndexError, TypeError) as e:
    #         print(f"❌ Error processing RPM data: {str(e)}")
    #         return 0.0, 0.0  # Return default value to avoid errors


    def get_rpm(self):
        if not self.is_connected():
            print("⚠️ Unable to get RPM speed: Modbus connection lost!")
            return 0.0, 0.0

        registers = self.modbus_fail_read_handler(modbus_register.L_FB_RPM, 2)
        if not registers or len(registers) < 2:
            print("❌ Error: Unable to read RPM from the motor!")
            return 0.0, 0.0

        try:
            l_raw, r_raw = registers[0], registers[1]
            fb_L_rpm = self._u16_to_s16(l_raw) / 10.0
            fb_R_rpm = self._u16_to_s16(r_raw) / 10.0
            return fb_L_rpm, fb_R_rpm
        except Exception as e:
            print(f"❌ Error processing RPM data: {e}")
            return 0.0, 0.0


    # def set_rpm(self, L_rpm, R_rpm):
    #     """Send RPM control command, ensure reconnection if connection is lost"""
        
    #     if not self.is_connected():  # Use the fixed is_connected() method
    #         print("⚠ Unable to send RPM command because Modbus is not connected!")
    #         return  # ❌ Do not send command if not connected
        
    #     L_rpm = max(min(L_rpm, self.param.max_rpm), -self.param.max_rpm)
    #     R_rpm = max(min(R_rpm, self.param.max_rpm), -self.param.max_rpm)

    #     left_bytes = self.int16Dec_to_int16Hex(L_rpm)
    #     right_bytes = self.int16Dec_to_int16Hex(R_rpm)

    #     try:
    #         self.client.write_registers(modbus_register.L_CMD_RPM, [left_bytes, right_bytes], unit=self.ID)
    #     except pymodbus.exceptions.ConnectionException as e:
    #         print(f"❌ Modbus connection error when sending set_rpm: {str(e)}")
    #     except Exception as e:
    #         print(f"❌ Unknown error when sending set_rpm: {str(e)}")

    def set_rpm(self, L_rpm, R_rpm):
        if not self.is_connected():
            print("⚠ Unable to send RPM command because Modbus is not connected!")
            return

        L_rpm = max(min(L_rpm, self.param.max_rpm), -self.param.max_rpm)
        R_rpm = max(min(R_rpm, self.param.max_rpm), -self.param.max_rpm)

        left_u16  = self._int16_to_u16(L_rpm)
        right_u16 = self._int16_to_u16(-R_rpm)

        try:
            self.client.write_registers(modbus_register.L_CMD_RPM, [left_u16, right_u16], unit=self.ID)
        except pymodbus.exceptions.ConnectionException as e:
            print(f"❌ Modbus connection error when sending set_rpm: {e}")
        except Exception as e:
            print(f"❌ Unknown error when sending set_rpm: {e}")



    def stop_motor_emergency(self):
        """Send Emergency stop command to stop the motor immediately"""
        try:
            if not self.is_connected():  # 🔍 Check connection status
                self.node.get_logger().warn("⚠ Modbus connection lost! Attempting to reconnect ...")
                for _ in range(5):  # 🔄 Thử lại tối đa 5 lần
                    if self.client.connect():
                        self.node.get_logger().info("✅ Modbus reconnection successful!")
                        break
                    time.sleep(0.5)  # ⏳ Wait reconnection successful

            if self.is_connected():  # 🔍 Only send command if reconnected successfully
                self.client.write_register(modbus_register.CONTROL_REG, modbus_register.EMER_STOP, unit=self.ID)
                self.node.get_logger().info("🛑 Emergency stop command received by motor!")
            else:
                self.node.get_logger().error("❌ Unable to reconnect Modbus. Please check the wiring or power source")

        except Exception as e:
            self.node.get_logger().error(f"❌ Error sending Emergency Stop {str(e)}")


    def clear_alarm(self):
        """Clear error alarms on the controller"""
        if not self.is_connected():
            self.node.get_logger().error("❌No connection! Cannot send clear_alarm() command.")
            return False

        result = self.client.write_register(modbus_register.CONTROL_REG, modbus_register.ALRM_CLR, unit=self.ID)

        if result.isError():
            self.node.get_logger().error("❌ Unable to clear error! Check the connection and retry.")
            return False

        self.node.get_logger().info("✅ Error cleared on the controller.")
        return True

    
    def get_linear_velocities(self):
        """Get the linear velocities of the two wheels (m/s)"""
        rpmL, rpmR = self.get_rpm()
        
        if rpmL is None or rpmR is None:
            self.node.get_logger().error("❌ Unable to get RPM from the motor.")
            return None, None

        # VL = self.rpm_to_linear(rpmL)
        # VR = self.rpm_to_linear(-rpmR)

        return self.rpm_to_linear(rpmL), self.rpm_to_linear(-rpmR)

    
    def map(self, val, in_min, in_max, out_min, out_max):
        """Map value from the range '[in_min, in_max]' sang '[out_min, out_max]'"""
        if in_min == in_max:
            self.node.get_logger().error("❌ Error! Cannot map when 'in_min == in_max' (divide by 0).")
            return None

        return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    # def get_wheels_travelled(self):
    #     """Get the distance travelled by the wheels (radian unit)"""
        
    #     registers = self.modbus_fail_read_handler(modbus_register.L_FB_POS_HI, 4)
    #     if registers is None or len(registers) < 4:
    #         self.node.get_logger().error("❌ Error reading wheel position data from Modbus!")
    #         return None, None

    #     try:
    #         l_pul_hi, l_pul_lo = registers[0], registers[1]
    #         r_pul_hi, r_pul_lo = registers[2], registers[3]

    #         # Combine registers into 32-bit value
    #         l_pulse = np.int32(((l_pul_hi & 0xFFFF) << 16) | (l_pul_lo & 0xFFFF))
    #         r_pulse = np.int32(((r_pul_hi & 0xFFFF) << 16) | (r_pul_lo & 0xFFFF))

    #         # Convert to angle (radian)
    #         l_travelled = ((float(l_pulse) / self.cpr) * self.travel_in_one_rev) / self.R_Wheel
    #         r_travelled = ((float(r_pulse) / self.cpr) * self.travel_in_one_rev) / self.R_Wheel
            
    #         return -l_travelled, r_travelled

    #     except OverflowError:
    #         self.node.get_logger().error("❌ Overflow error while calculating wheel position!")
    #         return None, None
    def get_wheels_travelled(self):
        regs = self.modbus_fail_read_handler(modbus_register.L_FB_POS_HI, 4)
        if not regs or len(regs) < 4:
            self.node.get_logger().error("❌ Error reading wheel position data from Modbus!")
            return None, None
        try:
            l_pulse = self._join_u16_to_s32(regs[0], regs[1])
            r_pulse = self._join_u16_to_s32(regs[2], regs[3])

            # -> rad: (pulses/CPR) * (2πR) / R = (pulses/CPR)*2π
            # travel_in_one_rev = 2πR  (như bạn đang làm) => công thức dưới OK
            l_travelled = ((float(l_pulse) / float(self.cpr)) * float(self.travel_in_one_rev)) / float(self.R_Wheel)
            r_travelled = ((float(r_pulse) / float(self.cpr)) * float(self.travel_in_one_rev)) / float(self.R_Wheel)
            return float(l_travelled), -float(r_travelled)
        except Exception as e:
            self.node.get_logger().error(f"❌ Overflow/compute error while calculating wheel position: {e}")
            return None, None

    # def get_wheels_tick(self):
    #     """Get the number of ticks from the left and right wheel encoders"""
        
    #     try:
    #         registers = self.modbus_fail_read_handler(modbus_register.L_FB_POS_HI, 4)
    #         if registers is None or len(registers) < 4:
    #             self.node.get_logger().error("❌ Error reading encoder data from Modbus!")
    #             return None, None

    #         l_pul_hi, l_pul_lo = registers[0], registers[1]
    #         r_pul_hi, r_pul_lo = registers[2], registers[3]

    #         # Combine register into 32-bit value
    #         l_tick = np.int32(((l_pul_hi & 0xFFFF) << 16) | (l_pul_lo & 0xFFFF))
    #         r_tick = np.int32(((r_pul_hi & 0xFFFF) << 16) | (r_pul_lo & 0xFFFF))

    #         return l_tick, r_tick

    #     except OverflowError:
    #         self.node.get_logger().error("❌ Overflow error while reading encoder tick!")
    #         return None, None

    #     except IndexError:
    #         self.node.get_logger().error("❌ Modbus data returned insufficient elements!")
    #         return None, None

    def get_wheels_tick(self):
        regs = self.modbus_fail_read_handler(modbus_register.L_FB_POS_HI, 4)
        if not regs or len(regs) < 4:
            self.node.get_logger().error("❌ Error reading encoder data from Modbus!")
            return None, None
        try:
            l_tick = self._join_u16_to_s32(regs[0], regs[1])
            r_tick = self._join_u16_to_s32(regs[2], regs[3])
            return int(l_tick), int(r_tick)
        except Exception as e:
            self.node.get_logger().error(f"❌ Overflow/compute error while reading encoder tick: {e}")
            return None, None

    def set_position_async_control(self):
        """Set asynchronous position control mode"""
        try:
            result = self.client.write_register(modbus_register.POS_CONTROL_TYPE, modbus_register.ASYNC, unit=self.ID)
            self.node.get_logger().info("✅ ASYNC mode set for the motor.")
            return result
        except Exception as e:
            self.node.get_logger().error(f"❌ Error setting ASYNC mode: {str(e)}")
            return None


    def move_left_wheel(self):
        """Send command to move the left wheel"""
        try:
            result = self.client.write_register(modbus_register.CONTROL_REG, modbus_register.POS_L_START, unit=self.ID)
            self.node.get_logger().info("✅ Left wheel move command sent.")
            return result
        except Exception as e:
            self.node.get_logger().error(f"❌ Error sending left wheel move command: {str(e)}")
            return None

    

    # def int16Dec_to_int16Hex(self, int16):
    #     """onvert 16-bit integer from decimal to hex"""
    #     try:
    #         int16 = np.int16(int16)  # Ensure valid value
    #         lo_byte = int16 & 0x00FF
    #         hi_byte = (int16 & 0xFF00) >> 8
    #         return (hi_byte << 8) | lo_byte
    #     except Exception as e:
    #         self.node.get_logger().error(f"❌ Error converting number {str(e)}")
    #         return None
    def int16Dec_to_int16Hex(self, val):
        try:
            # Trả về u16 để ghi xuống 1 thanh ghi 16-bit (pymodbus chấp nhận 0..65535)
            return self._int16_to_u16(val)
        except Exception as e:
            self.node.get_logger().error(f"❌ Error converting number {e}")
            return None



    def close_connect(self):
        """Close Modbus connection"""
        try:
            if self.client:
                self.client.close()
                print("✅ Closed Modbus connection.")
            else:
                print("⚠ No Modbus connection to close.")
        except Exception as e:
            print(f"❌ Error when closing Modbus connection: {str(e)}")

    def is_motor_enabled(self):
        """Check if the motor is in the 'enabled' state"""
        try:
            registers = self.modbus_fail_read_handler(modbus_register.CONTROL_REG, 1)
            if registers is None:
                self.node.get_logger().warn("⚠ Unable to check motor state!")
                return False
            return registers[0] == modbus_register.ENABLE
        except Exception as e:
            self.node.get_logger().error(f"❌ Error checking motor state: {str(e)}")
            return False


    def get_motor_status(self):
        """Get the full motor status (with error handling)"""
        try:
            return {
                "connected": self.is_connected(),
                "enabled": self.is_motor_enabled(),
                "mode": self.get_mode() if self.is_connected() else None,
                "rpm": self.get_rpm() if self.is_connected() else (None, None),
                "linear_velocity": self.get_linear_velocities() if self.is_connected() else (None, None),
                "position": self.get_wheels_travelled() if self.is_connected() else (None, None),
                "fault": self.get_fault_code() if self.is_connected() else (None, None),}
        
        except Exception as e:
            self.node.get_logger().error(f"❌ Error retrieving motor status: {str(e)}")
            return None
        
    def get_battery_voltage(self):
        """Read bus voltage (battery voltage) from the drive"""
        try:
            registers = self.client.read_holding_registers(0x20A1, 1, unit=self.ID)
            if registers.isError():
                print("❌ Error reading bus voltage!")
                return None
            voltage = registers.registers[0] * 0.01  # Convert to volts
            return voltage
        except Exception as e:
            print(f"❌ Error reading bus voltage: {str(e)}")
            return None


    def get_mode(self):
        """Get current control mode"""
        try:
            registers = self.modbus_fail_read_handler(modbus_register.OPR_MODE, 1)
            return registers[0] if registers else None
        except Exception as e:
            print(f"❌ Error reading control mode: {e}")
            return None
        
    def get_brake_state(self):
        """Check brake status"""
        try:
            result = self.client.read_holding_registers(0x201A, 1, unit=self.ID)
            if result and not result.isError():
                return "Closed" if result.registers[0] == 1 else "Open"
        except Exception as e:
            print(f"❌ Error reading brake status: {str(e)}")
        return None

        
    """
    These two functions are used to check the fault status of the left/right motor by reading data from L_FAULT (0x20A5) and R_FAULT (0x20A6).
    If the motor encounters a fault, it will return the corresponding error code.

    Possible faults include:
        0x0001 - Over Voltage (Too high voltage)
        0x0002 - Under Voltage (Too low voltage)
        0x0004 - Over Current (Too much current)
        0x0008 - Over Load (Too much load)
        0x0010 - Current Out of Tolerance (Current out of range)
        0x0020 - Encoder Error (Encoder malfunction)
        0x0040 - Motor Bad (Motor malfunction)
        0x0080 - Reference Voltage Error (Reference voltage error)
        0x0100 - EEPROM Error (EEPROM malfunction)
        0x0200 - Hall Sensor Error (Hall sensor malfunction)
        0x0400 - Motor Over Temperature (Overheating)
    """

    def troubleshooting_left(self):
        """Check left motor faults"""
        try:
            registers = self.modbus_fail_read_handler(modbus_register.L_FAULT, 12)
            if registers:
                return registers
            else:
                return None
        except Exception as e:
            print(f"❌ Error checking left motor: {e}")
            return None


    def troubleshooting_right(self):
        """🔍 Check right motor faults"""
        try:
            registers = self.modbus_fail_read_handler(modbus_register.R_FAULT, 12)
            if registers:
                return registers
            else:
                return None
        except Exception as e:
            print(f"❌ Error checking right motor: {e}")
            return None


    def get_motor_faults(self):
        """Get left/right motor fault codes"""
        try:
            result = self.client.read_holding_registers(0x20A5, 2, unit=self.ID)
            if result and not result.isError():
                L_fault = result.registers[0]
                R_fault = result.registers[1]
                return L_fault, R_fault
        except Exception as e:
            print(f"❌ Error reading fault status: {str(e)}")
        return None, None
    
    def get_motor_temperature(self):
        """Read motor temperature"""
        try:
            result = self.client.read_holding_registers(0x20A4, 1, unit=self.ID)
            if result and not result.isError():
                temp = result.registers[0]  # Unit: 1°C
                return temp
        except Exception as e:
            print(f"❌ Error reading motor temperature: {str(e)}")
        return None

    def get_driver_temperature(self):
        """Read driver temperature"""
        try:
            result = self.client.read_holding_registers(0x20B0, 1, unit=self.ID)
            if result and not result.isError():
                temp = result.registers[0] * 0.1  #  Unit: 0.1°C
                return temp
        except Exception as e:
            print(f"❌ Error reading driver temperature: {str(e)}")
        return None
    



## Common
CONTROL_REG                     = 0x200E # {0x05: emergency stop, 0x06: clear stop, 0x07: stop, 0x08: enable}
OPR_MODE                        = 0x200D     # control mode
L_ACL_TIME                      = 0x2080
R_ACL_TIME                      = 0x2081
L_DCL_TIME                      = 0x2082
R_DCL_TIME                      = 0x2083

# clear feedback position
CLR_FB_POS                      = 0x2005

## Velocity control (Speed RPM Control)
# Target velocity Range: --450~450/min

L_CMD_RPM                       = 0x2088
R_CMD_RPM                       = 0x2089



# Position control
L_FB_POS_HI                     = 0x20A7
POS_CONTROL_TYPE                = 0x200F  # Synchronous/asynchronous control status

L_MAX_RPM_POS                   = 0x208E # Not in use
L_CMD_REL_POS_HI                = 0x208A # Not in use


# Actual velocity  unit: 0.1r/min
L_FB_RPM                        = 0x20AB
R_FB_RPM                        = 0x20AC



########################
## Control CMDs (REG) ##
########################

EMER_STOP                       = 0x05 # emergency stop
ALRM_CLR                        = 0x06 # clear default 
DOWN_TIME                       = 0x07 # stop
ENABLE                          = 0x08    # enable

POS_SYNC                        = 0x10
POS_L_START                     = 0x11
POS_R_START                     = 0x12


####################
## Operation Mode ##
####################

POS_REL_CONTROL                 = 1
POS_ABS_CONTROL = 2
VEL_CONTROL                     = 3

ASYNC                           = 0
SYNC                            = 1


## Troubleshooting
L_FAULT                         = 0x20A5
R_FAULT                         = 0x20A6

#################
## Fault codes ##
#################

NO_FAULT                        = 0x0000       # No error
OVER_VOLT                       = 0x0001      # Over voltage
UNDER_VOLT                      = 0x0002     # Under voltage
OVER_CURR                       = 0x0004      # Over current
OVER_LOAD                       = 0x0008      # Over load
CURR_OUT_TOL                    = 0x0010   # Current out of tolerance
ENCOD_OUT_TOL                   = 0x0020  # Encoder out of tolerance
MOTOR_BAD                       = 0x0040      # Velocity out of tolerance
REF_VOLT_ERROR                  = 0x0080 # Reference voltage error
EEPROM_ERROR                    = 0x0100   # EEPROM error
WALL_ERROR                      = 0x0200     # Hall error
HIGH_TEMP                       = 0x0400      # Motor temperature over temperature

FAULT_LIST                      = [OVER_VOLT, UNDER_VOLT, OVER_CURR, OVER_LOAD, CURR_OUT_TOL, ENCOD_OUT_TOL, \
                                    MOTOR_BAD, REF_VOLT_ERROR, EEPROM_ERROR, WALL_ERROR, HIGH_TEMP]



from rclpy.node import Node

class NodeParameters:
    def __init__(self, node: Node):
        node.get_logger().info('Initializing parameters for ZLAC8015D motor drivers')

        node.declare_parameter(name='connection_type', value='rtu')
        node.declare_parameter(name='modbus_port', value='/dev/zlac_8015d')  # zlac_8015d / ttyUSB0
        node.declare_parameter(name='modbus_baudrate', value=115200)
        node.declare_parameter(name='joint_state_topic', value='/nhatbot/JointState')
        node.declare_parameter(name='wheel_rotation_topic', value='/nhatbot/wheel_rotational_vel')
        node.declare_parameter(name='zlac_status_topic', value='/nhatbot/zlac_status')
        node.declare_parameter(name='reset_encoder_service', value='/nhatbot/reset_feedback_position')
        node.declare_parameter(name='joint_state_frequency', value=0.1)
        node.declare_parameter(name='zlac_status_frequency', value=1.0)
        node.declare_parameter(name='set_accel_time', value=500)
        node.declare_parameter(name='set_decel_time', value=500)
        node.declare_parameter(name='set_operation_mode', value=3)
        node.declare_parameter(name='set_stop_rpm', value=0)
        node.declare_parameter(name='set_clear_wheel_encoders', value=3)
        node.declare_parameter(name='ignore_small_speed_threshold', value=0.5)
        node.declare_parameter(name='max_rpm', value=200)
        node.declare_parameter(name='travel_in_one_rev', value=0.336)
        node.declare_parameter(name='cpr', value=4096)
        node.declare_parameter(name='wheel_radius', value=0.0535)

        try:
            self.connection_type = node.get_parameter('connection_type').get_parameter_value().string_value
            self.modbus_port = node.get_parameter('modbus_port').get_parameter_value().string_value
            self.modbus_baudrate = node.get_parameter('modbus_baudrate').get_parameter_value().integer_value
            self.joint_state_topic = node.get_parameter('joint_state_topic').get_parameter_value().string_value
            self.wheel_rotation_topic = node.get_parameter('wheel_rotation_topic').get_parameter_value().string_value
            self.zlac_status_topic = node.get_parameter('zlac_status_topic').get_parameter_value().string_value
            self.reset_encoder_service = node.get_parameter('reset_encoder_service').get_parameter_value().string_value
            self.joint_state_frequency = node.get_parameter('joint_state_frequency').get_parameter_value().double_value
            self.zlac_status_frequency = node.get_parameter('zlac_status_frequency').get_parameter_value().double_value
            self.set_accel_time = node.get_parameter('set_accel_time').get_parameter_value().integer_value
            self.set_decel_time = node.get_parameter('set_decel_time').get_parameter_value().integer_value
            self.set_operation_mode = node.get_parameter('set_operation_mode').get_parameter_value().integer_value
            self.set_stop_rpm = node.get_parameter('set_stop_rpm').get_parameter_value().integer_value
            self.set_clear_wheel_encoders = node.get_parameter('set_clear_wheel_encoders').get_parameter_value().integer_value
            self.ignore_small_speed_threshold = node.get_parameter('ignore_small_speed_threshold').get_parameter_value().double_value
            self.max_rpm = node.get_parameter('max_rpm').get_parameter_value().integer_value
            self.travel_in_one_rev = node.get_parameter('travel_in_one_rev').get_parameter_value().double_value
            self.cpr = node.get_parameter('cpr').get_parameter_value().integer_value
            self.wheel_radius = node.get_parameter('wheel_radius').get_parameter_value().double_value


        except Exception as e:
            node.get_logger().warn('Could not get parameters...setting variables to default')
            node.get_logger().warn('Error: "%s"' % e)

