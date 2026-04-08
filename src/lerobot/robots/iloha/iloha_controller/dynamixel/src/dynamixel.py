import asyncio
import logging
from logging import Formatter, StreamHandler, getLogger
from typing import Any

from dynamixel_sdk import GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler

from .constants import (
    Baudrate,
    ControlParams,
    DynamixelParams,
    DynamixelSeries,
    OperatingMode,
    ProtocolVersion,
)

# Logger configuration
logger = getLogger(__name__)
logger.setLevel(logging.ERROR)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setLevel(logging.ERROR)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)


# Byte conversion helper functions
def int_to_1byte(value: int) -> list[int]:
    """1バイトの整数をバイト配列に変換します。"""
    return [value & 0xFF]


def int_to_2byte_list(value: int) -> list[int]:
    """2バイトの整数をバイト配列に変換します。"""
    # 負の値を16bit符号なし整数として扱う
    if value < 0:
        value += 65536
    return [value & 0xFF, (value >> 8) & 0xFF]


def int_to_4byte_list(value: int) -> list[int]:
    """4バイトの整数をバイト配列に変換します (SDKのサンプルに準拠)。"""
    # 負の値を32bit符号なし整数として扱う
    if value < 0:
        value += 4294967296
    return [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]


def bytes_to_4byte_int(value_bytes: bytes) -> int:
    """4バイトのバイト配列を符号付き整数に変換します。"""
    value = int.from_bytes(value_bytes, byteorder="little", signed=True)
    return value


class Dynamixel:
    def __init__(self, series: DynamixelSeries, id: int, param: ControlParams):
        self.dynamixel_params: DynamixelParams = DynamixelParams(series)
        self.control_params: ControlParams = param
        self.id: int = id


class DynamixelController:
    def __init__(
        self,
        port: str,
        motors: list[Dynamixel],
        protocol_version: ProtocolVersion = ProtocolVersion.V2_0,
        baudrate: Baudrate = Baudrate.BAUD_57600,
    ):
        self.port = port
        self.motors = {motor.id: motor for motor in motors}  # IDをキーとする辞書
        self.portHandler = PortHandler(self.port)
        self.protocol_version = protocol_version.value
        self.packetHandler = PacketHandler(self.protocol_version)
        self.baudrate = baudrate

        # Group ハンドラの初期化
        if not motors:
            raise ValueError("Motor list cannot be empty.")

        # 全モーター共通のパラメータを取得 (最初のモーターを代表とする)
        param = motors[0].dynamixel_params.param
        self.param = param

        # --- GroupSyncWrite ハンドラ ---
        # 目標位置 (4byte) のための GroupSyncWrite
        self.groupWriteGoalPosition = GroupSyncWrite(
            self.portHandler, self.packetHandler, param.ADDR_GOAL_POSITION, 4
        )
        # 目標速度 (4byte) のための GroupSyncWrite
        self.groupWriteGoalVelocity = GroupSyncWrite(
            self.portHandler, self.packetHandler, param.ADDR_GOAL_VELOCITY, 4
        )
        # トルク有効 (1byte) のための GroupSyncWrite
        self.groupWriteTorqueEnable = GroupSyncWrite(
            self.portHandler, self.packetHandler, param.ADDR_TORQUE_ENABLE, 1
        )
        # オペレーティングモード (1byte) のための GroupSyncWrite
        self.groupWriteOperatingMode = GroupSyncWrite(
            self.portHandler, self.packetHandler, param.ADDR_OPERATING_MODE, 1
        )
        # 目標電流 (2byte) のための GroupSyncWrite
        self.groupWriteGoalCurrent = GroupSyncWrite(
            self.portHandler, self.packetHandler, param.ADDR_GOAL_CURRENT, 2
        )

        # --- GroupSyncRead ハンドラ ---
        # 現在位置 (4byte) のための GroupSyncRead
        self.groupReadPresentPosition = GroupSyncRead(
            self.portHandler, self.packetHandler, param.ADDR_PRESENT_POSITION, 4
        )

    def _read_2byte(self, motor_id: int, address: int) -> tuple[int, bool]:
        """指定したアドレスから2バイトのデータを読み取ります。"""
        value, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(
            self.portHandler, motor_id, address
        )
        if dxl_comm_result != 0:
            logger.error(self.packetHandler.getTxRxResult(dxl_comm_result))
            return 0, False
        if dxl_error != 0:
            logger.error(self.packetHandler.getRxPacketError(dxl_error))
            return 0, False

        # 符号付き16bit整数に対応
        if value > 32767:
            value = value - 65536
        return value, True

    def radian_to_pulse(self, radian: float, pulse_per_revolution: int) -> int:
        """radian値をパルス値に変換します."""
        pulse = int((radian / (2 * 3.14159265359)) * pulse_per_revolution)
        return pulse

    # 非同期・一斉送受信メソッド

    async def set_goal_positions_async(self, positions: dict[int, int]) -> bool:
        """複数のモーターに目標位置(パルス値)を一斉送信します。"""
        self.groupWriteGoalPosition.clearParam()
        for motor_id, position in positions.items():
            if motor_id not in self.motors:
                logger.warning(f"Motor ID {motor_id} not in controller.")
                continue

            offset = self.motors[motor_id].control_params.offset
            value_with_offset = position + offset
            param_bytes = int_to_4byte_list(value_with_offset)

            if not self.groupWriteGoalPosition.addParam(motor_id, param_bytes):
                logger.error(f"Failed to add param for motor ID {motor_id}")
                return False

        # txPacketはブロッキングI/Oなので、別スレッドで実行
        dxl_comm_result = await asyncio.to_thread(self.groupWriteGoalPosition.txPacket)

        if dxl_comm_result != 0:
            logger.error(self.packetHandler.getTxRxResult(dxl_comm_result))
            return False

        logger.info(f"Set goal positions for {len(positions)} motors.")
        return True

    async def get_present_positions_async(self) -> dict[int, int | None]:
        """複数のモーターの現在位置(パルス値)を一斉受信します。"""
        self.groupReadPresentPosition.clearParam()
        motor_ids = list(self.motors.keys())
        for motor_id in motor_ids:
            if not self.groupReadPresentPosition.addParam(motor_id):
                logger.error(f"Failed to add param for motor ID {motor_id}")
                return {mid: None for mid in motor_ids}

        # txRxPacketはブロッキングI/Oなので、別スレッドで実行
        dxl_comm_result = await asyncio.to_thread(
            self.groupReadPresentPosition.txRxPacket
        )

        if dxl_comm_result != 0:
            logger.error(self.packetHandler.getTxRxResult(dxl_comm_result))
            return {mid: None for mid in motor_ids}

        results = {}
        for motor_id in motor_ids:
            # データが利用可能かチェック
            if self.groupReadPresentPosition.isAvailable(
                motor_id, self.param.ADDR_PRESENT_POSITION, 4
            ):
                # データを取得
                raw_value = self.groupReadPresentPosition.getData(
                    motor_id, self.param.ADDR_PRESENT_POSITION, 4
                )
                # 4バイトの生データを符号付き整数に変換
                value = bytes_to_4byte_int(raw_value.to_bytes(4, "little"))

                offset = self.motors[motor_id].control_params.offset
                results[motor_id] = value - offset
            else:
                logger.warning(f"Failed to get data for motor ID {motor_id}")
                results[motor_id] = None

        return results

    async def set_torque_enable_async(self, torques: dict[int, bool]) -> bool:
        """複数のモーターのトルクON/OFFを一斉送信します。"""
        self.groupWriteTorqueEnable.clearParam()
        for motor_id, enable in torques.items():
            if motor_id not in self.motors:
                continue

            value = self.param.TORQUE_ENABLE if enable else self.param.TORQUE_DISABLE
            param_bytes = int_to_1byte(value)

            if not self.groupWriteTorqueEnable.addParam(motor_id, param_bytes):
                logger.error(f"Failed to add param for motor ID {motor_id}")
                return False

        dxl_comm_result = await asyncio.to_thread(self.groupWriteTorqueEnable.txPacket)

        if dxl_comm_result != 0:
            logger.error(self.packetHandler.getTxRxResult(dxl_comm_result))
            return False

        logger.info(f"Set torque enable for {len(torques)} motors.")
        return True

    async def set_goal_velocities_async(self, velocities: dict[int, int]) -> bool:
        """複数のモーターに目標速度を一斉送信します。"""
        self.groupWriteGoalVelocity.clearParam()
        for motor_id, velocity in velocities.items():
            if motor_id not in self.motors:
                logger.warning(f"Motor ID {motor_id} not in controller.")
                continue

            param_bytes = int_to_4byte_list(velocity)

            if not self.groupWriteGoalVelocity.addParam(motor_id, param_bytes):
                logger.error(f"Failed to add param for motor ID {motor_id}")
                return False

        dxl_comm_result = await asyncio.to_thread(self.groupWriteGoalVelocity.txPacket)

        if dxl_comm_result != 0:
            logger.error(self.packetHandler.getTxRxResult(dxl_comm_result))
            return False

        logger.info(f"Set goal velocities for {len(velocities)} motors.")
        return True

    async def set_operating_modes_async(self, modes: dict[int, OperatingMode]) -> bool:
        """複数のモーターのオペレーティングモードを一斉送信します。"""
        self.groupWriteOperatingMode.clearParam()
        for motor_id, mode in modes.items():
            if motor_id not in self.motors:
                continue

            param_bytes = int_to_1byte(mode.value)
            if not self.groupWriteOperatingMode.addParam(motor_id, param_bytes):
                logger.error(f"Failed to add param for motor ID {motor_id} (Mode)")
                return False

        dxl_comm_result = await asyncio.to_thread(self.groupWriteOperatingMode.txPacket)

        if dxl_comm_result != 0:
            logger.error(self.packetHandler.getTxRxResult(dxl_comm_result))
            return False

        logger.info(f"Set operating modes for {len(modes)} motors.")
        return True

    async def set_goal_currents_async(self, currents: dict[int, int]) -> bool:
        """複数のモーターに目標電流(パルス値)を一斉送信します。"""
        self.groupWriteGoalCurrent.clearParam()
        for motor_id, current in currents.items():
            if motor_id not in self.motors:
                continue

            param_bytes = int_to_2byte_list(current)
            if not self.groupWriteGoalCurrent.addParam(motor_id, param_bytes):
                logger.error(f"Failed to add param for motor ID {motor_id} (Current)")
                return False

        dxl_comm_result = await asyncio.to_thread(self.groupWriteGoalCurrent.txPacket)

        if dxl_comm_result != 0:
            logger.error(self.packetHandler.getTxRxResult(dxl_comm_result))
            return False

        logger.info(f"Set goal currents for {len(currents)} motors.")
        return True

    async def set_position_and_current_goals_async(
        self, goals: dict[int, tuple[int, int]]
    ) -> bool:
        """
        複数のモーターに「目標位置」と「目標電流」を一斉送信します。
        goals: { motor_id: (position, current) }

        注意: ADDR_GOAL_CURRENTとADDR_GOAL_POSITIONは連続していないため、
        2回に分けてSyncWriteで送信します。
        """
        # 1. 先に目標電流を設定
        currents = {motor_id: current for motor_id, (_, current) in goals.items()}
        if not await self.set_goal_currents_async(currents):
            logger.error("Failed to set goal currents")
            return False

        # 2. 次に目標位置を設定
        positions = {motor_id: position for motor_id, (position, _) in goals.items()}
        if not await self.set_goal_positions_async(positions):
            logger.error("Failed to set goal positions")
            return False

        logger.info(f"Set position and current goals for {len(goals)} motors.")
        return True

    async def set_position_and_current_goals_rad_async(
        self, goals: dict[int, tuple[float, int]]
    ) -> bool:
        """
        複数のモーターに「目標位置(radian)」と「目標電流」をBulkWriteで一斉送信します。
        goals: { motor_id: (position_rad, current) }
        """
        # ラジアン値をパルス値に変換
        pulse_goals = {}
        for motor_id, (position_rad, current) in goals.items():
            if motor_id not in self.motors:
                continue
            pulse_position = self.radian_to_pulse(
                position_rad,
                self.motors[motor_id].dynamixel_params.param.PULSE_PER_REVOLUTION,
            )
            pulse_goals[motor_id] = (pulse_position, current)
            logger.info(
                f"Motor ID {motor_id}: {position_rad:.3f} rad -> {pulse_position} pulse, current: {current} units"
            )

        # パルス値を使って既存のメソッドを呼び出す
        return await self.set_position_and_current_goals_async(pulse_goals)

    # 非同期接続・切断メソッド

    async def connect_async(self) -> bool:
        """接続処理を非同期化"""
        logger.info(f"Connecting to port {self.port} at {self.baudrate.value} bps...")
        if not await asyncio.to_thread(self.portHandler.openPort):
            logger.error("Failed to open the port.")
            return False
        if not await asyncio.to_thread(
            self.portHandler.setBaudRate, self.baudrate.value
        ):
            logger.error("Failed to change the baudrate.")
            return False

        # 接続確認 (ここは個別に実行)
        for motor_id in self.motors.keys():
            # _read_2byte はブロッキングなのでラップする
            value, success = await asyncio.to_thread(
                self._read_2byte,
                motor_id,
                self.param.ADDR_PRESENT_POSITION,
            )
            if not success:
                logger.error(f"Failed to connect to motor ID {motor_id}")
                return False
            logger.info(f"Successfully connected to motor ID {motor_id}")

        logger.info("Successfully connected to all motors.")
        return True

    async def disconnect_async(self) -> None:
        """切断処理を非同期化"""
        if self.portHandler.is_open:
            await asyncio.to_thread(self.portHandler.closePort)
            logger.info("Serial port closed.")

    async def __aenter__(self) -> "DynamixelController":
        """with構文の開始時に接続と全モーターのトルクONを行います。"""
        if not await self.connect_async():
            raise IOError("Failed to connect to Dynamixel.")

        # オペレーティングモードを一斉設定
        # (注: モード設定はトルクOFF中に行う必要があります)
        all_modes = {
            motor_id: motor.control_params.ctrl_mode
            for motor_id, motor in self.motors.items()
        }
        if not await self.set_operating_modes_async(all_modes):
            await self.disconnect_async()
            raise IOError("Failed to set operating mode for all motors.")

        logger.info("All motors operating modes set.")

        # 全モーターのトルクを一斉に有効化
        all_torque_on = {motor_id: True for motor_id in self.motors.keys()}
        if not await self.set_torque_enable_async(all_torque_on):
            await self.disconnect_async()
            raise IOError("Failed to enable torque for all motors.")

        logger.info("All motors torque enabled.")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """with構文の終了時に全モーターを停止し、トルクOFFと切断を行います。"""
        logger.info("Safely shutting down...")

        # 速度制御モードのモーターがあれば、速度0を一斉送信
        velocity_motors = {
            motor_id: 0
            for motor_id, motor in self.motors.items()
            if motor.control_params.ctrl_mode == OperatingMode.VELOCITY_CONTROL
        }
        if velocity_motors:
            await self.set_goal_velocities_async(velocity_motors)
            await asyncio.sleep(0.2)

        # 全てのモーターのトルクを一斉に無効化
        all_torque_off = {motor_id: False for motor_id in self.motors.keys()}
        await self.set_torque_enable_async(all_torque_off)

        await asyncio.sleep(0.1)
        await self.disconnect_async()
