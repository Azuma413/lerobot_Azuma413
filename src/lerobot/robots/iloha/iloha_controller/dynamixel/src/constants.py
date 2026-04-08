import enum
from dataclasses import dataclass


class DynamixelSeries(enum.Enum):
    """Dynamixelシリーズの列挙型"""

    XM430_W350 = 1
    XM540_W270 = 2
    # 他のシリーズもここに追加可能


@dataclass
class Param:
    """
    Dynamixel XM430-W350-Rのパラメータ
    """

    # Control Table Addresses #
    ADDR_TORQUE_ENABLE = 64
    ADDR_OPERATING_MODE = 11

    # Position Control
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132

    # Velocity Control
    ADDR_GOAL_VELOCITY = 104
    ADDR_PRESENT_VELOCITY = 128

    # PWM Control
    ADDR_GOAL_PWM = 100
    ADDR_PRESENT_PWM = 124

    # Current Control
    ADDR_GOAL_CURRENT = 102
    ADDR_PRESENT_CURRENT = 126
    ###############################

    # Dynamixel Parameters #
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    RESOLUTION = 4096  # 0-4095 (12-bit)
    PULSE_PER_REVOLUTION = 4096  # 1回転あたりのパルス数

    # Current conversion parameters
    CURRENT_UNIT = 2.69  # mA per unit (XM430-W350)
    MAX_CURRENT = 1193  # mA (maximum current)


class OperatingMode(enum.Enum):
    """
    Dynamixelのオペレーティングモード
    """

    CURRENT_CONTROL = 0
    VELOCITY_CONTROL = 1
    POSITION_CONTROL = 3
    EXTENDED_POSITION_CONTROL = 4
    CURRENT_BASED_POSITION_CONTROL = 5
    PWM_CONTROL = 16


@dataclass
class ControlParams:
    """Dynamixelモーターの制御パラメータ"""

    # リミット値
    max_position: int = 4095
    min_position: int = 0

    # 制御パラメータ
    ctrl_mode: OperatingMode = OperatingMode.POSITION_CONTROL
    offset: int = 0  # オフセット値（デフォルトは0）


class DynamixelParams:
    """Dynamixelモーターの基本クラス"""

    def __init__(self, series: DynamixelSeries):
        self.series = series
        # 他の初期化コード

    @property
    def param(self) -> Param:
        """Dynamixelシリーズに応じたコントロールテーブルを返す"""
        if self.series == DynamixelSeries.XM430_W350:
            return Param()
        elif self.series == DynamixelSeries.XM540_W270:
            return Param()
        else:
            raise ValueError("Unsupported Dynamixel series")


class ProtocolVersion(enum.Enum):
    """Dynamixelのプロトコルバージョン"""

    V1_0 = 1.0
    V2_0 = 2.0


class Baudrate(enum.Enum):
    """Dynamixelの通信速度（ボーレート）"""

    BAUD_57600 = 57600
