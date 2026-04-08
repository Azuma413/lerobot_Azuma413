from dataclasses import dataclass

from .dynamixel.src.constants import (
    ControlParams,
    DynamixelSeries,
    OperatingMode,
)


@dataclass
class Robstride01Constants:
    """根本のRobStrideの定数"""

    ID = 1
    DEFAULT_OFFSET = 0.0  # デフォルトオフセット [rad]


@dataclass
class Robstride02Constants:
    """2番目のRobStrideの定数"""

    ID = 2
    DEFAULT_OFFSET = 0.0  # デフォルトオフセット [rad]


@dataclass
class Robstride03Constants:
    """3番目のRobStrideの定数"""

    ID = 3
    DEFAULT_OFFSET = 0.0  # デフォルトオフセット [rad]


@dataclass
class Dynamixel01Constants:
    """根本のDynamixelの定数"""

    SERIES = DynamixelSeries.XM540_W270
    ID = 1
    CONTROL_PARAMS = ControlParams(
        max_position=2*4096 - 1,
        min_position=-4096,
        ctrl_mode=OperatingMode.EXTENDED_POSITION_CONTROL,
        offset=int(4096 / 2),
    )


@dataclass
class Dynamixel02Constants:
    """2番目のDynamixelの定数"""

    SERIES = DynamixelSeries.XM540_W270
    ID = 2
    CONTROL_PARAMS = ControlParams(
        max_position=4095,
        min_position=0,
        ctrl_mode=OperatingMode.POSITION_CONTROL,
        offset=int(4096 / 2),
    )


@dataclass
class Dynamixel03Constants:
    """3番目のDynamixelの定数"""

    SERIES = DynamixelSeries.XM430_W350
    ID = 3
    CONTROL_PARAMS = ControlParams(
        max_position=4095,
        min_position=0,
        ctrl_mode=OperatingMode.POSITION_CONTROL,
        offset=int(4096 / 2),
    )


@dataclass
class Dynamixel04Constants:
    """4番目のDynamixelの定数"""

    SERIES = DynamixelSeries.XM430_W350
    ID = 4
    CONTROL_PARAMS = ControlParams(
        max_position=4095,
        min_position=0,
        ctrl_mode=OperatingMode.CURRENT_BASED_POSITION_CONTROL,
        offset=int(4096 / 2),
    )
