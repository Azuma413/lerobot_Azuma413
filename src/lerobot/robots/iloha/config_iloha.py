from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig

@RobotConfig.register_subclass("my_aloha")
@dataclass
class IlohaConfig(RobotConfig):
    right_robstride_port: str
    left_robstride_port: str
    right_dynamixel_port: str
    left_dynamixel_port: str
    enable_right_arm: bool = True
    enable_left_arm: bool = True
    max_relative_target_1: float = 0.1 # radians
    max_relative_target_2: float = 0.1 # radians
    max_relative_target_3: float = 0.1 # radians
    max_relative_target_4: float = 0.1 # radians
    max_relative_target_5: float = 0.1 # radians
    max_relative_target_6: float = 0.1 # radians
    current_limit_gripper_R: float = 0.3 # Amperes
    current_limit_gripper_L: float = 0.3 # Amperes
