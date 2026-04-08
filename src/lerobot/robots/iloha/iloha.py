from lerobot.robots.iloha.config_iloha import IlohaConfig
import numpy as np
from typing import Any
import asyncio
from .iloha_controller.aloha_controller import AlohaController, AlohaArm

JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]

class Iloha():
    config_class = IlohaConfig
    name = "iloha"

    def __init__(
        self,
        config: IlohaConfig,
        debug: bool = False,
        cameras: dict | None = None,
    ):
        # super().__init__(config)
        self.config = config
        self.debug = debug
        self.aloha = None
        self.cameras = cameras if cameras is not None else {}
        # old_action を ndarray で管理（14要素：L側7要素 + R側7要素）
        self.old_action = np.zeros(14, dtype=np.float32)
        # 指数平滑化フィルタの設定
        self.filter_alpha = 0.5
        self.filtered_joint_angles = None

    @property
    def observation_features(self) -> dict:
        """
        観測データの構造を定義
        - 画像: 各カメラごとにLeRobotのデータセット形式で定義
        - 関節角度: 状態ベクトルとして定義
        """
        features = {}
        # カメラ画像の特徴 - データセット形式では (height, width, channels)
        for name in self.cameras.keys():
            features[name] = (480, 640, 3)
        # 関節角度の特徴 - 状態ベクトルとして定義
        for joint_name in JOINT_NAMES:
            features[joint_name] = float
        return features

    @property
    def action_features(self) -> dict:
        """
        アクションデータの構造を定義
        - 14個の関節角度目標値
        """
        features = {}
        for joint_name in JOINT_NAMES:
            features[joint_name] = float
        return features

    def get_observation(self) -> dict[str, Any]:
        """
        現在の観測データを取得
        - カメラ画像: そのまま（H, W, C形式で取得、データセット保存時に適切に処理される）
        - 関節角度: "state"という単一のキーで14要素のベクトルとして返す
        """
        obs = {}
        for name, camera in self.cameras.items():
            obs[name] = camera.read()
        for i, joint_name in enumerate(JOINT_NAMES):
            obs[joint_name] = self.old_action[i]
        return obs

    async def connect(self) -> None:
        if not self.debug:
            self.aloha = AlohaController(
                self.config.right_robstride_port,
                self.config.left_robstride_port,
                self.config.right_dynamixel_port,
                self.config.left_dynamixel_port,
            )
            # AlohaControllerを非同期で初期化
            await self.aloha.__aenter__()
            # グリッパー電流を設定
            await self.aloha.set_gripper_current("left", self.config.current_limit_gripper_L*1000)
            await self.aloha.set_gripper_current("right", self.config.current_limit_gripper_R*1000)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        action_L = AlohaArm(
            motor1=action.get(JOINT_NAMES[0], self.old_action_L.motor1),
            motor2=action.get(JOINT_NAMES[1], self.old_action_L.motor2),
            motor3=action.get(JOINT_NAMES[2], self.old_action_L.motor3),
            motor4=action.get(JOINT_NAMES[3], self.old_action_L.motor4),
            motor5=action.get(JOINT_NAMES[4], self.old_action_L.motor5),
            motor6=action.get(JOINT_NAMES[5], self.old_action_L.motor6),
            motor7=-action.get(JOINT_NAMES[6], self.old_action_L.motor7)*np.pi/3,
        )
        action_R = AlohaArm(
            motor1=action.get(JOINT_NAMES[7], self.old_action_R.motor1),
            motor2=action.get(JOINT_NAMES[8], self.old_action_R.motor2),
            motor3=action.get(JOINT_NAMES[9], self.old_action_R.motor3),
            motor4=action.get(JOINT_NAMES[10], self.old_action_R.motor4),
            motor5=action.get(JOINT_NAMES[11], self.old_action_R.motor5),
            motor6=action.get(JOINT_NAMES[12], self.old_action_R.motor6),
            motor7=-action.get(JOINT_NAMES[13], self.old_action_R.motor7)*np.pi/3,
        )
        asyncio.run(self.aloha.update_pos(action_R, action_L)) # ループが既にある場合はバグるかも
        return action

    async def async_send_action(self, action: np.ndarray, use_relative=False, use_filter=True, use_unwrap=True) -> dict[str, float]:
        # 1. unwrap 処理（ndarray で実行）
        if use_unwrap:
            unwrapped_action = self._unwrap_angle_target(action, self.old_action)
        else:
            unwrapped_action = action.copy()
        
        # 2. use_relative=True の場合にのみ、変化量を制限する（ndarray で実行）
        if use_relative:
            limited_action = self._limit_relative_target(unwrapped_action, self.old_action)
        else:
            limited_action = unwrapped_action
        
        # 3. use_filter=True の場合、指数平滑化フィルタを適用（ndarray で実行）
        if use_filter:
            if self.filtered_joint_angles is None:
                # 初回はそのまま使用
                self.filtered_joint_angles = limited_action.copy()
            else:
                # フィルタリング
                self.filtered_joint_angles = (
                    self.filter_alpha * limited_action +
                    (1 - self.filter_alpha) * self.filtered_joint_angles
                )
            final_action = self.filtered_joint_angles
        else:
            final_action = limited_action
        
        # 4. ndarray を AlohaArm に変換（update_pos 用）
        final_action_L = AlohaArm(
            motor1=float(final_action[0]),
            motor2=float(final_action[1]),
            motor3=float(final_action[2]),
            motor4=float(final_action[3]),
            motor5=float(final_action[4]),
            motor6=float(final_action[5]),
            motor7=-float(final_action[6])*np.pi/3,
        )
        final_action_R = AlohaArm(
            motor1=float(final_action[7]),
            motor2=float(final_action[8]),
            motor3=float(final_action[9]),
            motor4=float(final_action[10]),
            motor5=float(final_action[11]),
            motor6=float(final_action[12]),
            motor7=-float(final_action[13])*np.pi/3,
        )
        # 5. 最終的なアクションをロボットに送信し、状態を更新する
        if not self.debug:
            await self.aloha.update_pos(final_action_R, final_action_L)
        else:
            # print(f"L: {final_action_L.motor4*180/np.pi:.3f} R: {final_action_R.motor4*180/np.pi:.3f}")
            pass
        self.old_action = final_action.copy()
        return action

    def _unwrap_angle_target(self, current: np.ndarray, old: np.ndarray) -> np.ndarray:
        """
        新しい目標角度(current)を、古い角度(old)に最も近い連続的な値に変換（アンラップ）します。
        ndarray版：14要素の配列を処理
        """
        unwrapped = current.copy()
        # グリッパー以外の関節（インデックス0-5と7-12）に対してアンラップを適用
        # グリッパー（インデックス6と13）は除外
        for i in range(14):
            if i == 6 or i == 13:  # グリッパーはスキップ
                continue
            delta = current[i] - old[i]
            if delta > np.pi:
                unwrapped[i] -= 2 * np.pi
            elif delta < -np.pi:
                unwrapped[i] += 2 * np.pi
        return unwrapped

    def _limit_relative_target(self, current: np.ndarray, old: np.ndarray) -> np.ndarray:
        """
        連続的な角度(current)と古い角度(old)の差分を計算し、最大変化量で制限します。
        この関数は、入力角度が既にアンラップされていることを前提とします。
        ndarray版：14要素の配列を処理
        """
        limited = current.copy()
        # 最大変化量のリスト（L側7要素 + R側7要素）
        max_deltas = [
            self.config.max_relative_target_1,  # L motor1
            self.config.max_relative_target_2,  # L motor2
            self.config.max_relative_target_3,  # L motor3
            self.config.max_relative_target_4,  # L motor4
            self.config.max_relative_target_5,  # L motor5
            self.config.max_relative_target_6,  # L motor6
            0,  # L gripper（制限なし）
            self.config.max_relative_target_1,  # R motor1
            self.config.max_relative_target_2,  # R motor2
            self.config.max_relative_target_3,  # R motor3
            self.config.max_relative_target_4,  # R motor4
            self.config.max_relative_target_5,  # R motor5
            self.config.max_relative_target_6,  # R motor6
            0,  # R gripper（制限なし）
        ]
        
        for i in range(14):
            if i == 6 or i == 13:  # グリッパーはスキップ
                continue
            delta = current[i] - old[i]
            if abs(delta) > max_deltas[i]:
                delta = max_deltas[i] * np.sign(delta)
            limited[i] = old[i] + delta
        
        return limited


    async def disconnect(self):
        if self.aloha is not None:
            await self.aloha.disable()
            self.aloha = None
