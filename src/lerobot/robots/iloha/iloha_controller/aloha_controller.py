import asyncio
from typing import Any, List, Optional

from .aloha_arm_controller import AlohaArm, AlohaArmController


class AlohaController:
    """
    ALOHA双腕ロボットの制御クラス
    左右のAlohaArmControllerを統合して制御
    """

    def __init__(
        self,
        right_robstride_port: str,
        left_robstride_port: str,
        right_dynamixel_port: str,
        left_dynamixel_port: str,
        right_robstride_constants: Optional[List[Any]] = None,
        right_dynamixel_constants: Optional[List[Any]] = None,
        left_robstride_constants: Optional[List[Any]] = None,
        left_dynamixel_constants: Optional[List[Any]] = None,
    ):
        """
        ALOHAコントローラーを初期化

        Args:
            right_robstride_port: 右アームRobStrideのポート名
            left_robstride_port: 左アームRobStrideのポート名
            right_dynamixel_port: 右アームDynamixelのポート名
            left_dynamixel_port: 左アームDynamixelのポート名
            right_robstride_constants: 右アーム RobStride設定リスト
            right_dynamixel_constants: 右アーム Dynamixel設定リスト
            left_robstride_constants: 左アーム RobStride設定リスト
            left_dynamixel_constants: 左アーム Dynamixel設定リスト
        """
        # デフォルト設定の読み込み
        if right_robstride_constants is None or right_dynamixel_constants is None:
            from .right_settings import (
                Dynamixel01Constants,
                Dynamixel02Constants,
                Dynamixel03Constants,
                Dynamixel04Constants,
                Robstride01Constants,
                Robstride02Constants,
                Robstride03Constants,
            )

            right_robstride_constants = [
                Robstride01Constants,
                Robstride02Constants,
                Robstride03Constants,
            ]
            right_dynamixel_constants = [
                Dynamixel01Constants,
                Dynamixel02Constants,
                Dynamixel03Constants,
                Dynamixel04Constants,
            ]

        if left_robstride_constants is None or left_dynamixel_constants is None:
            from .left_settings import (
                Dynamixel01Constants,
                Dynamixel02Constants,
                Dynamixel03Constants,
                Dynamixel04Constants,
                Robstride01Constants,
                Robstride02Constants,
                Robstride03Constants,
            )

            left_robstride_constants = [
                Robstride01Constants,
                Robstride02Constants,
                Robstride03Constants,
            ]
            left_dynamixel_constants = [
                Dynamixel01Constants,
                Dynamixel02Constants,
                Dynamixel03Constants,
                Dynamixel04Constants,
            ]

        # アームコントローラーのインスタンス
        self.right_arm_controller: Optional[AlohaArmController] = None
        self.left_arm_controller: Optional[AlohaArmController] = None

        # 初期化パラメータを保存
        self.right_params = {
            "robstride_port": right_robstride_port,
            "dynamixel_port": right_dynamixel_port,
            "robstride_constants": right_robstride_constants,
            "dynamixel_constants": right_dynamixel_constants,
        }
        self.left_params = {
            "robstride_port": left_robstride_port,
            "dynamixel_port": left_dynamixel_port,
            "robstride_constants": left_robstride_constants,
            "dynamixel_constants": left_dynamixel_constants,
        }

    async def _initialize_controllers(self) -> None:
        """両アームコントローラーを初期化（並列実行）"""
        try:
            print("🤖 ALOHA双腕コントローラー初期化中...")

            # 右アームと左アームのコントローラーを作成
            self.right_arm_controller = AlohaArmController(**self.right_params)
            self.left_arm_controller = AlohaArmController(**self.left_params)

            # 両アームを並列に初期化
            print("  両アーム並列初期化中...")
            await asyncio.gather(
                self.right_arm_controller.__aenter__(),
                self.left_arm_controller.__aenter__(),
            )

            print("✅ ALOHA双腕コントローラー初期化完了!")

        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            await self.disable()
            raise

    async def update_pos(self, right_arm: AlohaArm, left_arm: AlohaArm) -> None:
        """
        両アームの位置を一括更新（並列実行）

        Args:
            right_arm: 右アームの目標位置
            left_arm: 左アームの目標位置
        """
        if self.right_arm_controller is None or self.left_arm_controller is None:
            raise RuntimeError("コントローラーが初期化されていません")

        # 両アームを並列更新
        await asyncio.gather(
            self.right_arm_controller.update_pos(right_arm),
            self.left_arm_controller.update_pos(left_arm),
        )

    async def update_motor_pos(
        self, arm: str, motor_num: int, target_pos: float
    ) -> None:
        """
        特定のアームの特定のモーターの位置を更新

        Args:
            arm: アーム指定 ("right" または "left")
            motor_num: モーター番号 (1-7)
            target_pos: 目標位置 (radian)
        """
        if self.right_arm_controller is None or self.left_arm_controller is None:
            raise RuntimeError("コントローラーが初期化されていません")

        if arm == "right":
            await self.right_arm_controller.update_motor_pos(motor_num, target_pos)
        elif arm == "left":
            await self.left_arm_controller.update_motor_pos(motor_num, target_pos)
        else:
            raise ValueError("armは'right'または'left'を指定してください")

    async def set_gripper_current(self, arm: str, current_mA: float) -> None:
        """
        指定したアームのグリッパーの電流を設定

        Args:
            arm: アーム指定 ("right" または "left")
            current_mA: 目標電流 (mA)
        """
        if self.right_arm_controller is None or self.left_arm_controller is None:
            raise RuntimeError("コントローラーが初期化されていません")

        if arm == "right":
            await self.right_arm_controller.set_gripper_current(current_mA)
        elif arm == "left":
            await self.left_arm_controller.set_gripper_current(current_mA)
        else:
            raise ValueError("armは'right'または'left'を指定してください")

    async def disable(self) -> None:
        """全モーターを初期位置に戻し、接続を切断（並列実行）"""
        print("🔄 ALOHA双腕コントローラー終了処理中...")

        # 右アームと左アームの終了処理を並列実行
        tasks = []

        if self.right_arm_controller:

            async def disable_right() -> None:
                try:
                    print("  右アーム終了処理中...")
                    await self.right_arm_controller.disable()
                except Exception as e:
                    print(f"⚠️ 右アーム終了処理エラー: {e}")
                finally:
                    self.right_arm_controller = None

            tasks.append(disable_right())

        if self.left_arm_controller:

            async def disable_left() -> None:
                try:
                    print("  左アーム終了処理中...")
                    await self.left_arm_controller.disable()
                except Exception as e:
                    print(f"⚠️ 左アーム終了処理エラー: {e}")
                finally:
                    self.left_arm_controller = None

            tasks.append(disable_left())

        if tasks:
            await asyncio.gather(*tasks)

        print("✅ ALOHA双腕コントローラー終了完了")

    async def __aenter__(self) -> "AlohaController":
        await self._initialize_controllers()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disable()
