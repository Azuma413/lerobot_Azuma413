import asyncio
import math

from ..aloha_arm_controller import AlohaArm, AlohaArmController
from ..left_settings import (
    Dynamixel01Constants,
    Dynamixel02Constants,
    Dynamixel03Constants,
    Dynamixel04Constants,
    Robstride01Constants,
    Robstride02Constants,
    Robstride03Constants,
)

# from kvt_aloha.right_settings import (
#     Dynamixel01Constants,
#     Dynamixel02Constants,
#     Dynamixel03Constants,
#     Dynamixel04Constants,
#     Robstride01Constants,
#     Robstride02Constants,
#     Robstride03Constants,
# )


# --- 設定項目 ---
# 各ポート名をご自身の環境に合わせて変更してください
# ROBSTRIDE_PORT = "COM3"
# DYNAMIXEL_PORT = "COM4"
ROBSTRIDE_PORT = "COM17"
DYNAMIXEL_PORT = "COM7"
# -----------------

robstride_constants = [
    Robstride01Constants,
    Robstride02Constants,
    Robstride03Constants,
]

dynamixel_constants = [
    Dynamixel01Constants,
    Dynamixel02Constants,
    Dynamixel03Constants,
    Dynamixel04Constants,
]


async def main() -> None:
    """
    ALOHA単腕ロボットの基本制御デモ
    """
    print("🤖 ALOHA Basic Control Demo 開始")
    print("=" * 50)

    try:
        # ALOHAコントローラー初期化
        async with AlohaArmController(
            robstride_port=ROBSTRIDE_PORT,
            dynamixel_port=DYNAMIXEL_PORT,
            robstride_constants=robstride_constants,
            dynamixel_constants=dynamixel_constants,
        ) as aloha:

            print("\n🎯 デモシーケンス開始")

            # --- デモ1: 初期位置確認 ---
            print("\n📍 デモ1: 初期位置 (全モーター 0.0 rad)")
            initial_arm = AlohaArm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            await aloha.update_pos(initial_arm)
            await asyncio.sleep(3)

            # グリッパーの電流を設定
            await aloha.set_gripper_current(500.0)  # 500mAに設定
            await asyncio.sleep(2)

            # # --- デモ2: 全モーター同時動作 ---
            print("\n📍 デモ2: 全モーター同時動作")
            sync_arm = AlohaArm(
                -math.pi / 6,  # 30度
                -math.pi / 6,
                -math.pi / 6,
                -math.pi / 6,
                -math.pi / 6,
                -math.pi / 6,
                -math.pi / 6,
            )
            await aloha.update_pos(sync_arm)
            await asyncio.sleep(4)

            # # --- デモ3: 個別モーター動作 ---
            print("\n📍 デモ3: 個別モーター動作")
            # 7から１まで順に０に戻す
            for motor_num in range(7, 0, -1):
                print(f"  → モーター{motor_num}を0.0 radに移動")
                await aloha.update_motor_pos(motor_num, 0.0)
                await asyncio.sleep(1.5)

            # --- 最終: 初期位置復帰 ---
            print("\n📍 最終: 初期位置復帰")
            await aloha.update_pos(initial_arm)
            await asyncio.sleep(3)

            print("\n🎉 全てのデモが正常に完了しました！")
            print("モーター構成: 1-3番(RobStride), 4-7番(Dynamixel)")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

# 実行コマンド: uv run -m samples.aloha_single_arm_demo
