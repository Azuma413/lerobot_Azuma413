import asyncio
import math
import time
from lerobot.robots.iloha.iloha_controller.aloha_controller import AlohaArm, AlohaController

# --- 設定項目 ---
# 各ポート名をご自身の環境に合わせて変更してください
RIGHT_DYNAMIXEL_PORT = "/dev/ttyUSB1"
RIGHT_ROBSTRIDE_PORT = "/dev/ttyUSB0"
LEFT_ROBSTRIDE_PORT = "/dev/ttyUSB2"
LEFT_DYNAMIXEL_PORT = "/dev/ttyUSB3"
# -----------------
Freq_Hz = 60  # 制御周波数

async def main() -> None:
    """
    ALOHA双腕ロボットの基本制御デモ
    """
    print("🤖 ALOHA双腕制御デモ開始")
    print("=" * 50)

    try:
        # ALOHA双腕コントローラー初期化
        async with AlohaController(
            right_robstride_port=RIGHT_ROBSTRIDE_PORT,
            right_dynamixel_port=RIGHT_DYNAMIXEL_PORT,
            left_robstride_port=LEFT_ROBSTRIDE_PORT,
            left_dynamixel_port=LEFT_DYNAMIXEL_PORT,
        ) as aloha:

            print("\n🎯 双腕デモシーケンス開始")

            # --- デモ1: 初期位置確認 ---
            print("\n📍 デモ1: 初期位置 (全モーター 0.0 rad)")

            # グリッパーの電流を設定（並列実行）
            await asyncio.gather(
                aloha.set_gripper_current("right", 500.0),  # 500mAに設定
                aloha.set_gripper_current("left", 500.0),  # 500mAに設定
            )

            await asyncio.sleep(2)

            right_arm = AlohaArm(
                0.0,
                -math.pi / 6,
                -math.pi / 6,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            left_arm = AlohaArm(
                0.0,
                -math.pi / 6,
                -math.pi / 6,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            await aloha.update_pos(right_arm=right_arm, left_arm=left_arm)

            await asyncio.sleep(2)
            start_time = time.time()
            loop_start_time = None
            while time.time() - start_time < 30.0:
            # --- デモ2: 動作 ---
            # print("\n📍 デモ2: 動作")
                loop_start_time = time.time()
                noise = 0.3 * math.sin(2.0 * math.pi * 0.1 * (time.time() - start_time))
                right_arm = AlohaArm(
                    0.0,
                    -math.pi / 6,
                    -math.pi / 6 + noise,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                left_arm = AlohaArm(
                    0.0,
                    -math.pi / 6,
                    -math.pi / 6 + noise,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                await aloha.update_pos(right_arm=right_arm, left_arm=left_arm)
                delta_time = time.time() - loop_start_time
                await asyncio.sleep(max(0.0, (1.0 / Freq_Hz) - delta_time))
            # await asyncio.sleep(120)

            # --- デモ3: 個別モーター制御 ---
            # 7から１まで順に０に戻す
            print("\n📍 デモ3: 個別モーター制御")
            for motor_num in range(7, 0, -1):
                print(f"  → 右アーム モーター{motor_num}を 0.0 radに移動")
                await aloha.update_motor_pos(
                    arm="right", motor_num=motor_num, target_pos=0.0
                )
                await asyncio.sleep(1)

                print(f"  → 左アーム モーター{motor_num}を 0.0 radに移動")
                await aloha.update_motor_pos(
                    arm="left", motor_num=motor_num, target_pos=0.0
                )
                await asyncio.sleep(1)

            print("\n🎉 双腕デモが正常に完了しました！")
            print("構成: 右アーム(1-3番RobStride, 4-7番Dynamixel)")
            print("      左アーム(1-3番RobStride, 4-7番Dynamixel)")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

# uv run -m kvt_aloha_python_controller.samples.aloha_dual_arm_demo