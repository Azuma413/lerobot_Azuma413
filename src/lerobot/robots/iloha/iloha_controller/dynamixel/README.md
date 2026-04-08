# Dynamixel U2D2 Python Controller

Dynamixel XM430-W350-R用のPythonコントローラーライブラリです。非同期処理(asyncio)とGroupSync機能に対応し、複数のバスでの並行操作が可能です。

## 前提条件
- Windows 11
- Dynamixel XM430-W350-R
- U2D2
- Pythonパッケージマネージャにuvを使用

## 使用方法
1. このレポジトリをクローンして移動
2. `uv sync` でパッケージをインストール
3. U2D2をDynamixelに接続し、モーターには電源を供給
4. samples/以下のコードが使用可能です

## 主な機能

### 非同期処理対応
- `asyncio`を使用した非同期I/O処理
- 複数バスでの並行操作が可能
- `async with` 構文によるコンテキスト管理

### GroupSync機能
- GroupSyncWrite: 複数モーターへの一斉送信
- GroupSyncRead: 複数モーターからの一斉受信
- 効率的な通信による高速制御

### GroupBulkWrite機能
- 異なる制御パラメータを同時に送信
- 位置と電流制限を1パケットで設定可能
- 電流ベース位置制御モード(CURRENT_BASED_POSITION_CONTROL)に対応

### 非同期メソッド
- `set_goal_positions_async()`: 複数モーターに目標位置を一斉送信
- `get_present_positions_async()`: 複数モーターから現在位置を一斉受信
- `set_torque_enable_async()`: 複数モーターのトルクON/OFFを一斉送信
- `set_goal_velocities_async()`: 複数モーターに目標速度を一斉送信
- `set_operating_modes_async()`: 複数モーターのオペレーティングモードを一斉設定
- `set_goal_currents_async()`: 複数モーターに目標電流を一斉送信
- `set_position_and_current_goals_async()`: 複数モーターに位置と電流を同時送信 (BulkWrite)

## サンプルコード

### 単一バスでの非同期位置制御
```python
import asyncio
from dynamixel import DynamixelController, Dynamixel
from constants import DynamixelSeries, ControlParams, Baudrate

async def main():
    motor1 = Dynamixel(DynamixelSeries.XM430_W350, 1, ControlParams())
    motor2 = Dynamixel(DynamixelSeries.XM430_W350, 2, ControlParams())
    
    controller = DynamixelController("COM3", [motor1, motor2], baudrate=Baudrate.BAUD_57600)
    
    async with controller:
        # 現在位置を取得
        positions = await controller.get_present_positions_async()
        print(f"Current positions: {positions}")
        
        # 目標位置を設定 (複数モーターに一斉送信)
        goal_positions = {1: 1000, 2: 3000}
        await controller.set_goal_positions_async(goal_positions)
        
        await asyncio.sleep(2.0)

if __name__ == "__main__":
    asyncio.run(main())
```

### 複数バスでの並行操作
```python
import asyncio
from dynamixel import DynamixelController, Dynamixel
from constants import DynamixelSeries, ControlParams, Baudrate

async def main():
    # バスA (COM3)
    controller_A = DynamixelController("COM3", [...], baudrate=Baudrate.BAUD_57600)
    # バスB (COM4)
    controller_B = DynamixelController("COM4", [...], baudrate=Baudrate.BAUD_57600)
    
    async with controller_A, controller_B:
        # 両バスの位置を同時に取得
        positions_A, positions_B = await asyncio.gather(
            controller_A.get_present_positions_async(),
            controller_B.get_present_positions_async()
        )
        
        # 両バスに同時に目標位置を送信
        await asyncio.gather(
            controller_A.set_goal_positions_async({1: 1000, 2: 1500}),
            controller_B.set_goal_positions_async({10: 3000, 11: 2500})
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### BulkWriteを使った位置・電流同時制御
```python
import asyncio
from dynamixel import DynamixelController, Dynamixel
from constants import DynamixelSeries, ControlParams, Baudrate, OperatingMode

async def main():
    # 電流ベース位置制御モードを使用
    current_pos_params = ControlParams(
        ctrl_mode=OperatingMode.CURRENT_BASED_POSITION_CONTROL,
        offset=0
    )
    
    motor1 = Dynamixel(DynamixelSeries.XM430_W350, 1, current_pos_params)
    motor2 = Dynamixel(DynamixelSeries.XM430_W350, 2, current_pos_params)
    
    controller = DynamixelController("COM3", [motor1, motor2], baudrate=Baudrate.BAUD_57600)
    
    async with controller:
        # BulkWriteで位置と電流制限を同時に設定
        # 辞書の形式: { motor_id: (position, current_limit_units) }
        # XM430では約2.69mA/unit
        goals = {
            1: (2000, 186),  # 位置2000、電流約500mA
            2: (3000, 112)   # 位置3000、電流約300mA
        }
        await controller.set_position_and_current_goals_async(goals)
        
        await asyncio.sleep(2.0)

if __name__ == "__main__":
    asyncio.run(main())
```

詳細なサンプルは `src/samples/` ディレクトリを参照してください:
- `async_position_sample.py`: 単一バスでの非同期位置制御
- `async_multi_bus_sample.py`: 複数バスでの並行操作
- `async_bulk_write_sample.py`: BulkWriteを使った位置・電流同時制御
- `async_multi_bus_bulk_write_sample.py`: 複数バスでのBulkWrite並行制御
- `position_sample.py`: 従来の同期的な位置制御
- `velocity_sample.py`: 速度制御サンプル
- `multi_motor_position_sample.py`: 複数モーター位置制御サンプル

## 既存コードとの互換性

従来の同期的なメソッド(`connect()`, `disconnect()`, `set_goal_position()` など)も引き続き使用できます。新しい非同期機能は追加機能として提供されています。
