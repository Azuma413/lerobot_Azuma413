import argparse
import concurrent.futures
import shutil
import threading
import time
from pathlib import Path
from threading import Thread

import numpy as np
from PIL import Image
import sounddevice as sd
import pyroomacoustics as pra

from lerobot.common.robot_devices.cameras.configs import SoundCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceNotConnectedError,
    busy_wait,
)

def save_image(img_array, frame_index, images_dir):
    img = Image.fromarray(img_array)
    path = images_dir / f"sound_camera_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def save_images_from_cameras(
    images_dir: Path,
    top_mic_idx=4,
    right_mic_idx=6,
    left_mic_idx=5,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
    mock=False,
):
    cameras = []
    config = SoundCameraConfig(
        top_mic_idx=top_mic_idx,
        right_mic_idx=right_mic_idx,
        left_mic_idx=left_mic_idx,
        fps=fps,
        width=width,
        height=height,
        color_mode="rgb",
        mock=mock,
    )
    camera = SoundCamera(config)
    camera.connect()
    cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            now = time.perf_counter()

            for camera in cameras:
                # If we use async_read when fps is None, the loop will go full speed, and we will endup
                # saving the same images from the cameras multiple times until the RAM/disk is full.
                image = camera.read() if fps is None else camera.async_read()

                executor.submit(
                    save_image,
                    image,
                    frame_index,
                    images_dir,
                )

            if fps is not None:
                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

            print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            if time.perf_counter() - start_time > record_time_s:
                break

            frame_index += 1

    print(f"Images have been saved to {images_dir}")


class SoundCamera:
    """
    マイクアレイから音声を取得し、PyroomAcousticsでDOA推定を行い、その空間応答から画像を生成するカメラデバイスクラスです。
    """

    def __init__(self, config: SoundCameraConfig):
        # configからマイクデバイスパス・画像サイズ等を取得
        self.config = config
        self.top_mic_device = f"TAMAGO-03: USB Audio (hw:{config.top_mic_idx},0)"
        self.right_mic_device = f"TAMAGO-03: USB Audio (hw:{config.right_mic_idx},0)"
        self.left_mic_device = f"TAMAGO-03: USB Audio (hw:{config.left_mic_idx},0)"
        self.width = config.width or 640
        self.height = config.height or 480
        self.fps = config.fps or 16
        self.color_mode = config.color_mode
        self.channels = 3  # top, right, left
        self.mock = config.mock
        self.is_connected = False

        # 録音用パラメータ
        self.fs = 16000
        self.record_seconds = 0.1  # 0.1秒録音
        # デバイス名リストに修正
        self.device_paths = [self.top_mic_device, self.right_mic_device, self.left_mic_device]

        # PyroomAcoustics DOAパラメータ
        self.nfft = 256
        self.freq_range = [300, 3500]

        # 非同期読み出し用
        self.thread = None
        self.stop_event = None
        self.sound_image = None

    def connect(self):
        """デバイスを接続状態にします（実際のopenは録音時に行う）"""
        if self.is_connected:
            return
        self.is_connected = True

    def read(self) -> np.ndarray:
        """
        3つのマイクデバイスから同時に音声を取得し、PyroomAcousticsでDOA推定を行い、空間応答を画像化して返す
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("SoundCameraはconnect()されていません。")

        # mockモードの場合はランダム画像を返す
        if self.mock:
            img = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            return img

        # 3つのマイクデバイスで独立にDOA推定し、RGB画像化（env/tasks/sound.py参照）
        sound_maps = []
        for mic_idx, dev in enumerate(self.device_paths):
            try:
                rec = sd.rec(
                    int(self.fs * self.record_seconds),
                    samplerate=self.fs,
                    channels=8,
                    dtype='float32',
                    device=dev,
                )
                sd.wait()
                # 8chマイクアレイ仮定
                data = rec.T  # shape: (8, サンプル数)
                # マイク座標（円形配置、z=0.1m）
                mic_locs = pra.circular_2D_array(center=[0, 0], M=8, phi0=0, radius=0.035)
                mic_positions = np.vstack([mic_locs, np.ones((1, 8)) * 0.1])  # shape: (3, 8)
                # STFT
                X = pra.transform.stft.analysis(data.T, self.nfft, self.nfft // 2)
                X = X.transpose([2, 1, 0])
                doa = pra.doa.algorithms['MUSIC'](
                    mic_positions, self.fs, self.nfft, c=343., num_src=1, max_four=4
                )
                doa.locate_sources(X, freq_range=self.freq_range)
                spatial_resp = doa.grid.values
                # 0-1正規化
                spatial_resp = (spatial_resp - spatial_resp.min()) / (spatial_resp.max() - spatial_resp.min() + 1e-8)
            except Exception as e:
                spatial_resp = np.zeros(360)

            # 画像化
            # マイクアレイ中心を画像中央にマッピング
            mic_coord = [self.height // 2, self.width // 2]
            y_idx, x_idx = np.indices((self.height, self.width))
            angles = (np.arctan2(y_idx - mic_coord[0], x_idx - mic_coord[1]) * 180 / np.pi + 90) % 360
            angles_idx = angles.astype(int)
            sound_map = spatial_resp[angles_idx]
            sound_maps.append(sound_map)

        # RGB画像化（env/tasks/sound.pyの画像化処理に近づける）
        sound_image_array = np.array(sound_maps)  # (3, H, W)
        # y軸反転
        sound_image_array = np.flip(sound_image_array, axis=2)
        # (H, W, 3)へ転置
        sound_image_array = np.transpose(sound_image_array, (1, 2, 0))
        # 0-255にスケーリング＆uint8化
        sound_image_array = np.clip(sound_image_array * 255, 0, 255).astype(np.uint8)
        return sound_image_array

    def read_loop(self):
        """
        別スレッドでread()を繰り返し呼び出し、最新の画像をself.sound_imageに保存し続けるループです。
        """
        while not self.stop_event.is_set():
            try:
                self.sound_image = self.read()
            except Exception as e:
                print(f"SoundCameraスレッドでのreadエラー: {e}")

    def async_read(self):
        """
        非同期で画像を取得します。最初の呼び出し時にスレッドを起動し、以降は最新の画像を返します。
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("SoundCameraはconnect()されていません。")

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while True:
            if self.sound_image is not None:
                return self.sound_image

            time.sleep(1 / self.fps)
            num_tries += 1
            if num_tries > self.fps * 2:
                raise TimeoutError("async_read()の開始待機中にタイムアウトしました。")

    def disconnect(self):
        """デバイスを切断し、スレッドも安全に停止します"""
        if not self.is_connected:
            return
        self.is_connected = False
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

    def __del__(self):
        # デストラクタでのdisconnectは例外が出ても無視
        try:
            if getattr(self, "is_connected", False):
                self.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `OpenCVCamera` for all cameras connected to the computer, or a selected subset."
    )
    
    parser.add_argument(
        "--top-mic-idx",
        type=int,
        default=4,
        help="Set the path to the top microphone device. Default is '/dev/ttyUSB0'.",
    )
    parser.add_argument(
        "--right-mic-idx",
        type=int,
        default=6,
        help="Set the path to the right microphone device. Default is '/dev/ttyUSB1'.",
    )
    parser.add_argument(
        "--left-mic-idx",
        type=int,
        default=5,
        help="Set the path to the left microphone device. Default is '/dev/ttyUSB2'.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_opencv_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=4.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
