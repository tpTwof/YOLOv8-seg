import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

# 只启动彩色相机，压力最低
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

try:
    print("[INFO] Starting RealSense camera...")
    profile = pipeline.start(config)

    device = profile.get_device()
    try:
        usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
        print("[INFO] USB Type:", usb_type)
    except Exception:
        pass

    print("[INFO] Camera started. Press q to quit.")

    while True:
        frames = pipeline.wait_for_frames(timeout_ms=10000)
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("[WARN] No color frame")
            continue

        img = np.asanyarray(color_frame.get_data())

        cv2.imshow("RealSense RGB", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Camera stopped.")