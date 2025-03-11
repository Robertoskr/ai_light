import cv2
import numpy as np
import time 
from src.constants import FRAME_HEIGHT, FRAME_WIDTH
try: 
    from picamera2 import Picamera2
except: 
    print("WARNING: Picamera is None")
    Picamera2 = None
    Controls = None
    libcamera = None

class CameraDevice: 
    def __init__(self): 
        self.picam2 = None
        if Picamera2 is not None: 
            self.picam2 = Picamera2()
            
            # Create a full-resolution configuration with wide field of view
            camera_config = self.picam2.create_preview_configuration(
                main={
                    "format": "RGB888",
                    "size": self.picam2.sensor_resolution  # Use full sensor resolution
                },
                lores={"format": "YUV420"},
                display="lores",
            )
            
            # Apply the night configuration
            self.picam2.configure(camera_config)
            # Configure for night vision
            # Maximum exposure time (in microseconds) - longer exposure captures more light
            # 1,000,000 = 1 second exposure (very long, may cause motion blur)
            exposure_time = 500000 // 2 # 500ms exposure - balance between light sensitivity and motion blur
            
            # Maximum analog gain - amplifies the sensor signal (higher = more noise but better low light)
            # Values between 8.0-16.0 are good for night, but higher values introduce more noise
            analog_gain = 16.0
            
            # Digital gain - software amplification after analog (use sparingly as it increases noise)
            digital_gain = 4.0
            
            # Disable auto white balance for night mode (tends to make night images too blue/green)
            awb_mode = 0  # Manual mode
            
            # Set red and blue gains for better night color (slightly warmer tones work better at night)
            awb_gains = (1.5, 1.0)  # (red gain, blue gain)
            
            # Set controls for night vision
            self.picam2.set_controls({
                "ExposureTime": exposure_time,
                "AnalogueGain": analog_gain,
                "ColourGains": awb_gains,
                "AeEnable": False,  # Disable auto exposure
                "AwbEnable": False,  # Disable auto white balance
                "Brightness": 0.1,   # Slight brightness boost
                "Contrast": 1.2,     # Increase contrast slightly
                "Sharpness": 0.0,    # Reduce sharpness to minimize noise
                "NoiseReductionMode": 2  # High noise reduction
            })

    def capture_frame(self, is_test: bool = False): 
        if self.picam2 is None: 
            return np.random.randn(3, FRAME_HEIGHT, FRAME_WIDTH)

        full_res_frame = self.picam2.capture_array()
        resized_frame = cv2.resize(full_res_frame, (FRAME_HEIGHT, FRAME_WIDTH), interpolation=cv2.INTER_AREA)
        frame = np.transpose(resized_frame, (2, 0, 1))
        return frame 

    def start(self): 
        if self.picam2: 
            self.picam2.start()
            time.sleep(2)

    def stop(self): 
        if self.picam2: 
            self.picam2.stop()

    def __enter__(self, *args, **kwargs): 
        self.start()

    def __exit__(self, *args, **kwargs): 
        self.stop()
