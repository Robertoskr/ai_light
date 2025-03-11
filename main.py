from tapo import ApiClient
import asyncio
from dataclasses import dataclass
import numpy as np
import torch
from src.constants import (
    FPS, 
    MINUTES_BUFFER_SIZE, 
    FRAMES_TO_DETECT_CHANGE, 
    FRAMES_SEQ_SIZE,
    IMAGES_MEAN, 
    IMAGES_STD
)
from src.camera import CameraDevice
from src.wifi_light import LightDevice
from src.information_store import InformationStore
from src.ai import VideoCNNLSTM
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='AI Light Control System')
    parser.add_argument('--no-ai', action='store_true', 
                        help='Skip running the CNN video model for predictions')
    return parser.parse_args()

async def main(): 
    args = parse_arguments()
    
    camera = CameraDevice()
    light = LightDevice()
    buffer_size=int(FPS * 60 * MINUTES_BUFFER_SIZE)
    information_store = InformationStore(buffer_size=buffer_size)
    camera.start()
    
    # Only initialize the model if AI predictions are enabled
    model = VideoCNNLSTM(
        hidden_size=256,
        num_layers=2,
        dropout=0.5
    )
    model.eval()
    # Flag to indicate the if the last change in the light status was made by our ai model, and avoid storing that video buffer in the disk.
    # We only want to save video buffers from real user interactions. 
    changed_by_ai = False

    while True: 
        await asyncio.sleep(1 / FPS)
        light_status = await light.get_light_info()
        frame = camera.capture_frame()
        print(f"Light: {light_status.on}, frame_shape: {frame.shape}")
        # In each frame store the frame information. 
        information_store.store_frame(frame, light_status)

        # If we are not currently using the ai model to change the light status, 
        # We will store video frames every time we detect a change in the lightning. 
        if args.no_ai and not changed_by_ai: 
            # Check if we detect a change in the lightning and we want to save this to the disk. To avoid noise, we check changes using the frames t and t-d
            frames_range = information_store.get_range_from_end(count=FRAMES_TO_DETECT_CHANGE + 1)
            if len(frames_range) == FRAMES_TO_DETECT_CHANGE + 1: 
                last = frames_range[-1].light_information["on"]
                first = frames_range[0].light_information["on"]
                second = frames_range[1].light_information["on"]

                if last != first and last == second: 
                    information_store.save_to_disk()
                    changed_by_ai = False

        # If ai is enabled, lets predict the light status and set the light status to true/false. 
        elif not args.no_ai: 
            frames_range = information_store.get_range_from_end(count=FRAMES_SEQ_SIZE)
            if len(frames_range) >= FRAMES_SEQ_SIZE: 
                X = torch.tensor(
                    (np.array([f.frame for f in frames_range]) - IMAGES_MEAN) / IMAGES_STD
                ).float().unsqueeze(0)
                prediction = model(X).sigmoid()
                print(f"AI prediction: {prediction}")
                light_on = bool(prediction[0].item() > 0.5)

                if light_status.on != light_on: 
                    if light_on: 
                        await light.on()
                    else: 
                        await light.off()

                    changed_by_ai = True

    camera.stop()

if __name__ == "__main__": 
    asyncio.run(main())
