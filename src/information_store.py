import numpy as np 
import os
import json
from src.wifi_light import DeviceInfo
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from collections import deque

@dataclass
class FrameInfo: 
    timestamp: int 
    frame: list[list[list[int]]] # 3 * height * width matrix
    light_information: dict

class InformationStore: 
    def __init__(self, buffer_size: int, storage_dir: str = "data"): 
        """
        Initialize the information store with a fixed buffer size.
        
        Args:
            buffer_size: Maximum number of frames to keep in memory
            storage_dir: Directory where frames will be stored on disk
        """
        self.store = []
        self.buffer_size = buffer_size
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        self.buffer_size = buffer_size

    def store_frame(self, frame, light_status: DeviceInfo): 
        frame_info = FrameInfo(
            timestamp=int(datetime.now(UTC).timestamp()), 
            frame=frame,
            light_information={
                "on": light_status.on,
                "brightness": light_status.brightness,
                "color_temp": light_status.color_temperature,
            }
        )

        # Add the new frame - if buffer is full, deque automatically removes the oldest item
        self.store.append(frame_info)

        # TODO: This takes O(n), optimize when Possible/Needed.  
        if len(self.store) > self.buffer_size: 
            self.store.pop(0)

    def clear_buffer(self): 
        self.store = []
    
    def save_to_disk(self, filename=None):
        if not self.store:
            return None
            
        # Create a timestamp-based filename if none provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frames_{timestamp}"
            
        # Ensure the filename has the .npz extension
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        filepath = os.path.join(self.storage_dir, filename)
        
        # Extract frames and metadata
        frames_data = []
        metadata = []
        
        for frame_info in self.store:
            frames_data.append(frame_info.frame)
            
            # Store metadata separately
            meta = {
                'timestamp': frame_info.timestamp,
                'light_information': frame_info.light_information
            }
            metadata.append(meta)
        
        # Save frames as a compressed numpy array
        np.savez_compressed(
            filepath,
            frames=np.array(frames_data),
            metadata=json.dumps(metadata)
        )
        
        print(f"Saved {len(self.store)} frames to {filepath}")
        return filepath
    
    def get_range_from_end(self, count=1): 
        buffer_size = len(self.store)
        return self.store[buffer_size - count:]

