from tapo import ApiClient
from dataclasses import dataclass 
from src.constants import TAPO_USERNAME, TAPO_PASSWORD, LIGHT_DEVICE_ADDR

@dataclass
class DeviceInfo:
    on: bool
    brightness: int = 0
    color_temperature: int = 0

DEFAULT_DEVICE_INFO = DeviceInfo(on=False, brightness=0, color_temperature=0)

class LightDevice: 
    def __init__(
        self, 
        tapo_username: str = TAPO_USERNAME, 
        tapo_password: str = TAPO_PASSWORD, 
        ip_addr: str = LIGHT_DEVICE_ADDR
    ): 
        self.tapo_username = tapo_username
        self.tapo_password = tapo_password
        self.ip_addr = ip_addr
        self.device = None
        self.client = ApiClient(
            self.tapo_username, self.tapo_password, timeout_s=3
        )

    async def _initialize_device(self): 
        self.device = await self.client.l530(self.ip_addr)

    async def _initialize_try(self): 
        if self.device: 
            return 
        try: 
            await self._initialize_device()
        except: 
            pass

    async def get_light_info(self) -> DeviceInfo: 
        if not self.device: 
            try: 
                await self._initialize_device()
            except: 
                return DEFAULT_DEVICE_INFO

        if not self.device: 
            return DEFAULT_DEVICE_INFO

        try: 
            device_info = (await self.device.get_device_info()).to_dict()
        except: 
            self.device = None
            return DEFAULT_DEVICE_INFO

        brightness = device_info["brightness"]
        color_temp = device_info["color_temp"]
        device_on = device_info["device_on"]

        return DeviceInfo(
            brightness=brightness,
            color_temperature=color_temp,
            on=bool(device_on),
        )

    async def on(self): 
        await self._initialize_try()
        if self.device: 
            await self.device.on()

    async def off(self): 
        await self._initialize_try()
        if self.device: 
            await self.device.off()

