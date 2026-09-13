from dataclasses import dataclass
from .errors import DeviceError


@dataclass(frozen=True)
class Device:
    name: str

    def __post_init__(self):
        if self.name != "cpu":
            raise DeviceError(
                f"Nexora 0.1 supports only the CPU backend; received device={self.name!r}."
            )

    def __str__(self):
        return self.name


def device(name="cpu"):
    if isinstance(name, Device):
        return name
    return Device(str(name).lower())
