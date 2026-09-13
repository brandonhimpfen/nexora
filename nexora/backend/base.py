from abc import ABC, abstractmethod


class Backend(ABC):
    name: str

    @abstractmethod
    def array(self, data, dtype=None):
        raise NotImplementedError
