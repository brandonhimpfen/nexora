class NexoraError(Exception):
    """Base class for all Nexora errors."""


class TensorError(NexoraError):
    pass


class ShapeError(TensorError):
    pass


class ShapeMismatchError(ShapeError):
    pass


class DTypeError(TensorError):
    pass


class DeviceError(TensorError):
    pass


class AutogradError(NexoraError):
    pass


class GradientError(AutogradError):
    pass


class GraphError(AutogradError):
    pass


class ModuleError(NexoraError):
    pass


class OptimizerError(NexoraError):
    pass


class DataError(NexoraError):
    pass


class SerializationError(NexoraError):
    pass
