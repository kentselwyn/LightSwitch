"""Custom exceptions for LightSwitch."""


class LightSwitchError(Exception):
    """Base class for package-specific errors."""


class ModelAlreadyRegisteredError(LightSwitchError):
    """Raised when registering a model name that is already managed."""


class ModelNotRegisteredError(LightSwitchError):
    """Raised when a model name is not known by the manager."""


class InsufficientVRAMError(LightSwitchError):
    """Raised when VRAM cannot be freed enough for a requested model."""


class VRAMQueryError(LightSwitchError):
    """Raised when GPU memory information cannot be queried."""

