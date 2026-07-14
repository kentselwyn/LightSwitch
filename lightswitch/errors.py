"""Custom exceptions for LightSwitch."""


class LightSwitchError(Exception):
    """Base class for package-specific errors."""


class ModelAlreadyRegisteredError(LightSwitchError):
    """Raised when registering a model name that is already managed."""


class ModelNotRegisteredError(LightSwitchError):
    """Raised when a model name is not known by the manager."""


class ModelTransitionError(LightSwitchError):
    """Raised when a framework adapter fails a residency transition."""


class InsufficientRAMError(LightSwitchError):
    """Raised when system RAM cannot be freed enough for a request."""


class InsufficientVRAMError(LightSwitchError):
    """Raised when VRAM cannot be freed enough for a request."""


class RAMQueryError(LightSwitchError):
    """Raised when system RAM information cannot be queried."""


class VRAMQueryError(LightSwitchError):
    """Raised when GPU memory information cannot be queried."""
