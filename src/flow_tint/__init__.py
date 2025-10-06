from importlib.metadata import version, PackageNotFoundError

from .core.constants import BOS_TOKEN, EOS_TOKEN

try:
    __version__ = version("flow-tint")
except PackageNotFoundError:
    __version__ = "unknown"
