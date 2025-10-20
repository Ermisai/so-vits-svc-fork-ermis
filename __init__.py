"""
so_vits_svc_fork_ermis - Isolated inference module
Contains only the necessary components for running Svc inference (batch and sequential)
"""

__version__ = "0.1.0"

from so_vits_svc_fork_ermis.inference.core import Svc

__all__ = ["Svc", "__version__"]
