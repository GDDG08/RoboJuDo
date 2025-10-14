# fix libgomp issue on ARM platform
import torch  # noqa: F401, I001
import numpy  # noqa: F401, I001
import scipy.spatial.transform.rotation  # noqa: F401, I001

# load all packages
import robojudo.config  # ensure configs are registered first  # noqa: E402, F401, I001

import robojudo.controller  # ensure controllers are registered  # noqa: E402, F401, I001
import robojudo.environment  # ensure environments are registered  # noqa: E402, F401, I001
import robojudo.pipeline  # ensure pipelines are registered  # noqa: E402, F401, I001
import robojudo.policy  # ensure policies are registered  # noqa: E402, F401, I001

# Initialize the global logger
from robojudo.utils.logger import setup_logger  # noqa: E402, I001

logger = setup_logger()

# Print version info
__version__ = "1.4.1"
logger.debug(f"{'=' * 10} robojudo-{__version__} init done {'=' * 10}\n")
