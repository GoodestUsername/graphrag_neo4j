from abc import ABC, abstractmethod
from typing import List

import numpy as np
from torch import Tensor


# Default paths can be overridden through environment variables for flexibility
class Embedder(ABC):
    @abstractmethod
    def encode(
        self, sentences: List[str] | str, *, batch_size: int = 32
    ) -> list[Tensor] | np.ndarray | Tensor | list[dict[str, Tensor]]:
        pass
