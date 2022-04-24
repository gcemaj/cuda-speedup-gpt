from abc import ABC, abstractmethod
from typing import Dict, Set, List

from nyu.gpu.speedup.generator.variables import Variable, VariableType

INDENT = "    "

class BaseBlock(ABC):

    @abstractmethod
    def generate(self, depth: int, variables: Dict[VariableType, Set[Variable]]) -> str:
        pass

    @abstractmethod
    def generate_gpu(self, depth: int, block_depth: int, max_block_depth : int, variables: Dict[VariableType, Set[Variable]]) -> str:
        pass

    @abstractmethod
    def compute_depth(self) -> int:
        pass

class NoneBlock(BaseBlock):

    def generate(self, variables:  Dict[VariableType, Set[Variable]], device_kernels : List[str]) -> str:
        return ""

    def generate_gpu(self, depth: int, block_depth: int, max_block_depth : int, variables: Dict[VariableType, Set[Variable]]) -> str:
        return self.generate(depth, variables)
    
    def compute_depth(self) -> int:
        return 0