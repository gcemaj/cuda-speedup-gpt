from typing import Dict, List, Set

from nyu.gpu.speedup.generator.variables import Variable, VariableType
from .base_block import BaseBlock, NoneBlock

class InterBlock(BaseBlock):

    def generate(self, depth: int, variables: Dict[VariableType, Set[Variable]]) -> str:
        result = self.current.generate(depth, variables)
        match self.next:
            case NoneBlock():
                return result
            case _ :
                return result + f"\n" + self.next.generate(depth, variables)

    def generate_gpu(self, depth: int, block_depth: int, max_block_depth, variables: Dict[VariableType, Set[Variable]], device_kernels : List[str]) -> str:
        result = self.current.generate_gpu(depth, block_depth, max_block_depth, variables, device_kernels)
        match self.next:
            case NoneBlock():
                return result
            case _ :
                return result + f"\n" + self.next.generate_gpu(depth, block_depth, max_block_depth, variables, device_kernels)

    def compute_depth(self) -> int:
        depth = self.current.compute_depth()
        match self.next:
            case NoneBlock():
                return depth
            case _ :
                return max(depth, self.next.compute_depth())


