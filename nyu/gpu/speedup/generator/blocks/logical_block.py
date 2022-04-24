from typing import Dict, List, Optional, Set
from autogoal.grammar import ContinuousValue

from nyu.gpu.speedup.generator.variables import Variable, VariableType

from .base_block import BaseBlock, INDENT

class SyncThreadsBlock(BaseBlock):

    def __init__(self) -> None:
        super().__init__()

    def generate(self, depth : int, variables: Dict[VariableType, Set[Variable]]) -> str:
        return ""

    def generate_gpu(self, depth: int, block_depth: int, max_block_depth : int,  variables: Dict[VariableType, Set[Variable]], device_kernels : List[str]) -> str:
        return f"{INDENT*depth}__syncthreads();"
    
    def compute_depth(self) -> int:
        return 0

class TempStorageBlock(BaseBlock):

    def __init__(self) -> None:
        super().__init__()
        self.var : Optional[Variable] = None

    def generate(self, depth : int, variables: Dict[VariableType, Set[Variable]]) -> str:
        self.var = Variable(VariableType.ARRAY_TEMP)
        variables[self.var.type].add(self.var)
        return  f"""
{INDENT*depth}float *{self.var.name};
{INDENT*depth}{self.var.name} = (float *)malloc(N*sizeof(float));
"""
    def generate_gpu(self, depth: int, block_depth: int, max_block_depth : int,  variables: Dict[VariableType, Set[Variable]], device_kernels : List[str]) -> str:
        variables[self.var.type].add(self.var)
        return f""" 
{INDENT*depth}__shared__ float {self.var.name}[1024];
"""
    
    def compute_depth(self) -> int:
        return 0

