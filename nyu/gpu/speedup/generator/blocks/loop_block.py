from typing import Dict, Optional, Set, List
from collections import defaultdict

from nyu.gpu.speedup.generator.variables import Variable, VariableType

from .inter_block import InterBlock
from .base_block import BaseBlock, INDENT

class ForLoopBlock(BaseBlock):

    LEVELS = ["x", "y", "z"]

    def __init__(self, child: InterBlock) -> None:
        super().__init__()
        self.child = child
        self.index_var : Optional[Variable] = None

    def compute_depth(self) -> int:
        return self.child.compute_depth() + 1

    def _generate(self, depth : int, variables: Dict[VariableType, Set[Variable]]) -> str:
        code =  f"""
{INDENT*depth}for (int {self.index_var} = 0; {self.index_var} <= N; {self.index_var}++){{
{self.child.generate(depth+1, variables)}
{INDENT*depth}}}
        """
        return code

    def generate(self, depth : int, variables: Dict[VariableType, Set[Variable]]) -> str:
        self.index_var = Variable(VariableType.INDEX)
        new_variables = defaultdict(set, variables)
        new_variables[self.index_var.type].add(self.index_var)
        code = self._generate(depth, new_variables)
        new_variables[self.index_var.type].remove(self.index_var)
        return code


    def generate_gpu(self, depth: int, block_depth: int, max_block_depth: int, variables: Dict[VariableType, Set[Variable]], device_kernels : List[str]) -> str:
        new_variables = defaultdict(set, variables)
        new_variables[self.index_var.type].add(self.index_var)

        if block_depth >= max_block_depth:
            return self._generate(depth, new_variables)

        level_letter = self.LEVELS[block_depth]
        
        code = ""

        if block_depth == 0:
            for i in range(max_block_depth):
                i_letter = self.LEVELS[i]
                code += f"{INDENT*depth}int b{i_letter} = blockIdx.{i_letter}, bd{i_letter} = blockDim.{i_letter},  t{i_letter} = threadIdx.{i_letter};\n"

        code +=  f"""
{INDENT*depth}int {self.index_var} = b{level_letter}*bd{level_letter} + t{level_letter};
{INDENT*depth}if ({self.index_var} < N){{
{self.child.generate(depth+1, new_variables) if block_depth > max_block_depth - 1 else self.child.generate_gpu(depth+1, block_depth+1, max_block_depth, new_variables, device_kernels)}
{INDENT*depth}}}
{INDENT*depth}
        """

        new_variables[self.index_var.type].remove(self.index_var)
        return code
