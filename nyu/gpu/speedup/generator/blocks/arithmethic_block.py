import random

from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
from autogoal.grammar import CategoricalValue

from nyu.gpu.speedup.generator.variables import Variable, VariableType
from nyu.gpu.speedup.generator import GenerationException

from .base_block import BaseBlock, INDENT

ATOMIC_MUL  = """
__device__ float atomicMult(float* address, float val) 
{ 
  int* address_as_int = (int*)address; 
  int old = *address_as_int, assumed; 
  do { 
    assumed = old; 
    old = atomicCAS(address_as_int, assumed, __float_as_int(val * __float_as_int(assumed))); 
 } while (assumed != old); return __int_as_float(old);
}
"""

ATOMIC_DIV  = """
__device__ float atomicDiv(float* address, float val) 
{ 
  int* address_as_int = (int*)address; 
  int old = *address_as_int, assumed; 
  do { 
    assumed = old; 
    old = atomicCAS(address_as_int, assumed, __float_as_int(val / __float_as_int(assumed))); 
 } while (assumed != old); return __int_as_float(old);
}
"""

def generate_variable(variable_list: List[Variable], index_variable_list : List[Variable]) -> Tuple[str, str]:
    var = random.choice(variable_list)
    match var.type, len(index_variable_list):
        case VariableType.SINGLE:
            return str(var), None
        case  VariableType.ARRAY_TEMP | VariableType.ARRAY_INPUT | VariableType.ARRAY_OUTPUT, 0:
            return None, None
        case VariableType.ARRAY_INPUT | VariableType.ARRAY_OUTPUT, 1:
            return f"{var}[{random.choice(index_variable_list)}]", f"{var}[{random.choice(index_variable_list)}]"
        case VariableType.ARRAY_INPUT | VariableType.ARRAY_OUTPUT, 2:
            idx_1, idx_2 = random.choices(index_variable_list, k=2)
            return f"{var}[index2D({idx_1}, {idx_2}, N)]", f"{var}[index2D({idx_1}, {idx_2}, N)]"
        case VariableType.ARRAY_INPUT | VariableType.ARRAY_OUTPUT, 3:
            idx_1, idx_2, idx_3 = random.choices(index_variable_list, k=3)
            return f"{var}[index3D({idx_1}, {idx_2}, {idx_3}, N)]", f"{var}[index3D({idx_1}, {idx_2}, {idx_3}, N)]",
        case VariableType.ARRAY_INPUT | VariableType.ARRAY_OUTPUT, _:
            return f"{var}[{index_variable_list[0]}]",  f"{var}[{index_variable_list[0]}]"
        case VariableType.ARRAY_INPUT | VariableType.ARRAY_OUTPUT, 1:
            return f"{var}[{random.choice(index_variable_list)}]", f"{var}[tx]"
        case VariableType.ARRAY_INPUT | VariableType.ARRAY_OUTPUT, 2:
            idx_1, idx_2 = random.choices(index_variable_list, k=2)
            return f"{var}[index2D({idx_1}, {idx_2}, N)]", f"{var}[index2D(tx, ty, 1024)]"
        case VariableType.ARRAY_INPUT | VariableType.ARRAY_OUTPUT, 3:
            idx_1, idx_2, idx_3 = random.choices(index_variable_list, k=3)
            return f"{var}[index3D({idx_1}, {idx_2}, {idx_3}, N)]", f"{var}[index3D(tx, ty, tz, 1024)]"
        case VariableType.ARRAY_INPUT | VariableType.ARRAY_OUTPUT, _:
            return f"{var}[{index_variable_list[0]}]", f"{var}[index3D(tx, ty, tz, 1024)]"
    return None, None

class Operators(Enum):
    ADD = 0
    SUB = 1
    MULT = 2
    DIV = 3

    def to_string(self) -> str:
        match self:
            case Operators.ADD:
                return "+"
            case Operators.SUB:
                return  "-"
            case Operators.MULT:
                return "*"
            case Operators.DIV:                
                return "/"
        return None

class ArithmeticAccumulationBlock(BaseBlock):

    def __init__(self, operator: CategoricalValue(*list(Operators))) -> None:
        super().__init__()
        self.operator = operator
        self.var_in_cpu : Optional[Variable] = None
        self.var_out_cpu : Optional[Variable] = None
        self.var_in_cpu : Optional[Variable] = None
        self.var_out_gpu : Optional[Variable] = None
    
    def _generate_helper(self, variables: Dict[VariableType, Set[Variable]]) -> Tuple[str, str]:
        input_variable_list = list(variables[VariableType.SINGLE]) + list(variables[VariableType.ARRAY_INPUT]) + list(variables[VariableType.ARRAY_TEMP])
        output_variable_list = list(variables[VariableType.ARRAY_OUTPUT]) + list(variables[VariableType.ARRAY_TEMP])
        index_variable_list = list(variables[VariableType.INDEX])
        if len(input_variable_list) >= 1 and output_variable_list:
            var_in = generate_variable(input_variable_list, index_variable_list)
            var_out = generate_variable(output_variable_list, index_variable_list)
            return var_in, var_out
        return (None, None), (None, None)

    def _generate(self, depth: int, var_in : str, var_out : str) -> Tuple[str, str]:
        operator_str = self.operator.to_string()
        if None in [var_in, var_out, operator_str]:
            raise GenerationException
        return f"{INDENT*depth}{var_out} {operator_str}= {var_in};"

    def generate(self, depth: int, variables: Dict[VariableType, Set[Variable]]) -> str:
        (self.var_in_cpu, self.var_in_gpu), (self.var_out_cpu, self.var_out_gpu) = self._generate_helper(variables)
        return self._generate(depth, self.var_in_cpu, self.var_out_cpu)

    def generate_gpu(self, depth: int, block_depth: int, max_block_depth : int, variables: Dict[VariableType, Set[Variable]], device_kernels : List[str]) -> str:
        if random.random() > 0.6:
            operator_str = ""
            sign = ""
            match self.operator:
                case Operators.ADD:
                    operator_str = "atomicAdd"
                case Operators.SUB:
                    operator_str = "atomicAdd"
                    sign = "-"
                case Operators.MULT:
                    operator_str = "atomicMult"
                    device_kernels.append(ATOMIC_MUL)
                case Operators.DIV:                
                    operator_str = "atomicDiv"
                    device_kernels.append(ATOMIC_DIV)

            return f"{INDENT*depth}{operator_str}(&{self.var_out_gpu}, {sign}{self.var_in_gpu});"
        return self._generate(depth, self.var_in_gpu, self.var_out_gpu)
    
    def compute_depth(self) -> int:
        return 0

class ArithmeticOperationBlock(BaseBlock):

    def __init__(self, operator: CategoricalValue(*list(Operators))) -> None:
        super().__init__()
        self.operator = operator
        self.var_in_1_cpu : Optional[Variable] = None
        self.var_in_2_cpu : Optional[Variable] = None
        self.var_out_cpu : Optional[Variable] = None
        self.var_in_1_gpu : Optional[Variable] = None
        self.var_in_2_gpu : Optional[Variable] = None
        self.var_out_gpu : Optional[Variable] = None

    def _generate_helper(self, variables: Dict[VariableType, Set[Variable]]) -> Tuple[str, str, str]:
        input_variable_list = list(variables[VariableType.SINGLE]) + list(variables[VariableType.ARRAY_INPUT]) + list(variables[VariableType.ARRAY_TEMP])
        output_variable_list = list(variables[VariableType.ARRAY_OUTPUT]) + list(variables[VariableType.ARRAY_TEMP])
        index_variable_list = list(variables[VariableType.INDEX])
        if len(input_variable_list) >= 2 and output_variable_list:
            var_in_1 = generate_variable(input_variable_list, index_variable_list)
            var_in_2 = generate_variable(input_variable_list, index_variable_list)
            var_out = generate_variable(output_variable_list, index_variable_list)
            return var_in_1, var_in_2, var_out
        return (None, None), (None, None), (None, None)

    def _generate(self, depth : int, var_in_1 : str, var_in_2 : str, var_out : str) -> str:
        operator_str = self.operator.to_string()
        if None in [var_in_1, var_in_2, var_out, operator_str]:
            raise GenerationException
        return f"{INDENT*depth}{var_out} = {var_in_1} {operator_str} {var_in_2};"

    def generate(self, depth: int, variables: Dict[VariableType, Set[Variable]]) -> str:
        (self.var_in_1_cpu, self.var_in_1_gpu), (self.var_in_2_cpu, self.var_in_2_gpu), (self.var_out_cpu, self.var_out_gpu) = self._generate_helper(variables)
        return self._generate(depth, self.var_in_1_cpu, self.var_in_2_cpu, self.var_out_cpu)

    def generate_gpu(self, depth: int, block_depth: int, max_block_depth : int, variables: Dict[VariableType, Set[Variable]], device_kernels : List[str]) -> str:
        return self._generate(depth, self.var_in_1_gpu, self.var_in_2_gpu, self.var_out_gpu)
    
    def compute_depth(self) -> int:
        return 0
