from collections import defaultdict
from typing import Dict, List, Set
from autogoal.grammar import CategoricalValue

from nyu.gpu.speedup.generator.variables import Variable, VariableType

from .loop_block import ForLoopBlock
from .base_block import BaseBlock, INDENT

class FunctionBlock(BaseBlock):

    def __init__(self, number_inputs : CategoricalValue(1, 2), number_outputs : CategoricalValue(1, 2), root_loop: ForLoopBlock) -> None:
        self.inputs = [Variable(VariableType.ARRAY_INPUT) for i in range(number_inputs)]
        self.outputs = [Variable(VariableType.ARRAY_OUTPUT) for i in range(number_outputs)]
        self.name = "func_cpu"
        self.name_gpu = "func_gpu"
        self.root_loop = root_loop

    def compute_depth(self) -> int:
        return self.root_loop.compute_depth()

    def generate(self, depth: int, variables: Dict[VariableType, Set[Variable]]) -> str:
        new_variables = defaultdict(set, variables)
        new_variables[VariableType.ARRAY_INPUT].update(self.inputs)
        new_variables[VariableType.ARRAY_OUTPUT].update(self.outputs)
        inputs_str = ', '.join(f"float * {i}" for i in self.inputs)
        outputs_str = ', '.join(f"float * {i}" for i in self.outputs)
        return f"""
{INDENT*depth}void {self.name}({inputs_str}, {outputs_str}, unsigned int N){{
{self.root_loop.generate(depth+1, new_variables)}
{INDENT*depth}}}
        """

    def generate_gpu(self, depth: int, block_depth: int, max_block_depth : int, variables: Dict[VariableType, Set[Variable]], device_kernels : List[str]) -> str:
        new_variables = defaultdict(set, variables)
        new_variables[VariableType.ARRAY_INPUT].update(self.inputs)
        new_variables[VariableType.ARRAY_OUTPUT].update(self.outputs)
        inputs_str = ', '.join(f"float * {i}" for i in self.inputs)
        outputs_str = ', '.join(f"float * {i}" for i in self.outputs)

        declare_gpu = '\n'.join(f"{INDENT*(depth+1)}float * {i}_gpu;" for i in self.inputs + self.outputs)
        malloc_gpu = '\n'.join(f"{INDENT*(depth+1)}cudaMalloc(&{i}_gpu, size);" for i in self.inputs + self.outputs)
        copy_to_gpu = '\n'.join(f"{INDENT*(depth+1)}cudaMemcpy({i}_gpu, {i}, size, cudaMemcpyHostToDevice);" for i in self.inputs)
        copy_from_gpu = '\n'.join(f"{INDENT*(depth+1)}cudaMemcpy({i}, {i}_gpu, size, cudaMemcpyDeviceToHost);" for i in self.outputs)
        input_output_call_gpu = ', '.join(f"{i}_gpu" for i in self.inputs + self.outputs)
        free_gpu = '\n'.join(f"{INDENT*(depth+1)}cudaFree({i}_gpu);" for i in self.inputs + self.outputs)

        kernel_code = self.root_loop.generate_gpu(depth+1, block_depth, max_block_depth, new_variables, device_kernels)

        device_kernels = "\n".join(device_kernels)

        return f"""
{device_kernels}

{INDENT*depth}__global__ void kernel({inputs_str}, {outputs_str}, unsigned int N){{
{kernel_code}
{INDENT*depth}}}

{INDENT*depth}void {self.name_gpu}({inputs_str}, {outputs_str}, unsigned int N, dim3 block, dim3 grid, size_t size){{
{declare_gpu}

{malloc_gpu}

{copy_to_gpu}

{INDENT*depth}    kernel<<<block, grid>>>({input_output_call_gpu}, N);

{copy_from_gpu}

{free_gpu}
{INDENT*depth}}}
        """