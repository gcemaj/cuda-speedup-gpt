from collections import defaultdict
from typing import Any, Dict, Set, Tuple
from autogoal.grammar import generate_cfg
from dataclasses import dataclass

import subprocess
import pandas as pd

from nyu.gpu.speedup.generator import GenerationException
from nyu.gpu.speedup.generator.variables import VariableType
from nyu.gpu.speedup.generator.blocks import FunctionBlock
from nyu.gpu.speedup.generator.driver import TEMPLATE

@dataclass
class ExecutionPoint:
    gpu_src : str
    cpu_src : str
    cpu_exec : float
    gpu_exec : float
    errors : int
    n : int
    b1 : int
    b2: int
    b3 : int
    g1 : int
    g2: int
    g3 : int

class Driver:

    def __init__(self) -> None:
        self.cfg = generate_cfg(FunctionBlock)

    def generate_code(self, cpu_vars : Dict[str, Set[str]], gpu_vars : Dict[str, Set[str]]) -> Tuple[str, str, int, Any, Any]:    
        while True:
            try:
                sample = self.cfg.sample()
                cpu = sample.generate(0, cpu_vars)
                depth = sample.compute_depth()
                gpu = sample.generate_gpu(0, 0, min(depth, 3), gpu_vars, [])
                return cpu, gpu, min(depth, 3), sample.inputs, sample.outputs
            except GenerationException:
                pass

    def generate_block_dim(self, max_size : int) -> str:
        match max_size:
            case 1:
                return "dim3 block((N+1)/1024, 1, 1);"
            case 2:
                return "dim3 block((N+1)/32, (N+1)/32, 1);"
            case 3:
                return "dim3 block((N+1)/16, (N+1)/8, (N+1)/8);"
            case _:
                return ""
    
    def generate_grid_dim(self, max_size : int) -> str:
        match max_size:
            case 1:
                return "dim3 grid(1024, 1, 1);"
            case 2:
                return "dim3 grid(32, 32, 1);"
            case 3:
                return "dim3 grid(16, 8, 8);"
            case _:
                return ""



    def generate_data(self, iterations: int):
        data = []
        for _ in range(iterations):
            cpu_vars = defaultdict(set)            
            gpu_vars = defaultdict(set)
            cpu, gpu, max_size, func_inputs, func_outputs = self.generate_code(cpu_vars, gpu_vars)

            size = '*'.join(['N']*max_size)

            init_cpu = "\n  ".join([f"float *{i}_cpu;" for i in func_inputs + func_outputs])
            init_gpu = "\n  ".join([f"float *{i}_gpu;" for i in func_inputs + func_outputs])

            malloc_cpu = "\n  ".join([f"{i}_cpu = (float *)malloc({size}*sizeof(float));" for i in func_inputs + func_outputs])
            malloc_gpu = "\n  ".join([f"{i}_gpu = (float *)malloc({size}*sizeof(float));" for i in func_inputs + func_outputs])

            random_init = "\n    ".join([f"{i}_cpu[i] = {i}_gpu[i] = (((float)rand()/(float)(RAND_MAX)) * 100.0f);" for i in func_inputs])

            cpu_call = ", ".join([f"{i}_cpu" for i in func_inputs + func_outputs])
            gpu_call = ", ".join([f"{i}_gpu" for i in func_inputs + func_outputs])

            free_cpu = "\n  ".join([f"free({i}_cpu);" for i in func_inputs + func_outputs])
            free_gpu = "\n  ".join([f"free({i}_gpu);" for i in func_inputs + func_outputs])

            compare = "\n  ".join([f"bad_count += {i}_cpu[i] == {i}_gpu[i]? 0 : 1;" for i in func_outputs])


            source_code = TEMPLATE.format(
                cpu=cpu, 
                gpu=gpu,
                init_cpu=init_cpu,
                init_gpu=init_gpu,
                malloc_cpu=malloc_cpu,
                malloc_gpu=malloc_gpu,
                random_init=random_init,
                size=size,
                compare=compare,
                cpu_call=cpu_call,
                gpu_call=gpu_call,
                free_cpu="",#free_cpu,
                free_gpu="",#free_gpu,
            )

            with open("dummy.cu", "w") as src_file:
                src_file.write(source_code)

            subprocess.call("nvcc -o dummy ./dummy.cu")

            calls = []
            match max_size:
                case 3:
                    calls = [
                        [10, 10, 10, 10, 1, 1, 1],
                        [100, 100, 100, 100, 1, 1, 1],
                        [100, 10, 10, 10, 10, 10, 10],
                        [100, 1, 1, 1, 100, 100, 100],
                        [10000, 10000, 10000, 10000, 1, 1, 1],
                        [10000, 1000, 1000, 1000, 10, 10, 10],
                    ]
                case 2:
                    calls = [
                        [10, 10, 10, 1, 1, 1, 1],
                        [100, 100, 100, 1, 1, 1, 1],
                        [100, 10, 10, 1, 10, 10, 1],
                        [100, 1, 1, 1, 100, 100, 1],
                        [10000, 10000, 10000, 1, 1, 1, 1],
                        [10000, 1000, 1000, 1, 10, 10, 1],
                        [10000, 100, 100, 1, 100, 100, 1],
                        [100000, 10000, 10000, 1, 10, 10, 1],
                        [100000, 1000, 1000, 1, 100, 10, 1],
                        [100000, 100, 100, 1, 1000, 1000, 1],
                        [100000, 1000, 1000, 1, 100, 100, 1],
                    ]
                case 1:
                    calls = [
                        [10, 10, 1, 1, 1, 1, 1],
                        [100, 100, 1, 1, 1, 1, 1],
                        [100, 10, 1, 1, 10, 1, 1],
                        [100, 1, 1, 1, 100, 1, 1],
                        [10000, 10000, 1, 1, 1, 1, 1],
                        [10000, 1000, 1, 1, 10, 1, 1],
                        [10000, 100, 1, 1, 100, 1, 1],
                        [100000, 10000, 1, 1, 10, 1, 1],
                        [100000, 1000, 1, 1, 100, 1, 1],
                        [100000, 100, 1, 1, 1000, 1, 1],
                        [100000, 1000, 1, 1, 100, 1, 1],
                    ]
            for call in calls:
                child = subprocess.Popen(["./dummy"] + call, stdout=subprocess.PIPE, shell=True)
                msg, _ = child.communicate()
                [cpu_time, gpu_time, errs] = msg.split("\n")

                data.append(
                    ExecutionPoint(
                        gpu,
                        cpu,
                        cpu_time.split("=")[1],
                        gpu_time.split("=")[1],
                        errs.split("=")[1],
                        *call
                    )
                )
        pd.DataFrame(data=data).to_csv("cuda_speedup.csv")

