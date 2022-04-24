from enum import Enum
import namesgenerator as n_gen

class VariableType(Enum):
    ARRAY_INPUT = 0
    ARRAY_OUTPUT = 1
    ARRAY_TEMP = 2
    SINGLE = 3
    INDEX = 4
    SHARED = 5

class Variable:

    def __init__(self, var_type: VariableType, name : str = None) -> None:
        self.type = var_type
        self.name = name or n_gen.get_random_name()
        if var_type == VariableType.ARRAY_INPUT:
            self.name += "_input"
        elif var_type == VariableType.ARRAY_OUTPUT:
            self.name += "_output"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name

    def __str__(self) -> str:
        return self.name
