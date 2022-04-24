from autogoal.grammar import Union


from .arithmethic_block import ArithmeticAccumulationBlock, ArithmeticOperationBlock
from .loop_block import ForLoopBlock
from .logical_block import SyncThreadsBlock, TempStorageBlock
from .function_block import FunctionBlock
from .inter_block import InterBlock

from .base_block import NoneBlock

BLOCKS = [
    ArithmeticAccumulationBlock,
    ArithmeticOperationBlock,
    ForLoopBlock,
    TempStorageBlock,
    SyncThreadsBlock
]

def leaf_init(self, current: Union("current", *BLOCKS), next: Union("next", InterBlock, NoneBlock)) -> None:
        self.current = current
        self.next = next

InterBlock.__init__ = leaf_init