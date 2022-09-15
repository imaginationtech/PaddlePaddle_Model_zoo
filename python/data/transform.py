
from .base import get_op


class Transform:
    def __init__(self, ops=None):
        if ops is None:
            return
        self.ops = []
        for op_pa in ops:
            op_name = list(op_pa)[0]
            op_params = op_pa.get(op_name, {})
            op = get_op(op_name)(**op_params)
            self.ops.append(op)

    def __call__(self, **kwargs):
        t = kwargs
        for op in self.ops:
            t = op(**t)
        return t

    def __add__(self, other):
        t = Transform([])
        t.ops.extend(self.ops)
        t.ops.extend(other.ops)
        return t
