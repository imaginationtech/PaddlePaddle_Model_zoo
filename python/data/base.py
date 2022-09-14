
class OpBase:
    pass


registry = {}


def op_register(op_name, op=None):

    if callable(op_name):
        op = op_name
        op_name = op_name.__name__

    def register(myo):
        t = registry.get(op_name, None)
        if t is None:
            registry[op_name] = myo
        return myo

    if op:
        return register(op)

    return register


def get_op(op_name):
    op = registry.get(op_name, None)
    if op is None:
        raise ValueError("it not register op %s" % op_name)
    return op

