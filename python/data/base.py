# Copyright (c) 2022 Imagination Technologies Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

