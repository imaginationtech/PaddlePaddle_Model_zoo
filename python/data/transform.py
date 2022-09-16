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
