"""Optional serial float32 kernels matching NumPy 1.26.4 prefix reductions.

The reduction order follows numpy/core/src/umath/loops_utils.h.src (NumPy,
BSD-3-Clause). Buffers and eight-lane grouping are numerical semantics, not
model parameters. No fastmath, parallel scheduler or global thread mutation.
"""

# NumPy reduction-order attribution and redistribution notice:
# Copyright (c) 2005-2023, NumPy Developers. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the NumPy Developers nor the names of any contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic
from llvmlite import ir


BUFFER_SIZE = 8192
PAIRWISE_BLOCK = 128
LANES = 8


@intrinsic
def _eight_lane_sum(typing_context, values, start, blocks, mean):
    """SIMD across independent lanes, never reassociate a lane's additions."""
    if values.dtype != types.float32:
        raise TypeError('Native prefix reductions require float32 input')
    signature = types.float32(values, start, blocks, mean)

    def codegen(context, builder, sig, args):
        array = context.make_array(sig.args[0])(context, builder, args[0])
        vector_type = ir.VectorType(ir.FloatType(), LANES)
        offset, count = args[1:3]
        mean_vector = ir.Constant(vector_type, ir.Undefined)
        if not isinstance(sig.args[3], types.NoneType):
            for lane in range(LANES):
                mean_vector = builder.insert_element(mean_vector, args[3], ir.Constant(ir.IntType(32), lane))

        def load(index):
            if sig.args[0].layout == 'C':
                pointer = builder.gep(array.data, [index])
                vector = builder.load(builder.bitcast(pointer, vector_type.as_pointer()), align=4)
            else:
                vector = ir.Constant(vector_type, ir.Undefined)
                for lane in range(LANES):
                    item = builder.add(index, context.get_constant(types.intp, lane))
                    pointer = cgutils.get_item_pointer(context, builder, sig.args[0], array, [item],
                                                       wraparound=False, boundscheck=False)
                    vector = builder.insert_element(vector, builder.load(pointer), ir.Constant(ir.IntType(32), lane))
            if not isinstance(sig.args[3], types.NoneType):
                delta = builder.fsub(vector, mean_vector)
                vector = builder.fmul(delta, delta)
            return vector

        total = cgutils.alloca_once(builder, vector_type)
        builder.store(load(offset), total)
        with cgutils.for_range(builder, builder.sub(count, context.get_constant(types.intp, 1))) as loop:
            index = builder.add(offset, builder.mul(builder.add(loop.index, context.get_constant(types.intp, 1)),
                                                    context.get_constant(types.intp, LANES)))
            builder.store(builder.fadd(builder.load(total), load(index)), total)
        vector = builder.load(total)
        lanes = [builder.extract_element(vector, ir.Constant(ir.IntType(32), lane)) for lane in range(LANES)]
        pairs = [builder.fadd(lanes[lane], lanes[lane + 1]) for lane in range(0, LANES, 2)]
        return builder.fadd(builder.fadd(pairs[0], pairs[1]), builder.fadd(pairs[2], pairs[3]))

    return signature, codegen


@njit(cache=True, fastmath=False, inline='always')
def _mapped(value, mean):
    if mean is None:
        return value
    delta = np.float32(value - mean)
    return np.float32(delta * delta)


@njit(cache=True, fastmath=False, inline='always')
def _pairwise_leaf(values, start, size, mean):
    if size < LANES:
        value = np.float32(-0.)
        for index in range(start, start + size):
            value = np.float32(value + _mapped(values[index], mean))
        return value
    full = size // LANES * LANES
    value = _eight_lane_sum(values, start, full // LANES, mean)
    for offset in range(full, size):
        value = np.float32(value + _mapped(values[start + offset], mean))
    return value


@njit(cache=True, fastmath=False)
def _pairwise(values, start, size, mean):
    if size <= PAIRWISE_BLOCK:
        return _pairwise_leaf(values, start, size, mean)
    middle = size // 2 // LANES * LANES
    return np.float32(_pairwise(values, start, middle, mean)
                      + _pairwise(values, start + middle, size - middle, mean))


@njit(cache=True, fastmath=False)
def _buffered_sum(values, size, mean):
    total = np.float32(0.)
    for start in range(0, size, BUFFER_SIZE):
        total = np.float32(total + _pairwise(values, start, min(BUFFER_SIZE, size - start), mean))
    return total


@njit(cache=True, fastmath=False)
def _cached_input_sum(values, start, size, cache):
    key = start % BUFFER_SIZE // LANES
    stored = size < cache.shape[1]
    if stored and not np.isnan(cache[key, size]):
        return cache[key, size]
    if size <= PAIRWISE_BLOCK:
        result = _pairwise_leaf(values, start, size, None)
    else:
        middle = size // 2 // LANES * LANES
        result = np.float32(_cached_input_sum(values, start, middle, cache)
                            + _cached_input_sum(values, start + middle, size - middle, cache))
    if stored:
        cache[key, size] = result
    return result


@njit(cache=True, fastmath=False)
def _prefix_means(values):
    block = min(BUFFER_SIZE, len(values))
    # At most ~16 MiB, discarded after one row. Only raw-input sums are reusable;
    # squared deviations depend on the current mean and are never reused.
    cache = np.empty((block // LANES + 1, block // 2 + 1), dtype=np.float32)
    means = np.empty(len(values), dtype=np.float32)
    complete = np.float32(0.)
    for start in range(0, len(values), BUFFER_SIZE):
        cache[:] = np.nan
        size = min(BUFFER_SIZE, len(values) - start)
        for count in range(1, size + 1):
            partial = _cached_input_sum(values, start, count, cache)
            total = np.float32(complete + partial)
            means[start + count - 1] = np.float32(np.float64(total) / (start + count))
        complete = np.float32(complete + _cached_input_sum(values, start, size, cache))
    return means


@njit(cache=True, fastmath=False, error_model='numpy')
def envelope_ranges(cumulative, averages):
    """Keep rounded extrema; discard only points safely dominated by endpoints."""
    size = len(cumulative)
    upper, lower = np.empty(size, dtype=np.int64), np.empty(size, dtype=np.int64)
    upper_size = lower_size = 0
    ranges = np.empty(size)
    scale = np.max(np.abs(cumulative)) + size * np.max(np.abs(averages))
    # Retain near-collinear points. The conservative margin covers both the
    # orientation calculation and rounding of y - t*average (no fused multiply).
    margin = 64 * np.finfo(np.float64).eps * scale
    for index in range(size):
        for sign in (1, -1):
            hull = upper if sign == 1 else lower
            count = upper_size if sign == 1 else lower_size
            while count >= 2:
                left, middle = hull[count - 2], hull[count - 1]
                span = np.float64(index - left)
                area = ((np.float64(cumulative[middle]) - np.float64(cumulative[left])) * span
                        - (np.float64(cumulative[index]) - np.float64(cumulative[left])) * (middle - left))
                if sign * area >= -margin * span:
                    break
                count -= 1
            hull[count] = index
            if sign == 1:
                upper_size = count + 1
            else:
                lower_size = count + 1
        low, high = np.inf, -np.inf
        average = averages[index]
        for offset in range(upper_size):
            point = upper[offset]
            centered = np.float64(cumulative[point]) - np.float64(point + 1) * average
            high = max(high, centered)
        for offset in range(lower_size):
            point = lower[offset]
            centered = np.float64(cumulative[point]) - np.float64(point + 1) * average
            low = min(low, centered)
        ranges[index] = high - low
    return ranges


@njit(cache=True, fastmath=False, error_model='numpy')
def prefixes(values, cumulative, averages):
    deviations = np.empty(len(values))
    means = _prefix_means(values)
    for count in range(1, len(values) + 1):
        variance = np.float32(np.float64(_buffered_sum(values, count, means[count - 1])) / count)
        deviations[count - 1] = np.sqrt(variance)
    return deviations, envelope_ranges(cumulative, averages)


@njit(cache=True, fastmath=False)
def hjorth_m4(values):
    total = np.float64(0.)
    previous = np.float32(values[1] - values[0])
    for index in range(1, len(values) - 1):
        current = np.float32(values[index + 1] - values[index])
        delta = np.float64(np.float32(current - previous))
        total += delta * delta
        previous = current
    return total / (len(values) - 2)
