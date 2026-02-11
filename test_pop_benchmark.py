#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Benchmark: pop from middle vs swap-with-end then pop on a 10M-item list.
Regenerates the list from scratch before each timed run (no reuse/caching).

When order isn't important, swap-with-end then pop is O(1) and the fastest
way to remove a random (or any) index from a list; no library does better.
"""

import random
import time

N = 10_000_000
MID = N // 2
# Empty-until-empty test: 10M would make pop-random O(n²) and take hours; reduce to run in reasonable time
N_EMPTY = 100_000


def fresh_list(size=N):
    """New list every time - no caching."""
    return list(range(size))


def bench_pop_middle():
    data = fresh_list()
    t0 = time.perf_counter()
    data.pop(MID)
    return time.perf_counter() - t0


def bench_swap_then_pop_end():
    data = fresh_list()
    t0 = time.perf_counter()
    data[MID], data[-1] = data[-1], data[MID]
    data.pop()
    return time.perf_counter() - t0


def bench_pop_random_index():
    data = fresh_list()
    i = random.randrange(len(data))
    t0 = time.perf_counter()
    data.pop(i)
    return time.perf_counter() - t0


def bench_swap_then_pop_random_index():
    data = fresh_list()
    i = random.randrange(len(data))
    t0 = time.perf_counter()
    data[i], data[-1] = data[-1], data[i]
    data.pop()
    return time.perf_counter() - t0


def bench_empty_by_pop_random():
    data = fresh_list(N_EMPTY)
    t0 = time.perf_counter()
    while data:
        data.pop(random.randrange(len(data)))
    return time.perf_counter() - t0


def bench_empty_by_swap_then_pop():
    data = fresh_list(N_EMPTY)
    t0 = time.perf_counter()
    while data:
        i = random.randrange(len(data))
        data[i], data[-1] = data[-1], data[i]
        data.pop()
    return time.perf_counter() - t0


if __name__ == "__main__":
    RUNS = 3
    print("=== 1) Fixed middle index (list size {}) ===".format(N))
    print("Runs: {}, fresh list every time, interleaved.".format(RUNS))
    print()

    pop_times = []
    swap_times = []
    for _ in range(RUNS):
        pop_times.append(bench_pop_middle())
        swap_times.append(bench_swap_then_pop_end())

    pop_min_us = min(pop_times) * 1e6
    print(
        "Pop from middle (index {}):  {:.2f} ms  (min of {} runs)".format(
            MID, pop_min_us / 1e3, RUNS
        )
    )
    swap_min_us = min(swap_times) * 1e6
    print(
        "Swap middle with end, then pop:  {:.2f} µs  (min of {} runs)".format(
            swap_min_us, RUNS
        )
    )
    if swap_min_us > 0:
        print(
            "Speedup (swap+pop vs pop middle): ~{:.0f}x".format(
                pop_min_us / swap_min_us
            )
        )
    print()

    print("=== 2) Random index, single pop (list size {}) ===".format(N))
    pop_r_times = []
    swap_r_times = []
    for _ in range(RUNS):
        pop_r_times.append(bench_pop_random_index())
        swap_r_times.append(bench_swap_then_pop_random_index())

    pop_r_min_us = min(pop_r_times) * 1e6
    print(
        "Pop at random index:  {:.2f} ms  (min of {} runs)".format(
            pop_r_min_us / 1e3, RUNS
        )
    )
    swap_r_min_us = min(swap_r_times) * 1e6
    print(
        "Swap random with end, then pop:  {:.2f} µs  (min of {} runs)".format(
            swap_r_min_us, RUNS
        )
    )
    if swap_r_min_us > 0:
        print("Speedup: ~{:.0f}x".format(pop_r_min_us / swap_r_min_us))
    print()

    print(
        "=== 3) Empty list by repeated random-index removal (list size {}) ===".format(
            N_EMPTY
        )
    )
    print(
        "(Set N_EMPTY=10_000_000 to use 10M; pop-random is O(n²) and will take hours.)"
    )
    print("Runs: 1 each (empty test is slow for pop-random).")
    empty_pop_t = bench_empty_by_pop_random()
    empty_swap_t = bench_empty_by_swap_then_pop()
    print("Empty by pop(random index):  {:.2f} s".format(empty_pop_t))
    print("Empty by swap(random, end) then pop:  {:.2f} s".format(empty_swap_t))
    if empty_swap_t > 0:
        print("Speedup: ~{:.0f}x".format(empty_pop_t / empty_swap_t))
