#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019-2023 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides classes for mutation testing.

Comes from https://github.com/se2p/mutpy-pynguin/blob/main/mutpy/utils.py.
"""
from pynguin.utils import randomness


class RandomSampler:
    def __init__(self, percentage: int) -> None:
        assert 0 <= percentage and percentage <= 100
        self.percentage = percentage

    def is_mutation_time(self) -> bool:
        return randomness.next_int(0, 100) < self.percentage
