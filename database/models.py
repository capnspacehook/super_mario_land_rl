# Code generated by sqlc. DO NOT EDIT.
# versions:
#   sqlc v1.27.0
import dataclasses
import decimal
from typing import Optional


@dataclasses.dataclass()
class Cell:
    id: int
    hash: str
    hash_input: str
    action: Optional[int]
    max_no_ops: Optional[int]
    initial: bool
    section_index: int
    visits: int
    invalid: bool
    state: memoryview


@dataclasses.dataclass()
class CellScore:
    id: int
    cell_id: int
    score: decimal.Decimal


@dataclasses.dataclass()
class MaxSection:
    id: int
    section_index: Optional[int]


@dataclasses.dataclass()
class Section:
    name: str
    index: int
