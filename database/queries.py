# Code generated by sqlc. DO NOT EDIT.
# versions:
#   sqlc v1.27.0
# source: queries.sql
import dataclasses
import decimal
from typing import Optional

import sqlalchemy

from database import models


CELL_EXISTS = """-- name: cell_exists \\:one
SELECT EXISTS(
    SELECT 1
    FROM cells
    WHERE hash = :p1
)
"""


DELETE_CELLS_AND_CELL_SCORES = """-- name: delete_cells_and_cell_scores \\:exec
TRUNCATE cells CASCADE
"""


DELETE_OLD_CELL_SCORES = """-- name: delete_old_cell_scores \\:exec
WITH ranked_scores AS (
    SELECT 
        id,
        cell_id,
        ROW_NUMBER() OVER (PARTITION BY cell_id ORDER BY id DESC) AS rn
    FROM cell_scores
)
DELETE FROM cell_scores
WHERE id IN (
    SELECT id
    FROM ranked_scores
    WHERE rn > 10
)
"""


DELETE_SECTIONS = """-- name: delete_sections \\:exec
TRUNCATE sections CASCADE
"""


GET_CELL = """-- name: get_cell \\:one
SELECT id, action, max_no_ops, initial, state
FROM cells
WHERE id = :p1
"""


@dataclasses.dataclass()
class GetCellRow:
    id: int
    action: Optional[int]
    max_no_ops: Optional[int]
    initial: bool
    state: memoryview


GET_FIRST_CELL = """-- name: get_first_cell \\:one
SELECT id, action, max_no_ops, initial, state
FROM cells
WHERE section_index = :p1 AND initial = TRUE
LIMIT 1
"""


@dataclasses.dataclass()
class GetFirstCellRow:
    id: int
    action: Optional[int]
    max_no_ops: Optional[int]
    initial: bool
    state: memoryview


GET_RANDOM_CELL = """-- name: get_random_cell \\:one
WITH max_section AS (
    SELECT section_index AS max_section 
    FROM max_sections
    WHERE id = 1
    LIMIT 1
), mean_scores AS (
    -- get mean of last 10 scores of all cells in certain sections
    SELECT cell_id, AVG(score) AS mean_score
    FROM (
        SELECT 
            cs.cell_id,
            cs.score,
            ROW_NUMBER() OVER (PARTITION BY cs.cell_id ORDER BY cs.id DESC) AS rn
        FROM cell_scores AS cs
        JOIN cells AS c
        ON c.id = cs.cell_id
        CROSS JOIN max_section
        WHERE c.section_index <= max_section.max_section and c.invalid = FALSE
    ) AS q
    WHERE rn <= 10
    GROUP BY cell_id
), norm_scores AS (
    -- normalize mean scores to be between 0 and 100
    SELECT 
        cell_id,
        -- ensure we aren't dividing by 0
        ((mean_score - min_score) / COALESCE(NULLIF(max_score - min_score, 0), 1)) * 100 AS norm_score
    FROM (
        SELECT 
            cell_id,
            mean_score,
            MIN(mean_score) OVER () AS min_score,
            MAX(mean_score) OVER () AS max_score
        FROM mean_scores
    ) AS q
), weights AS (
    -- create weights for each cell based on number of visits and normalized score
    -- less visits and a lower normalized score results in a higher weight
    -- normalized score is prioritized over number of visits
    SELECT 
        c.id AS id,
        (100 / SQRT(c.visits + 1)) + (
            -- subtract normalized scores by max normalized score so
            -- cells with a greater normalized score have less weight
            (
                SELECT MAX(norm_score) AS max_score
                FROM norm_scores
            ) - SUM(ns.norm_score)
        ) +
        CASE
            -- add 5% of the max possible weight to cells in the current section
            WHEN c.section_index = MAX(max_section.max_section) THEN 10
            ELSE 0 
        END AS weight
    FROM cells AS c
    JOIN norm_scores AS ns
    ON ns.cell_id = c.id
    CROSS JOIN max_section
    GROUP BY c.id
), rand_pick AS (
    -- create value that will be used to pick a random cell
    -- multiply random number by sum of all weights so the weights don't have to add up to 100
    SELECT random() * (SELECT SUM(weight) FROM weights) AS pick
), rand_id AS (
    SELECT id
    FROM (
        SELECT id, SUM(weight) OVER (ORDER BY id) AS scaled_weight, pick
        FROM weights CROSS JOIN rand_pick
    ) AS q
    WHERE scaled_weight >= pick
    ORDER BY id
    LIMIT 1
)
SELECT c.id, action, max_no_ops, initial, state
FROM cells AS c
JOIN rand_id AS ri ON ri.id = c.id WHERE c.id = ri.id
"""


@dataclasses.dataclass()
class GetRandomCellRow:
    id: int
    action: Optional[int]
    max_no_ops: Optional[int]
    initial: bool
    state: memoryview


INCREMENT_CELL_VISIT = """-- name: increment_cell_visit \\:exec
UPDATE cells
SET visits = visits + 1
WHERE id = :p1
"""


INSERT_CELL = """-- name: insert_cell \\:one
INSERT INTO cells (
    hash, hash_input, action, max_no_ops, initial, section_index, state
) VALUES (
    :p1, :p2, :p3, :p4, :p5, :p6, :p7
)
ON CONFLICT DO NOTHING
RETURNING id
"""


INSERT_CELL_SCORE = """-- name: insert_cell_score \\:exec
INSERT INTO cell_scores (
    cell_id, score
) VALUES (
    :p1, :p2
)
"""


INSERT_SECTION = """-- name: insert_section \\:exec
INSERT INTO sections (name, index)
VALUES (:p1, :p2)
"""


SET_CELL_INVALID = """-- name: set_cell_invalid \\:exec
UPDATE cells
SET invalid = TRUE
where id = :p1
"""


UPDATE_MAX_SECTION = """-- name: update_max_section \\:exec
UPDATE max_sections
SET section_index = :p1
WHERE id = 1
"""


UPSERT_MAX_SECTION = """-- name: upsert_max_section \\:exec
INSERT INTO max_sections (id, section_index)
VALUES (1, :p1)
ON CONFLICT (id) DO UPDATE
SET section_index = :p1
"""


class Querier:
    def __init__(self, conn: sqlalchemy.engine.Connection):
        self._conn = conn

    def cell_exists(self, *, hash: str) -> Optional[bool]:
        row = self._conn.execute(sqlalchemy.text(CELL_EXISTS), {"p1": hash}).first()
        if row is None:
            return None
        return row[0]

    def delete_cells_and_cell_scores(self) -> None:
        self._conn.execute(sqlalchemy.text(DELETE_CELLS_AND_CELL_SCORES))

    def delete_old_cell_scores(self) -> None:
        self._conn.execute(sqlalchemy.text(DELETE_OLD_CELL_SCORES))

    def delete_sections(self) -> None:
        self._conn.execute(sqlalchemy.text(DELETE_SECTIONS))

    def get_cell(self, *, id: int) -> Optional[GetCellRow]:
        row = self._conn.execute(sqlalchemy.text(GET_CELL), {"p1": id}).first()
        if row is None:
            return None
        return GetCellRow(
            id=row[0],
            action=row[1],
            max_no_ops=row[2],
            initial=row[3],
            state=row[4],
        )

    def get_first_cell(self, *, section_index: int) -> Optional[GetFirstCellRow]:
        row = self._conn.execute(sqlalchemy.text(GET_FIRST_CELL), {"p1": section_index}).first()
        if row is None:
            return None
        return GetFirstCellRow(
            id=row[0],
            action=row[1],
            max_no_ops=row[2],
            initial=row[3],
            state=row[4],
        )

    def get_random_cell(self) -> Optional[GetRandomCellRow]:
        row = self._conn.execute(sqlalchemy.text(GET_RANDOM_CELL)).first()
        if row is None:
            return None
        return GetRandomCellRow(
            id=row[0],
            action=row[1],
            max_no_ops=row[2],
            initial=row[3],
            state=row[4],
        )

    def increment_cell_visit(self, *, id: int) -> None:
        self._conn.execute(sqlalchemy.text(INCREMENT_CELL_VISIT), {"p1": id})

    def insert_cell(self, *, hash: str, hash_input: str, action: Optional[int], max_no_ops: Optional[int], initial: bool, section_index: int, state: memoryview) -> Optional[int]:
        row = self._conn.execute(sqlalchemy.text(INSERT_CELL), {
            "p1": hash,
            "p2": hash_input,
            "p3": action,
            "p4": max_no_ops,
            "p5": initial,
            "p6": section_index,
            "p7": state,
        }).first()
        if row is None:
            return None
        return row[0]

    def insert_cell_score(self, *, cell_id: int, score: decimal.Decimal) -> None:
        self._conn.execute(sqlalchemy.text(INSERT_CELL_SCORE), {"p1": cell_id, "p2": score})

    def insert_section(self, *, name: str, index: int) -> None:
        self._conn.execute(sqlalchemy.text(INSERT_SECTION), {"p1": name, "p2": index})

    def set_cell_invalid(self, *, id: int) -> None:
        self._conn.execute(sqlalchemy.text(SET_CELL_INVALID), {"p1": id})

    def update_max_section(self, *, section_index: Optional[int]) -> None:
        self._conn.execute(sqlalchemy.text(UPDATE_MAX_SECTION), {"p1": section_index})

    def upsert_max_section(self, *, section_index: Optional[int]) -> None:
        self._conn.execute(sqlalchemy.text(UPSERT_MAX_SECTION), {"p1": section_index})
