import decimal
from typing import Tuple

import sqlalchemy

from database.queries import Querier


class StateManager(object):
    def __init__(self, engine: sqlalchemy.Engine) -> None:
        self.engine = engine

    def init_schema(self):
        with open("database/schema.sql") as f:
            schema = f.read()

        with self.engine.connect() as conn:
            conn.execute(sqlalchemy.text(schema))
            conn.commit()

    def delete_cells_and_cell_scores(self):
        with self.engine.connect() as conn:
            q = Querier(conn)
            q.delete_cells_and_cell_scores()
            q.delete_sections()
            conn.commit()

    def insert_section(self, name: str, index: int):
        with self.engine.connect() as conn:
            q = Querier(conn)
            q.insert_section(name=name, index=index)
            conn.commit()

    def cell_exists(self, hash: str) -> bool:
        with self.engine.connect() as conn:
            q = Querier(conn)
            return q.cell_exists(hash=hash)

    def insert_initial_cell(
        self, hash: str, hash_input: str, max_no_ops: int | None, section_index: int, state: memoryview
    ):
        with self.engine.connect() as conn:
            q = Querier(conn)
            id = q.insert_cell(
                hash=hash,
                hash_input=hash_input,
                action=None,
                max_no_ops=max_no_ops,
                initial=True,
                section_index=section_index,
                state=state,
            )
            if id is None:
                return
            q.insert_cell_score(cell_id=id, score=decimal.Decimal(0))
            conn.commit()

    def insert_cell(
        self,
        hash: str,
        hash_input: str,
        action: int,
        max_no_ops: int | None,
        section_index: int,
        state: memoryview,
    ):
        with self.engine.connect() as conn:
            q = Querier(conn)
            id = q.insert_cell(
                hash=hash,
                hash_input=hash_input,
                action=action,
                max_no_ops=max_no_ops,
                initial=False,
                section_index=section_index,
                state=state,
            )
            if id is None:
                return
            q.insert_cell_score(cell_id=id, score=decimal.Decimal(0))
            conn.commit()

    def upsert_max_section(self, section_index: int):
        with self.engine.connect() as conn:
            q = Querier(conn)
            q.upsert_max_section(section_index=section_index)
            conn.commit()

    def update_max_section(self, section_index: int):
        with self.engine.connect() as conn:
            q = Querier(conn)
            q.update_max_section(section_index=section_index)
            conn.commit()

    def get_random_cell(self) -> Tuple[int, int, int, bool, memoryview]:
        with self.engine.connect() as conn:
            q = Querier(conn)
            result = q.get_random_cell()
            if result is None:
                result = q.get_first_cell()
            return result.id, result.action, result.max_no_ops, result.initial, result.state

    def get_first_cell(self, section_index: int) -> Tuple[int, int, int, bool, memoryview]:
        with self.engine.connect() as conn:
            q = Querier(conn)
            result = q.get_first_cell(section_index=section_index)
            return result.id, result.action, result.max_no_ops, result.initial, result.state

    def record_score(self, cell_id: int, score: float):
        with self.engine.connect() as conn:
            q = Querier(conn)
            q.insert_cell_score(cell_id=cell_id, score=decimal.Decimal(score))
            q.increment_cell_visit(id=cell_id)
            conn.commit()

    def get_cell(self, id: int) -> Tuple[int, int, int, bool, memoryview]:
        with self.engine.connect() as conn:
            q = Querier(conn)
            result = q.get_cell(id=id)
            return result.id, result.action, result.max_no_ops, result.initial, result.state

    def set_cell_invalid(self, id: int):
        with self.engine.connect() as conn:
            q = Querier(conn)
            q.set_cell_invalid(id=id)
            conn.commit()

    def delete_old_cell_scores(self):
        with self.engine.connect() as conn:
            q = Querier(conn)
            q.delete_old_cell_scores()
            conn.commit()

    def close(self):
        self.engine.dispose()
