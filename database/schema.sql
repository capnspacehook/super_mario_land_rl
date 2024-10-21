CREATE TABLE IF NOT EXISTS cells (
    id         SERIAL   PRIMARY KEY,
    hash       TEXT     UNIQUE NOT NULL,
    hash_input TEXT     NOT NULL,
    action     INTEGER,
    max_no_ops INTEGER,
    initial    BOOLEAN  NOT NULL,
    section    TEXT     NOT NULL,
    visits     INTEGER  NOT NULL DEFAULT 0,
    invalid    BOOLEAN  NOT NULL DEFAULT FALSE,
    state      BYTEA    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_section ON cells(section);
CREATE INDEX IF NOT EXISTS idx_invalid ON cells(invalid);

CREATE TABLE IF NOT EXISTS cell_scores (
    id      SERIAL         PRIMARY KEY,
    cell_id INTEGER        NOT NULL REFERENCES cells(id),
    score   NUMERIC(10, 5) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cell_id ON cell_scores(cell_id);

CREATE TABLE IF NOT EXISTS max_sections (
    id      INTEGER PRIMARY KEY,
    section TEXT UNIQUE
);
