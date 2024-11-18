CREATE TABLE IF NOT EXISTS cells (
    id            SERIAL   PRIMARY KEY,
    hash          TEXT     UNIQUE NOT NULL,
    hash_input    TEXT     NOT NULL,
    action        INTEGER,
    max_no_ops    INTEGER,
    initial       BOOLEAN  NOT NULL,
    section_index INTEGER  NOT NULL REFERENCES sections(index),
    visits        INTEGER  NOT NULL DEFAULT 0,
    invalid       BOOLEAN  NOT NULL DEFAULT FALSE,
    state         BYTEA    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cells_section_index ON cells(section_index);
CREATE INDEX IF NOT EXISTS idx_cells_invalid ON cells(invalid);

CREATE TABLE IF NOT EXISTS cell_scores (
    id          SERIAL         PRIMARY KEY,
    cell_id     INTEGER        NOT NULL REFERENCES cells(id),
    epoch       INTEGER        NOT NULL,
    score       REAL           NOT NULL,
    length      INTEGER        NOT NULL,
    placeholder BOOLEAN        NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_cell_scores_cell_id ON cell_scores(cell_id);

CREATE TABLE IF NOT EXISTS sections (
    name  TEXT PRIMARY KEY,
    index INTEGER UNIQUE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sections_index ON sections(index);

CREATE TABLE IF NOT EXISTS max_sections (
    id            INTEGER PRIMARY KEY,
    section_index INTEGER UNIQUE REFERENCES sections(index)
);

CREATE TABLE IF NOT EXISTS epochs (
    id    INTEGER PRIMARY KEY,
    epoch INTEGER UNIQUE
);

CREATE TABLE IF NOT EXISTS cell_score_metrics (
    id          SERIAL PRIMARY KEY,
    epoch       INTEGER NOT NULL,
    cell_id     INTEGER NOT NULL REFERENCES cells(id),
    min_score   REAL    NOT NULL,
    max_score   REAL    NOT NULL,
    mean_score  REAL    NOT NULL,
    std_score   REAL    NOT NULL,
    min_length  REAL    NOT NULL,
    max_length  REAL    NOT NULL,
    mean_length REAL    NOT NULL,
    std_length  REAL    NOT NULL,
    visits      INTEGER NOT NULL
);
