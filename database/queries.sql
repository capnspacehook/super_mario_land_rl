-- name: DeleteCellsAndCellScores :exec
TRUNCATE cells CASCADE;

-- name: DeleteSections :exec
TRUNCATE sections CASCADE;

-- name: InsertSection :exec
INSERT INTO sections (name, index)
VALUES ($1, $2);

-- name: UpsertMaxSection :exec
INSERT INTO max_sections (id, section_index)
VALUES (1, $1)
ON CONFLICT (id) DO UPDATE
SET section_index = $1;

-- name: UpdateMaxSection :exec
UPDATE max_sections
SET section_index = $1
WHERE id = 1;

-- name: UpsertEpoch :exec
INSERT INTO epochs (id, epoch)
VALUES (1, 0)
ON CONFLICT (id) DO UPDATE
SET epoch = 0;

-- name: IncrementEpoch :exec
UPDATE epochs
SET epoch = epoch + 1
WHERE id = 1;

-- name: CellExists :one
SELECT EXISTS(
    SELECT 1
    FROM cells
    WHERE hash = $1
);

-- name: GetRandomCell :one
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
        WHERE c.section_index <= max_section.max_section AND c.invalid = FALSE
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
JOIN rand_id AS ri ON ri.id = c.id WHERE c.id = ri.id;

-- name: GetFirstCell :one
SELECT id, action, max_no_ops, initial, state
FROM cells
WHERE section_index = $1 AND initial = TRUE
LIMIT 1;

-- name: GetCell :one
SELECT id, action, max_no_ops, initial, state
FROM cells
WHERE id = $1;

-- name: InsertCell :one
INSERT INTO cells (
    hash, hash_input, action, max_no_ops, initial, section_index, state
) VALUES (
    $1, $2, $3, $4, $5, $6, $7
)
ON CONFLICT DO NOTHING
RETURNING id;

-- name: InsertCellScore :exec
INSERT INTO cell_scores (
    cell_id, epoch, score, length
) VALUES (
    $1,
    (SELECT epoch FROM epochs LIMIT 1),
    $2,
    $3
);

-- name: InsertPlaceholderCellScore :exec
INSERT INTO cell_scores (
    cell_id, epoch, score, length, placeholder
) VALUES (
    $1,
    (SELECT epoch FROM epochs LIMIT 1),
    0.0,
    0,
    TRUE
);

-- name: IncrementCellVisit :exec
UPDATE cells
SET visits = visits + 1
WHERE id = $1;

-- name: SetCellInvalid :exec
UPDATE cells
SET invalid = TRUE
WHERE id = $1;

-- name: InsertCellScoreMetrics :exec
WITH aggregated_scores AS (
    SELECT 
        epoch,
        cell_id,
        MIN(score) AS min_score,
        MAX(score) AS max_score,
        AVG(score) AS mean_score,
        STDDEV_POP(score) AS std_score,
        MIN(length) AS min_length,
        MAX(length) AS max_length,
        AVG(length) AS mean_length,
        STDDEV_POP(length) AS std_length,
        COUNT(score) AS visits
    FROM cell_scores
    WHERE epoch = (SELECT epoch FROM epochs LIMIT 1) AND placeholder = FALSE
    GROUP BY epoch, cell_id
)
INSERT INTO cell_score_metrics (
    epoch,
    cell_id,
    min_score,
    max_score,
    mean_score,
    std_score,
    min_length,
    max_length,
    mean_length,
    std_length,
    visits
)
SELECT
    ag.epoch,
    ag.cell_id,
    ag.min_score,
    ag.max_score,
    ag.mean_score,
    ag.std_score,
    ag.min_length,
    ag.max_length,
    ag.mean_length,
    ag.std_length,
    ag.visits
FROM aggregated_scores AS ag
JOIN cells AS c ON ag.cell_id = c.id;

-- name: DeleteOldCellScores :exec
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
);
