-- drop index before table
DROP INDEX IF EXISTS idx_cases_docket_number;
DROP TABLE IF EXISTS cases;
DROP TABLE IF EXISTS opinions;

CREATE TABLE IF NOT EXISTS cases (
    docket_number UNIQUE,
    case_status,
    selected_case_id,
    decision_date
);

CREATE INDEX IF NOT EXISTS idx_cases_docket_number ON cases (docket_number);

CREATE TABLE IF NOT EXISTS opinions (
    docket_number,
    opinion_number,
    cap_author,
    author
);