SELECT COUNT(*)
FROM opinions;

SELECT COUNT(*)
FROM opinions
WHERE opinions.cap_author IS NOT NULL;

SELECT COUNT(*)
FROM opinions
WHERE opinions.author IS NOT NULL;

SELECT *
FROM cases
ORDER BY decision_date;