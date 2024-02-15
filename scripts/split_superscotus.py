"""Gets Super-SCOTUS cases in range [1986,2019].

1986 is the start of the Rehnquist court and makes a good starting point for
analysis.

Super-SCOTUS paper: https://aclanthology.org/2023.nllp-1.20
Super-SCOTUS dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/POWQIT
"""
import jsonlines

selected_years = []
source = "data/super_scotus/case_with_all_sources_with_companion_cases_tag.jsonl"
with jsonlines.open(source, "r") as reader:
    for i, line in enumerate(reader):
        year = line["year"]
        if year >= 1986 and year <= 2019:
            selected_years.append(line)

target = "data/super_scotus/1986_to_2019.jsonl"
with jsonlines.open(target, "w") as f:
    num_cases = len(selected_years)
    f.write_all(selected_years)
    print(f"{num_cases} written to {target}")
