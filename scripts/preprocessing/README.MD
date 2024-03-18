
## Sourcing SCOTUS from Harvard's [Caselaw Access Project (CAP)](https://case.law/)
### Background
Goal: retrieve all opinions written by the Supreme Court for a specified year range.

SCOTUS denies thousands of cases every year, and each denial gets its own document, so we can't just grab all SCOTUS documents from CAP for a specified year. We need docket numbers for the cases that granted cert and argued before the court. Here, we source those docket numbers from the [Super-SCOTUS dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/POWQIT) [[paper](https://aclanthology.org/2023.nllp-1.20/)].

1. Get docket numbers for the years 1986-2019 from superscotus (TODO: rewrite, source is now SCDB directly)
2. For each year, request a small sample (~15) cases from CAP. (waiting on unmetered API access before pulling full set)


There also seem to be two maps from SCDB to CAP:
1. [Connecting U.S. Supreme Court Case Information and Opinion Authorship (SCDB) to Full Case Text Data (CAP), 1791-2011](https://zenodo.org/records/4344917).
1. CAP's own [matchup file](https://case.law/download/scdb/scdb_matchup_2020-01-16.csv) on this [page](https://case.law/download/scdb/).

The maps from SCDB end in 2011. From 2007 onward, the SCOTUS website has documents with docket numbers for cases granted argument. These are the **Granted \& Noted** [lists](https://www.supremecourt.gov/orders/grantednotedlists.aspx).

### Instructions
From the base of the repo, run `cap_bulk_download.py` to download data from CAP:

```
../scotus-metalang$ python scripts/preprocessing/cap_bulk_download.py
```

Then run `get_cases_from_bulk_cap.py` to map SCDB cases to the downloaded CAP files:
```
../scotus-metalang$ python scripts/preprocessing/get_cases_from_bulk_cap.py
```

Finally, run `extract_and_split_opinions.py.py` to extract/split opinions from the selected CAP files:
```
../scotus-metalang$ python scripts/preprocessing/extract_and_split_opinions.py
```
