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
