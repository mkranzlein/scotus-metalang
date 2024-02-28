# scotus-metalang
A diachronic analysis of legal metalanguage used by the U.S. Supreme Court from 1986 to 2019.

This is in-progress work that builds on the models developed in [mkranzlein/HiPool](https://github.com/mkranzlein/HiPool).

## Setup
1. Create conda/mamba environment.
    ```
    mamba env create -f environment.yml
    mamba activate scotus-metalang
    ```
2. Install scotus-metalang locally.
    ```
    pip install --upgrade build
    pip install -e .
    ```
3. Install HiPool locally.
    - TODO: add link and build command

### Caselaw Access Project (CAP) API Setup
1. Get an [API token](https://case.law/docs/site_features/api) for Harvard's [Caselaw Access Project](https://case.law/).
2. Add API token as environment variable.
    Create a file called `.env` in the root of your clone of this repo.
    In that file, define your CAP API token:
    ```
    CAP_TOKEN = {your_token_here}
    ```
    
### Download Super-SCOTUS data

Download `case_with_all_sources_with_companion_cases_tag.jsonl` [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/POWQIT) and place in [data/super_scotus](./data/super_scotus/).
