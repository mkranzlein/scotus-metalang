# scotus-metalang
A diachronic analysis of the legal metalanguage used by the U.S. Supreme Court


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

### Caselaw Access Project (CAP) API Setup
1. Get an [API token](https://case.law/docs/site_features/api) for Harvard's [Caselaw Access Project](https://case.law/).
2. Add API token as environment variable.
    
    Create directories:
    ```
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
    mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d/
    ```

    Create file `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`:
    ```
    #!/bin/sh

    export CAP_TOKEN="{TOKEN HERE}"
    ```
    Create file `$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh`:
    ```
    #!/bin/sh

    unset CAP_TOKEN
    ```
    `CAP_TOKEN` can now be accessed when the `scotus-metalang` environment is active:
    ```
    import os
    os.environ["CAP_TOKEN"]
    ```
    