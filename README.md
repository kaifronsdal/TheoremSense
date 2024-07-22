# TheoremSense
An Empirical Study of Proxy Metrics for Verifying Math Proofs


## Running generations

The structure of this repo is composed of two parts:
1. Using models to generate data in autoregressive and teacher forcing fashions
2. performing analysis on this generated data.

### Generating data
To generate data take a look at `theoremsense/all_generate.sh` which shows how to call `model_generate.py` for a wide range of models.