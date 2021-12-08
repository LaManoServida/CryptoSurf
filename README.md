# \[WIP\] CryptoSurf

Automatically trade cryptocurrencies using Deep Learning.

This project uses [python-binance](https://github.com/sammchardy/python-binance) module, an unofficial python wrapper
for the Binance exchange REST API v3.

## Installation

1. Clone repository
2. Create and activate a Python 3.9 virtual environment (preferably the latest Python version supported
   by [python-binance](https://github.com/sammchardy/python-binance))
3. Install requirements
4. Create `config-local.ini` as a copy of `config.ini` and fill in your api key and secret

## Getting started

1. Create a dataset with `create_dataset.py`. For example:

```bash
python3 create_dataset.py BTCUSDT -i 1h -n 200  # use '--help' for more details
```
