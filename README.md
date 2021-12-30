# Finance and Machine Learning

## Market simulation for Deep Hedging

Repository for our course project where we compare simulation techniques
within the deep hedging framework.

The main results can be seen in the following jupyter notebooks:
- gjr-model.ipynb
  -  here we train a deep hedging network on GJR-GARCH paths and evaluate it on simulated and real paths

-  res-model.ipynb
   - here we train a deep hedging network on RC paths and evaluate it on simulated and real paths

The notebook compare-sim-paths.ipynb includes the illustrations of the simulated paths shown in the report.

The notebook reservoir-simulation.ipynb generates the RC paths.

The GJRGARCH_ESTIMATION.R file generates the GJR-GARCH paths.

The generated data can be found here: https://www.kaggle.com/konmue/gjr-vol
