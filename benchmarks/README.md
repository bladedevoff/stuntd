# Encoder benchmarks

`encoders.py` trains one stuntd site per encoder on the same labelled rows and measures each head on
the official test split, so Laya and the stock sentence encoders can be compared on languages and
label counts the demos do not cover.

```
pip install "stuntd[train]" pyarrow
python benchmarks/encoders.py --data-dir DIR [--languages ru de es zh ja en] [--encoders laya intfloat/multilingual-e5-base ...]
```

`DIR` holds the downloads and the trained heads. Progress goes to stderr, the table to stdout.
`--languages` takes MASSIVE language codes; Banking77 always runs after them. A run on a laptop GPU
takes under two hours, much of it Laya on Banking77.

## What is measured

- **Data.** The `ru`, `de`, `es`, `zh` (`zh-CN`), `ja` and `en` parts of
  [`mteb/amazon_massive_intent`](https://huggingface.co/datasets/mteb/amazon_massive_intent) (60
  intents), read from its `refs/convert/parquet` revision because the main branch holds gzipped JSON,
  and [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) (77 intents). Training rows
  are a stratified sample of 3,000 from the train split (seed 0); every row of the official test
  split is scored.
- **Training.** Through `train_sites`, the code `stuntd train` runs, with the default holdout,
  `target_agreement` and novelty quantile, `training.cache_max_mb = 2048` and, for Laya,
  `training.epochs = 24` as in the demos. A stock encoder trains for its fixed 80 epochs. The
  captures are written once per dataset and shared by the four encoders.
- **Accuracy.** Share of test rows the head answers correctly, whatever its confidence.
- **Coverage, Accuracy at coverage.** Share of test rows whose confidence reaches the head's own
  threshold, and accuracy on them. A dash means the head found no threshold that reaches the 0.99
  target on its holdout, so it would answer nothing locally.
- **Local.** Share of test rows the daemon would answer from the head: confident enough and not
  stopped by the novelty gate at its default.
- **Train s.** Wall-clock seconds of `train_sites` for the site, the encoder already loaded.
- **p50 ms, p95 ms.** Milliseconds for one batch-1 decision on the CPU (`device = "cpu"`), over the
  first 200 test texts after a warm-up pass.

## Results

Run on 2026-10-10 on the commit after `3b1e6a5`, on an Intel Core Ultra 9 275HX
with 31 GB of RAM and an NVIDIA GeForce RTX 5060 Laptop GPU (8 GB, CUDA 13.0) under Windows 11,
Python 3.13.15, torch 2.14.0+cu130, transformers 5.17.0, laya 0.3.4. Training and the accuracy pass
ran on the GPU, the latency pass on the CPU. Dataset revisions: `mteb/amazon_massive_intent`
`refs/convert/parquet` at `dfe509a4cff77fcdead03f9275b6c0c346d049cb`, `mteb/banking77` at
`18072d2685ea682290f7b8924d94c62acc19c0b2`. Model revisions: `convaiinnovations/laya`
`7b928d828b7b0e022f929d9bd2e44165aa270148`, `intfloat/multilingual-e5-base`
`d128750597153bb5987e10b1c3493a34e5a4502a`, `intfloat/multilingual-e5-small`
`614241f622f53c4eeff9890bdc4f31cfecc418b3`, `sentence-transformers/all-MiniLM-L6-v2`
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

```
| Dataset | Encoder | Accuracy | Coverage | Accuracy at coverage | Local | Train s | p50 ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| massive-ru | laya | 26.2% | - | - | 0.0% | 306.0 | 353.3 | 392.4 |
| massive-ru | intfloat/multilingual-e5-base | 81.6% | 33.4% | 98.0% | 33.2% | 7.1 | 14.4 | 17.9 |
| massive-ru | intfloat/multilingual-e5-small | 78.6% | 23.0% | 98.5% | 22.7% | 3.5 | 8.1 | 13.5 |
| massive-ru | sentence-transformers/all-MiniLM-L6-v2 | 54.7% | 6.9% | 96.6% | 6.9% | 4.2 | 5.6 | 8.3 |
| massive-de | laya | 31.5% | 3.3% | 80.4% | 3.1% | 296.3 | 344.7 | 366.7 |
| massive-de | intfloat/multilingual-e5-base | 79.6% | 32.9% | 97.9% | 32.6% | 7.3 | 14.3 | 19.0 |
| massive-de | intfloat/multilingual-e5-small | 75.9% | 24.2% | 98.9% | 24.0% | 3.5 | 8.2 | 11.8 |
| massive-de | sentence-transformers/all-MiniLM-L6-v2 | 65.1% | 6.3% | 98.4% | 6.3% | 3.3 | 4.9 | 7.4 |
| massive-es | laya | 34.3% | 3.0% | 85.4% | 2.9% | 295.3 | 347.2 | 363.2 |
| massive-es | intfloat/multilingual-e5-base | 80.5% | 21.3% | 98.6% | 21.2% | 7.1 | 14.1 | 18.8 |
| massive-es | intfloat/multilingual-e5-small | 78.0% | 35.9% | 97.6% | 35.6% | 3.5 | 7.9 | 13.1 |
| massive-es | sentence-transformers/all-MiniLM-L6-v2 | 63.5% | 2.2% | 100.0% | 2.2% | 3.3 | 4.6 | 7.9 |
| massive-zh | laya | 44.1% | 0.1% | 50.0% | 0.1% | 300.1 | 372.2 | 463.2 |
| massive-zh | intfloat/multilingual-e5-base | 79.8% | 23.2% | 98.3% | 23.2% | 7.0 | 16.2 | 20.1 |
| massive-zh | intfloat/multilingual-e5-small | 78.3% | 3.0% | 100.0% | 3.0% | 3.6 | 9.1 | 14.1 |
| massive-zh | sentence-transformers/all-MiniLM-L6-v2 | 40.2% | 2.3% | 92.5% | 2.2% | 3.5 | 4.8 | 8.5 |
| massive-ja | laya | 32.1% | 2.7% | 91.1% | 2.6% | 299.3 | 385.6 | 417.5 |
| massive-ja | intfloat/multilingual-e5-base | 79.8% | 18.1% | 99.3% | 18.0% | 7.7 | 16.1 | 20.1 |
| massive-ja | intfloat/multilingual-e5-small | 79.5% | 27.1% | 98.5% | 27.0% | 3.9 | 9.5 | 12.7 |
| massive-ja | sentence-transformers/all-MiniLM-L6-v2 | 57.7% | 3.6% | 100.0% | 3.6% | 3.9 | 6.0 | 7.5 |
| massive-en | laya | 60.7% | 3.3% | 93.9% | 3.2% | 279.5 | 337.4 | 359.5 |
| massive-en | intfloat/multilingual-e5-base | 83.7% | 42.0% | 98.5% | 41.9% | 6.8 | 14.0 | 18.0 |
| massive-en | intfloat/multilingual-e5-small | 81.2% | 27.7% | 98.2% | 27.7% | 3.5 | 7.8 | 12.5 |
| massive-en | sentence-transformers/all-MiniLM-L6-v2 | 81.0% | 23.9% | 98.9% | 23.9% | 3.2 | 4.8 | 8.6 |
| banking77 | laya | 66.1% | 5.1% | 99.4% | 4.9% | 1768.8 | 544.2 | 612.3 |
| banking77 | intfloat/multilingual-e5-base | 89.6% | 60.8% | 98.8% | 60.8% | 11.1 | 18.4 | 24.5 |
| banking77 | intfloat/multilingual-e5-small | 87.5% | 58.5% | 99.0% | 58.3% | 5.7 | 11.0 | 15.0 |
| banking77 | sentence-transformers/all-MiniLM-L6-v2 | 90.0% | 58.2% | 98.8% | 58.2% | 4.9 | 6.1 | 7.9 |
```

A second full run from a clean data directory, same code and machine, printed this:

```
| Dataset | Encoder | Accuracy | Coverage | Accuracy at coverage | Local | Train s | p50 ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| massive-ru | laya | 27.2% | 1.5% | 91.3% | 1.5% | 308.7 | 353.6 | 376.4 |
| massive-ru | intfloat/multilingual-e5-base | 81.6% | 33.4% | 98.0% | 33.2% | 6.9 | 14.3 | 18.9 |
| massive-ru | intfloat/multilingual-e5-small | 78.6% | 23.0% | 98.5% | 22.7% | 3.4 | 8.1 | 15.1 |
| massive-ru | sentence-transformers/all-MiniLM-L6-v2 | 54.7% | 6.9% | 96.6% | 6.9% | 4.2 | 5.3 | 8.6 |
| massive-de | laya | 29.5% | 1.4% | 81.4% | 1.3% | 295.2 | 401.7 | 430.2 |
| massive-de | intfloat/multilingual-e5-base | 79.6% | 32.9% | 97.9% | 32.6% | 8.7 | 20.1 | 25.0 |
| massive-de | intfloat/multilingual-e5-small | 75.9% | 24.2% | 98.9% | 24.0% | 4.3 | 10.1 | 14.1 |
| massive-de | sentence-transformers/all-MiniLM-L6-v2 | 65.1% | 6.3% | 98.4% | 6.3% | 3.9 | 5.8 | 7.4 |
| massive-es | laya | 34.4% | 1.7% | 90.4% | 1.7% | 296.4 | 343.0 | 364.2 |
| massive-es | intfloat/multilingual-e5-base | 80.5% | 21.3% | 98.6% | 21.2% | 7.1 | 14.1 | 18.9 |
| massive-es | intfloat/multilingual-e5-small | 78.0% | 35.9% | 97.6% | 35.6% | 3.6 | 7.9 | 13.8 |
| massive-es | sentence-transformers/all-MiniLM-L6-v2 | 63.5% | 2.2% | 100.0% | 2.2% | 3.3 | 4.5 | 7.9 |
| massive-zh | laya | 42.7% | 0.4% | 81.8% | 0.4% | 299.1 | 344.4 | 370.0 |
| massive-zh | intfloat/multilingual-e5-base | 79.8% | 23.2% | 98.3% | 23.2% | 6.5 | 14.5 | 18.1 |
| massive-zh | intfloat/multilingual-e5-small | 78.3% | 3.0% | 100.0% | 3.0% | 3.3 | 8.5 | 14.1 |
| massive-zh | sentence-transformers/all-MiniLM-L6-v2 | 40.2% | 2.3% | 92.5% | 2.2% | 3.3 | 4.9 | 7.8 |
| massive-ja | laya | 32.2% | 2.1% | 96.8% | 2.1% | 299.0 | 356.5 | 411.8 |
| massive-ja | intfloat/multilingual-e5-base | 79.8% | 18.1% | 99.3% | 18.0% | 7.8 | 16.5 | 21.0 |
| massive-ja | intfloat/multilingual-e5-small | 79.5% | 27.1% | 98.5% | 27.0% | 3.8 | 9.7 | 12.5 |
| massive-ja | sentence-transformers/all-MiniLM-L6-v2 | 57.7% | 3.6% | 100.0% | 3.6% | 3.8 | 5.9 | 8.1 |
| massive-en | laya | 60.1% | 6.2% | 93.0% | 5.9% | 236.4 | 390.9 | 499.4 |
| massive-en | intfloat/multilingual-e5-base | 83.7% | 42.0% | 98.5% | 41.9% | 7.6 | 26.6 | 29.4 |
| massive-en | intfloat/multilingual-e5-small | 81.2% | 27.7% | 98.2% | 27.7% | 4.1 | 12.6 | 14.4 |
| massive-en | sentence-transformers/all-MiniLM-L6-v2 | 81.0% | 23.9% | 98.9% | 23.9% | 3.5 | 6.6 | 7.9 |
| banking77 | laya | 65.1% | 6.5% | 98.0% | 6.5% | 1860.3 | 559.0 | 602.4 |
| banking77 | intfloat/multilingual-e5-base | 89.6% | 60.8% | 98.8% | 60.8% | 11.7 | 25.0 | 30.7 |
| banking77 | intfloat/multilingual-e5-small | 87.5% | 58.5% | 99.0% | 58.3% | 5.6 | 12.1 | 14.0 |
| banking77 | sentence-transformers/all-MiniLM-L6-v2 | 90.0% | 58.2% | 98.8% | 58.2% | 4.7 | 6.9 | 8.0 |
```

## Reading the numbers

- Banking77 needs 2.1 GiB of encoder cache for Laya, over the 2,048 MB budget, so that site trained
  uncached, re-encoding every epoch; this is why its Laya training time is six times the MASSIVE ones.
- A GPU run is not bit-for-bit repeatable for Laya. Between the two runs above its accuracy moved
  by up to 2.0 points (`massive-de` 31.5% and 29.5%), `banking77` gave 66.1% and 65.1%, and the
  coverage and local share moved more: `massive-ru` went from nothing at the 0.99 target to 1.5%
  local, `massive-zh` from 0.1% to 0.4% coverage, `massive-en` from 3.2% to 5.9% local. The accuracy,
  coverage, accuracy at coverage and local columns of the three stock encoders matched to the digit
  in both runs. Compare Laya only across gaps larger than these.
- The latency and training-time columns did not reproduce within 25% on every row. The second run's
  `massive-en` p50 for multilingual-e5-base was 26.6 ms against 14.0, its p95 for Laya 499.4 ms
  against 359.5, and its Laya training 236.4 s against 279.5. The later datasets of the second run
  were slower across encoders, which points at the state of the machine, but that was not shown.
  Treat these columns as an order of magnitude, not as figures to match.
- Latency is one process on a laptop; the three-label demos of the root README
  measure the Laya head at 60 to 62 ms on the same CPU.
- Nothing here was run against a provider: the labels are the datasets' own, so the numbers say how
  well a head learns them, not how well it would copy a particular teacher.
