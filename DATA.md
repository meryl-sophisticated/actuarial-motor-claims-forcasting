Place `freMTPL2freq.rda` here.

The dataset is part of the R package [CASdatasets](https://dutangc.github.io/CASdatasets/) (Dutang & Charpentier, GPL ≥ 2). The simplest way to get the file is to download it from the package's GitHub repository:

```bash
curl -L -o freMTPL2freq.rda \
  https://github.com/dutangc/CASdatasets/raw/c49cbbb37235fc49616cac8ccac32e1491cdc619/data/freMTPL2freq.rda
```

Or install the package in R and save the data frame:

```r
install.packages("CASdatasets", repos = "http://cas.uqam.ca/pub/R/", type = "source")
library(CASdatasets)
data(freMTPL2freq)
save(freMTPL2freq, file = "freMTPL2freq.rda")
```

The file is not committed to this repository because of its size (3.4 MB).
