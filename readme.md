<img width='100%' src='presentation docs/dalle_banner_river.png'>

# ML Water Level - Predicting future river depth with ML
![GitHub release (latest by date including pre-releases)](https://img.shields.io/badge/release-v1.0-blue)
![GitHub issues](https://img.shields.io/badge/open%20issues-0-success)

## Question
Can we predict the depth of a river 14 days in the future given 49 days of historical depth, precipitation and temperature data.

## Data
This project retrieved 10 years of data on river gage (depth) from the United States Geological Survey’s (USGS) National Water Information System (NWIS) and the Global Historical Climatological Network (GHCN) for precipitation and temperature data. The precipation and temperature data closest to a river recording site was used as the true precipitation at that site. This approximation was valid due to the small distance between river recording sites and precipitation and temperature sites shown in the left figure below. The final dataset used a test-train split of 67-33% with an additional test set of rivers without any missing depth data. The location of the river sites is shown in the right figure below. 

<img width='49%' src='presentation docs/nwis_to_ghcn_distance.png' hspace="1%"><img width='49%' src='presentation docs/train_test_validate_nwis.png'>

The data was then converted into a set of features and labels using a sliding window of length 63 days, allowing for 49 days of historical data and a label 14 days in the future. This is shown in the figure below 

<img width='49%' src='presentation docs/sliding_window.png'>

## Results
Mean absolute error in depth prediction (ft) on various datasets to 3dp (lower is better). Best performer in all categories was the LSTM model.
|                 |     D48      |     Lasso    |     1 neuron    |     Multi-layer perceptron    |     LSTM     |     LSTM (small dataset)    |     CNN      |     CNN (small dataset)    |
|-----------------|--------------|--------------|-----------------|-------------------------------|--------------|-----------------------------|--------------|----------------------------|
|     Training    |     0.932    |     1.880    |     0.915       |     0.873                     |     0.829    |     1.949                   |     0.867    |     0.863                  |
|     Validate    |     0.892    |     2.195    |     0.878       |     0.810                     |     0.799    |     5.626                   |     0.831    |     0.834                  |
|     Test        |     0.774    |     1.492    |     0.771       |     0.712                     |     0.705    |     0.740                   |     0.741    |     0.733                  |

A histogram of the errors in the top performing models is shown in the picture on the left, whilst the predictions of the LSTM model on the test set are shown on the right. 

<img width='49%' src='presentation docs/error_best_models.png' hspace="1%"><img width='49%' src='presentation docs/LSTM model_test_pred.png'>


## Project files
- `./input_data`: Data downloaded by hand to direct automatic data ingestion step
- `./ingested_data`: Data downloaded by `nwis_ingest.py` and `ghcn_ingest.py`
- `./eda_results`: Lists of rivers with few NAs produced by `depth_eda.ipynb`, `precip_eda.py` and `temp_eda.py`
- `./singleneuron`, `./multineuron`, `./lstm`, `./lstm_small`, `./conv` and `./conv_small`: best Keras model of that type was exported for consistent analysis
- `./presentation_docs`: pictures for report and readme
- `nwis_ingest.py`: Collect river depth data from all rivers in `input_data/recordingsites.tsv` that contains all NWIS river sites
- `ghcn_ingest.py`: Collect precipitation and temperature data from GHCN sites closest to the river sites
- `river_depth_eda.ipynb`: Analyse `./ingested_data` for river sites with sufficient depth data for imputation to be acceptable
- `precip_eda.py`: Analyse `./ingested_data` for river sites with sufficient precipitation data for imputation to be acceptable
- `temp_eda.py`: Analyse `./ingested_data` for river sites with sufficient temperature data for imputation to be acceptable
- `collective_eda.py`: aggregating which rivers have sufficient depth, precipitation and temperature data and creating as output `eda_results/perfect_final_rivs.txt` `eda_results/imperfect_final_rivs.txt`. These are plotted on a graph of the US for interpreting which rivers have been selected
- `constructing_sliding_dataset_eda.ipynb`: EDA analysis of the final (cleaned) depth, precipitation and temperature datasets to assess how many days in the past we should use for the historical values as well as if any cyclicality exists in the variables. An  ARIMA model is also shown in this file but cannot be trained on the whole dataset so is ignored from future analysis
- `models.ipynb`: All the models are trained and analysed in this file. Pictures for reporting are saved to `./presentation_docs` 
- `useful_funcs.py`: contains common functions used across this project

## Packages
`conda create --name mlwaterlevel ipykernel tensorflow pandas matplotlib seaborn plotly statsmodels tqdm umap-learn scikit-learn jupyterlab ipywidgets ipympl --channel conda-forge`

## Acknowledgements
The banner was created using DALL-E 2 by Open AI. The badges were made using [Shields IO](https://shields.io/) and are inspired by the following [readme](https://github.com/navendu-pottekkat/awesome-readme).
Many thanks to Jon Hanke and Daniel Melesse for their ideas and support whilst I was working on this project.