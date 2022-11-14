# ML Water Level

## Question
Can we predict the depth of a river 10-20 days in the future given historical data.

## Project Layout
- *input_data:* Data downloaded by hand to direct automatic data ingestion step
- *ingested_data:* Data downloaded by data_ingest.py and process_ghcnd.py
- data_process.py: investigating how the ingested data looks

## Packages
`conda create --name mlwaterlevel ipykernel tensorflow pandas matplotlib seaborn plotly statsmodels tqdm umap-learn scikit-learn jupyterlab ipywidgets ipympl --channel conda-forge`

## Notes
Depth correlation under 0.5 on day 11
Precipitation correlation under 0.5 on day 1
Temperature correlation under 0.5 on day 49