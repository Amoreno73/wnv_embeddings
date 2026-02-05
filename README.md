# wnv_embeddings
Predicting WNV prevalence at a county level using <a href="https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/">Google AlphaEarth Foundations’ annual embeddings</a> data.

Predictor Variables: 
* 64-dimensional AlphaEarth embedding fields. The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.  

Target Variables:
* Normalized Human Cases for each county in a given state.

This project initiates Google Earth Engine "tasks," which are run on Google's cloud infrastructure. 

`utils.py` contains helper functions and scripts to start tasks to get the mean embedding value for each county.  
`main.py` is responsible for actually calling the script functions to begin pulling the data server side. 

`notebooks/national_embeddings/wnv_analysis.ipynb` is the notebook file with the national WNV analysis process.
This notebook also pulls in the assets that were started in Google Earth Engine by `main.py`.

Note: a Google Earth Engine (GEE) account is needed to initiate tasks.