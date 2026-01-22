import ee
import pandas as pd
import geemap
import config
import os

def convert_to_df(asset_id_path, export_csv=False, save_path="gee_df_to_csv.csv") -> pd.DataFrame:
  '''
  Convert given Google Earth Asset (path) to Pandas dataframe.\n
  Option to export saved df to csv: `export_csv`.\n
  Example save path: save_path='gee_df_to_csv.csv'\n
  FeatureCollection -> DataFrame
  '''
  try:
    fc = ee.FeatureCollection(asset_id_path)
    _ = fc.size().getInfo() # this is unused but forces python to trigger and catch GEE exception
  except Exception as e:
    print(f"Feature collection for {asset_id_path} not yet ready.\nCheck status at https://code.earthengine.google.com/tasks")
    print(e)
    return None
  
  df = geemap.ee_to_df(fc)

  if (export_csv):
    df = convert_to_df(asset_id_path)
    if os.path.exists(save_path):
        print(f"'{save_path}' already exists. Will overwrite.")
    else:
        print(f"{save_path} does not exist, creating {save_path}\n")
        print(f"retreived asset at '{asset_id_path}'\nsaved as CSV to: {save_path}")
    df.to_csv(save_path)

  return df


def embeddings_mean_all_years(state_fips: str, start_year=2017,end_year=2024, test=False):
  '''
  Calculate mean embeddings across the entire time range for all counties in a state.\n
  Note: end_year is now INCLUSIVE (2025 means through end of 2024)
  '''
  state_fips = state_fips.strip()

  if test == True:
    state = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq('STATEFP', state_fips)).limit(1)
  else:
    state = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq('STATEFP', state_fips))

  # collect all ee.Image objects for all years for the state
  yearly_images = []

  for year in range(start_year, end_year+1):
    embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{start_year}-01-01", f"{start_year+1}-01-01")
    # only calculating mean for current year to current year + 1 (ex. 2017-01-01 to 2018-01-01) for 2017
    embeddings_mean = embeddings.mean()

    # Rename bands to include year suffix (e.g., embedding_0_2017)
    band_names = embeddings_mean.bandNames()
    new_band_names = band_names.map(lambda name: ee.String(name).cat(f'_{year}'))
    embeddings_mean = embeddings_mean.rename(new_band_names)

    yearly_images.append(embeddings_mean)

  # concatenate all yearly images into one multi-band image
  combined_image = ee.Image.cat(yearly_images)

  # now, using the multi-band image, 
  # calculate the mean of pixels in the specified state (according to FIPS code input)
  county_embeddings = combined_image.reduceRegions(
    collection = state,
    reducer = ee.Reducer.mean(),
    scale = 10
  )

  # keep GEOID and all of the bands in the multi-band image
  band_names = combined_image.bandNames()
  props_to_keep = ee.List(['GEOID']).cat(band_names)
  county_table = county_embeddings.select(props_to_keep)

  return county_table

# main function to start task (use this in data cleaning to get asseet as df for then merging everything.)
def get_mean_embeddings_task(state_fips: str, start_year=2017,end_year=2025, test=False, asset_id=""):
  '''
  `state_fips` -> string of the state FIPS code.\n
  `start_year` -> inclusive of the time range.\n
  `end_year` -> inclusive of the time range.\n
  assign "test=True" to only run for one county for testing purposes

  Initiates main task to get county table with mean embedding data for the state. 
  '''
  # note: '2024-01-01' is the most recent availability
  # https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
  
  county_table = embeddings_mean_all_years(state_fips, start_year, end_year, test)

  export_task = ee.batch.Export.table.toAsset(
    collection = county_table,
    description = f'{state_fips}_{start_year}_{end_year}_AlphaEarth_Embeddings',
    assetId = asset_id
  )
  
  export_task.start()
  print(f"task export started for state_fips {state_fips}, date range ({start_year}, {end_year}).\n")

  return export_task