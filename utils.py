import ee
import pandas
import geemap
import config
import os

def convert_to_df(asset_id_path, export_csv=False, save_path="gee_df_to_csv.csv"):
  '''
  Convert given Google Earth Asset (path) to Pandas dataframe.\n
  Option to export saved df to csv.\n
  Example save path: save_path='gee_df_to_csv.csv'
  FeatureCollection -> DataFrame
  '''

  try:
    fc = ee.FeatureCollection(asset_id_path)
    _ = fc.size().getInfo() # this is unused but forces python to trigger and catch GEE exception
  except Exception as e:
    print(f"Feature collection for {asset_id_path} not yet ready.\nCheck status at https://code.earthengine.google.com/ under 'Tasks' ")
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


def embeddings_mean_time_range(state_fips="17", start_year="2017",end_year="2018", test=False):
  start_year = start_year.strip()
  end_year = end_year.strip()
  state_fips = state_fips.strip()
  
  embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{start_year}-01-01", f"{end_year}-01-01")
  embeddings_mean = embeddings.mean()

  state = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq('STATEFP', state_fips))
  
  if test == True:
    state = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq('STATEFP', state_fips)).limit(1)

  # embeddings uses a 10m x 10m resolution
  county_embeddings = embeddings_mean.reduceRegions(
    collection = state,
    reducer = ee.Reducer.mean(),
    scale = 10
  )

  # this will be used for creating the county column in the asset table
  band_names = embeddings_mean.bandNames()
  props_to_keep = ee.List(['GEOID']).cat(band_names)
  county_table = county_embeddings.select(props_to_keep)

  return county_table

# main function to start task (use this in data cleaning to get asseet as df for then merging everything.)
def get_mean_embeddings_task(state_fips="17",start_year="2017",end_year="2018", test=False, asset_id=""):
  '''
  ``state_fips`` -> should be a string of the state fips.\n
  ``start_year`` -> inclusive of the time range.\n
  ``end_year`` -> exclusive of the time range. \n
  ``test`` -> if true then this will run for one county only for the specified ``state_fips``.
  '''
  county_table = embeddings_mean_time_range(state_fips, start_year, end_year, test)

  export_task = ee.batch.Export.table.toAsset(
    collection = county_table,
    description = f'{state_fips}_{start_year}_{end_year}_AlphaEarth_Embeddings',
    assetId = asset_id
  )
  
  export_task.start()
  print(f"task export started for state_fips {state_fips}, date range ({start_year}, {end_year}).\n")

  return export_task