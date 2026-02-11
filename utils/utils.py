import ee
import pandas as pd
import geemap
import os
from typing import Optional

def convert_to_df(asset_id_path, export_csv=False, save_path="gee_df_to_csv.csv") -> Optional[pd.DataFrame]:
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

  if export_csv:
    if os.path.exists(save_path):
        print(f"'{save_path}' already exists. Will overwrite.")
    else:
        print(f"{save_path} does not exist, creating {save_path}\n")
        print(f"retrieved asset at '{asset_id_path}'\nsaved as CSV to: {save_path}")
    df.to_csv(save_path, index=False)

  return df

def embeddings_mean_all_years(state_fips: str, start_year=2017, end_year=2024, test=False):
  '''
  Calculate mean embeddings for each year separately, then combine results.
  '''
  state_fips = state_fips.strip()

  # future fix: this dataset needs to be updated but GEE does not have newer TIGER census boundaries
  # 2022 and later - Connecticut uses Planning Regions instead of Counties. 

  # limit to one county if testing, otherwise use all counties for the specified state fips code. 
  if test == True:
    state = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq('STATEFP', state_fips)).limit(1)
  else:
    state = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq('STATEFP', state_fips))

  # Resulting table only has the geoid of each county (unique identifier)
  # we will add data to this table
  county_table = state.select(['GEOID'])

  # process one year at once then add to table.
  # avoids the memory limit by not creating a massive multi-band image.
  for year in range(start_year, end_year + 1):
    embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(
      f"{year}-01-01", 
      f"{year + 1}-01-01" 
      )
    
    embeddings_mean = embeddings.mean()
    
    # this renames columns as: {band}_{year}
    # an example being A00_2017
    band_names = embeddings_mean.bandNames()
    new_band_names = band_names.map(lambda name: ee.String(name).cat(f'_{year}'))
    embeddings_mean = embeddings_mean.rename(new_band_names)
    
    # Reduce regions for this year only (64 bands processed)
    # old function involved processing 64 * 8 = 512 bands at once 
    year_embeddings = embeddings_mean.reduceRegions(
      collection=state, # compute statistics for all counties in state collection
      reducer=ee.Reducer.mean(), # mean of all pixels within each county
      scale=10 # 10m resolution native resolution
    )
    
    # Select only the embedding bands and GEOID for joining (resulting table keeps this)
    props_to_keep = ee.List(['GEOID']).cat(new_band_names)
    year_embeddings = year_embeddings.select(props_to_keep)
    
    # Join this year's data to the main table
    # 1. inner join to keep matches only
    # 2. Filter.equals to match rows where GEOID is the same
    # 3. .map -> flatten features by copying the current year properties to county_table
    county_table = ee.Join.inner().apply(
      county_table,
      year_embeddings,
      ee.Filter.equals(leftField='GEOID', rightField='GEOID')
    ).map(lambda feature: ee.Feature(
      ee.Feature(feature.get('primary')).copyProperties(
        ee.Feature(feature.get('secondary'))
      )
    ))
  # return table where: each county has GEOID + 64 bands × 8 years = 512 embedding columns
  return county_table

# main function to start task (use this in data cleaning to get asseet as df for then merging everything.)
def get_mean_embeddings_task(state_fips: str, start_year=2017,end_year=2024, test=False, asset_id=""):
  '''
  `state_fips` -> string of the state FIPS code.\n
  `start_year` -> inclusive of the time range.\n
  `end_year` -> inclusive of the time range.\n
  assign "test=True" to only run for one county for testing purposes

  Initiates main task to get county table with mean embedding data for the state. 
  '''
  # note: '2024-01-01' is the most recent availability
  # https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
  
  if not asset_id:
    raise ValueError("asset_id must be a non-empty EE asset path, e.g. 'users/you/folder/name'")

  county_table = embeddings_mean_all_years(state_fips, start_year, end_year, test)

  export_task = ee.batch.Export.table.toAsset(
    collection = county_table,
    description = f'state{state_fips}_{start_year}_{end_year}_AlphaEarth_Embeddings',
    assetId = asset_id
  )
  
  export_task.start()
  print(f"task export started for state_fips {state_fips}, date range ({start_year}, {end_year}).\n")

  return export_task


#### NEW FUNCTIONS FOR CONNECTICUT ONLY ####

def embeddings_mean_all_years_ct_only(state_fips="09", start_year=2017, end_year=2024):
  '''
  Calculate mean embeddings for each year separately, then combine results.
  '''
  state_fips = state_fips.strip()

  # using uploaded asset (TIGER/Line 2025 county boundaries)
  planning_regions = ee.FeatureCollection("projects/wnv-embeddings/assets/tl_2025_us_county").filter(ee.Filter.eq('STATEFP', state_fips))

  county_table = planning_regions.select(['GEOID']) # in reality these are planning region codes

  # process one year at once then add to table.
  # avoids the memory limit by not creating a massive multi-band image.
  for year in range(start_year, end_year + 1):
    embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(
      f"{year}-01-01", 
      f"{year + 1}-01-01" 
      )
    
    embeddings_mean = embeddings.mean()
    
    # this renames columns as: {band}_{year}
    # an example being A00_2017
    band_names = embeddings_mean.bandNames()
    new_band_names = band_names.map(lambda name: ee.String(name).cat(f'_{year}'))
    embeddings_mean = embeddings_mean.rename(new_band_names)
    
    # Reduce regions for this year only (64 bands processed)
    # old function involved processing 64 * 8 = 512 bands at once 
    year_embeddings = embeddings_mean.reduceRegions(
      collection=planning_regions, # compute statistics for all planning regions in connecticut
      reducer=ee.Reducer.mean(), # mean of all pixels within each county
      scale=10 # 10m resolution native resolution
    )
    
    # Select only the embedding bands and GEOID for joining (resulting table keeps this)
    props_to_keep = ee.List(['GEOID']).cat(new_band_names)
    year_embeddings = year_embeddings.select(props_to_keep)
    
    # Join this year's data to the main table
    # 1. inner join to keep matches only
    # 2. Filter.equals to match rows where GEOID is the same
    # 3. .map -> flatten features by copying the current year properties to county_table
    county_table = ee.Join.inner().apply(
      county_table,
      year_embeddings,
      ee.Filter.equals(leftField='GEOID', rightField='GEOID')
    ).map(lambda feature: ee.Feature(
      ee.Feature(feature.get('primary')).copyProperties(
        ee.Feature(feature.get('secondary'))
      )
    ))
  # return table where: each county has GEOID + 64 bands × 8 years = 512 embedding columns
  return county_table

def get_mean_embeddings_task_ct_only(state_fips: str, start_year=2017,end_year=2024, test=False, asset_id=""):
  '''
  `state_fips` -> string of the state FIPS code.\n
  `start_year` -> inclusive of the time range.\n
  `end_year` -> inclusive of the time range.\n
  assign "test=True" to only run for one county for testing purposes

  Initiates main task to get county table with mean embedding data for the state. 
  '''
  # note: '2024-01-01' is the most recent availability
  # https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
  
  if not asset_id:
    raise ValueError("asset_id must be a non-empty EE asset path, e.g. 'users/you/folder/name'")

  county_table = embeddings_mean_all_years_ct_only(state_fips, start_year, end_year)

  export_task = ee.batch.Export.table.toAsset(
    collection = county_table,
    description = f'{state_fips}_{start_year}_{end_year}_AlphaEarth_Embeddings_ct_new_only',
    assetId = asset_id
  )
  
  export_task.start()
  print(f"task export started for state_fips {state_fips}, date range ({start_year}, {end_year}).\n")

  return export_task