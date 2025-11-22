import ee
from utils import get_mean_embeddings_task
# The purpose of main is to export the GEE Embeddings task because it did not run properly in .ipynb.

# connecting to GEE for section 2 in data_cleaning.ipynb
ee.Authenticate()
# new project I registered for this research.
ee.Initialize(project="wnv-embeddings")

######## 2017 ########  
task = get_mean_embeddings_task(
  state_fips="17",
  start_year="2017",
  end_year="2018",
  test=False,
  asset_id='users/angel314/Analysis_AlphaEarth_Embeddings_2017'
)
print(task.status())

######## 2018 ########  
task = get_mean_embeddings_task(
  state_fips="17",
  start_year="2018",
  end_year="2019",
  test=False,
  asset_id='users/angel314/Analysis_AlphaEarth_Embeddings_2018'
)
print(task.status())

######## 2019 ########  
task = get_mean_embeddings_task(
  state_fips="17",
  start_year="2019",
  end_year="2020",
  test=False,
  asset_id='users/angel314/Analysis_AlphaEarth_Embeddings_2019'
)
print(task.status())

######## 2020 ########  
task = get_mean_embeddings_task(
  state_fips="17",
  start_year="2020",
  end_year="2021",
  test=False,
  asset_id='users/angel314/Analysis_AlphaEarth_Embeddings_2020'
)
print(task.status())

######## 2021 ########  
task = get_mean_embeddings_task(
  state_fips="17",
  start_year="2021",
  end_year="2022",
  test=False,
  asset_id='users/angel314/Analysis_AlphaEarth_Embeddings_2021'
)
print(task.status())

######## 2022 ########  
task = get_mean_embeddings_task(
  state_fips="17",
  start_year="2022",
  end_year="2023",
  test=False,
  asset_id='users/angel314/Analysis_AlphaEarth_Embeddings_2022'
)
print(task.status())

######## 2023 ########  
task = get_mean_embeddings_task(
  state_fips="17",
  start_year="2023",
  end_year="2024",
  test=False,
  asset_id='users/angel314/Analysis_AlphaEarth_Embeddings_2023'
)
print(task.status())

######## 2024 ########  
task = get_mean_embeddings_task(
  state_fips="17",
  start_year="2024",
  end_year="2025",
  test=False,
  asset_id='users/angel314/Analysis_AlphaEarth_Embeddings_2024'
)
print(task.status())

# once I have ran this I go back to data_cleaning to convert this to a csv