import ee
from utils import get_mean_embeddings_task
from config import *

# will prompt you to authorize access to GEE
ee.Authenticate()

# enter your own registered project name here
ee.Initialize(project="wnv-embeddings")

# all state fips codes available at: https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt

state_fips_codes_test = ["15","16","17"]

# note: i temporarily removed "15", "16", "17", from this list
# 15-17 were successful

state_fips_codes = [
  "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
  "13", "18", "19", "20", "21", "22", "23",
  "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
  "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
  "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56"
]

# ======== 10 m resolution function ========
for fips in state_fips_codes:
  # users/angel314/wnv_embeddings/{save_path_here}
  # an example save path: "users/angel314/wnv_embeddings/fips15_2017_2024_embeddings"
  # description is as follows: "state15_2017_2024_AlphaEarth_Embeddings"
  save_path = f"{GEE_BASE_SAVE_PATH}{fips}_2017_2024_embeddings"

  task = get_mean_embeddings_task(fips, asset_id=save_path)
  print(f"State {fips}: {task.status()}")

# began rest of tasks at 8:38 pm on 1/24/26
# once I have ran this I go back to data_cleaning to convert this to a csv