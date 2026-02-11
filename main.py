from pathlib import Path
import sys
PROJECT_ROOT = Path.cwd().parents[1]  # <-- wnv_embeddings
sys.path.insert(0, str(PROJECT_ROOT))
import ee
from utils.utils import get_mean_embeddings_task, get_mean_embeddings_task_ct_only
from config import *

# will prompt you to authorize access to GEE
ee.Authenticate()

# enter your own registered project name here
ee.Initialize(project="wnv-embeddings")


# ============= RUN GEE TASKS ============= #

run_gee_tasks = False # change this to true if needed but I am avoiding the stuff below for now 

if run_gee_tasks == True:
  # all state fips codes available at: https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt
  state_fips_codes = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56"
  ]

  for fips in state_fips_codes:
    # users/angel314/wnv_embeddings/{save_path_here}
    # an example save path: "users/angel314/wnv_embeddings/{fips}_2017_2024_embeddings"
    # description is as follows: "state15_2017_2024_AlphaEarth_Embeddings"
    save_path = f"{GEE_BASE_SAVE_PATH}{fips}_2017_2024_embeddings"

    task = get_mean_embeddings_task(fips, asset_id=save_path)
    print(f"State {fips}: {task.status()}")

  # began rest of tasks at 8:38 pm on 1/24/26
  # duration: 26 minutes for longest task, thus all the tasks finished in less than 26 minutes. 

# ============= RUN GEE TASK FOR CT - USING CUSTOM COUNTIES SHAPEFILE ASSET ============= #

# all state fips codes available at: https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt
connecticut = ["09"]

for fips in connecticut:
  # custom save path for connecticut only 
  save_path = f"{GEE_BASE_SAVE_PATH}{fips}_2017_2024_embeddings_ct_new"

  task = get_mean_embeddings_task_ct_only(fips, asset_id=save_path)
  print(f"State {fips}: {task.status()}")
