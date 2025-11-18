import ee
from utils import get_mean_embeddings_task

# The purpose of main is to export the GEE Embeddings task because it did not run properly in .ipynb.

# connecting to GEE for section 2 in data_cleaning.ipynb
ee.Authenticate()
# new project I registered for this research.
ee.Initialize(project="wnv-embeddings")

#running the full 102 county calculation
task = get_mean_embeddings_task(
  state_fips="17",
  start_year="2017",
  end_year="2018",
  test=False
)
print(task.status())

# once I have ran this I go back to data_cleaning to convert this to a csv