import os
from pyproj import datadir

# Check if the PROJ_LIB environment variable is set
proj_lib_env = os.environ.get('PROJ_LIB')
print("PROJ_LIB environment variable:", proj_lib_env)

# Get the data directory used by pyproj
current_proj_dir = datadir.get_data_dir()
print("pyproj data directory:", current_proj_dir)