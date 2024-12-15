 #!/bin/bash
for i in {1..12}
do
	python generate_dataset.py 
	-n /path/to/data/ml_ecrad_ape_R2B05_myrunscript_1year_183min/year${i}/ml_ecrad_ape_R2B05_myrunscript_1year_183min_atm_3d_DOM01_ml_0001_lonlat.nc 
	-s /path/to/save/year${i}/
done