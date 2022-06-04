### README

This repository contains jupyter notebooks and code developed to classify 2D high-content screening images based on texture and other features into 3 classes. 

### Data
- Coacervates data from 96-well plate, where each well has different concentrations of 2 proteins of interest
- The different concentrations lead to three categories of images: `droplets`, `aggregates`, `empty`.
	- Empty images look just like background 
	- Droplet images show transparent droplets (just circles) of various sizes
	- Aggregates - textured image, clumps of protein, no clear structure

### Analysis goal

Classify images automatically into one of the 3 categories. 

### Final workflow 
- Based on python using libraries `scikit-learn`, `mahotas`, `scikit-image`, `tifffile`.
- Workflow
	- `functions.py`: contains all the helper functions 
	- `1_read_images.ipynb`: read and save images for training - on laptop
	- `2_train_save_classifier.ipynb`: train classifier - on cluster
	- `3_do_prediction.ipynb`: apply model and get predictions - on laptop 
	- `README_Python_Installation.pdf`
	- `imageclassification.yml`: env file 
- Additionally 
	- `explorations`: contains additional notebooks from various explorations using `deeplearning`, `cnn` and `transfer-learning`
	- `1A_read_images_augment.ipynb` for data augmentation part 

	
*This project was original done in collaboration with Rudrarup Bose, PhD at Tang lab (Nov 2021, MPI-CBG, Dresden). The code is published with permission.*
	
