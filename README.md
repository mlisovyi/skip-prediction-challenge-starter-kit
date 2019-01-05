# Solution to the Spotify Skip Prediction
<img src="https://dnczkxd1gcfu5.cloudfront.net/images/challenges/image_file/50/spotify.png" alt="Spotify-Logo" width="150"/><img src="https://github.com/crowdAI/crowdai/raw/master/app/assets/images/misc/crowdai-logo-smile.svg?sanitize=true" alt="CrowdAI-Logo" width="300"/>

[The link to the competition](https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge)

# Installation
```
git clone https://github.com/mlisovyi/spotify_skip_prediction
cd spotify_skip_prediction
pip install -r requirements.txt
```

# Dataset
Please download the dataset from [https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge/dataset_files](https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge/dataset_files), and extract the files to the `data/` folder. [Untar](http://how-to.wikia.com/wiki/How_to_untar_a_tar_file_or_gzip-bz2_tar_file) them (this might take some time) to have the following directory structure:

```bash
|-- data/
|   |-- training_set/ (training sessions)
|   |-- test_set/ (leaderboard partial sessions)
|   |-- track_features/ (track metadata and audio features)
|   |-- submissions/ (submissions folder - contains sample submissions)
```
In particular, I've used the complete test set (14 GB) 
and [the first part of the training dataset](https://os.zhdk.cloud.switch.ch/swift/v1/crowdai-public/spotify-sequential-skip-prediction-challenge/split-files/training_set_0.tar.gz) (10 GB)
together with the track features (1.2 GB).
All those data were unpacked locally and each file was gziped locally to reduce disk footprint 
(tools like `pandas` and `dask` can read directly from gziped csv files).

# Data preprocessing
For preprocessing of the original data I've used [dask](https://docs.dask.org/en/latest/)
and in particular [dask.DataFrame](https://docs.dask.org/en/latest/dataframe.html)
to allow pandas-like data manipulation in multi-threaded fashion out-of-core 
(since my local laptop could not read in whole dataset into RAM).

The tools to do preprocessing are [Preprocess_Dask.ipynb] and [Preprocess_Dask_AcousticVector.ipynb]. 
There is one file produced byt each of those per input file. The two outputs can be directly concatenated together.
* The first extracts all features except `acoustic_vector`, adjusts feature types and stores output in `HDF`(==`h5`) format.
The format choice was made to optimise readout speed (more than factor 5 faster than `csv.gz`)
at the price of slightly higher disk space (50% more than `csv.gz`, but buch smaller than the plain `csv`).
* The second extracts only acoustic vector. 
This is an atrifact of the original pre-processing, that did not include these data
and a separate production was faster than complete re-run.

There is also [Preprocess_Pandas.ipynb], which is similar to the dask implementation and does pre-processing recursively.
However, pandas runs in a single thread only, so this version is much slower
and does not scale well to a high-performance cluster.

[Get_y_truth.ipynb] extracts the competition target for the second halves of the sessions in the training data.
THe format is a `pandas.Series` of lists of `skip_2` values. 
There is one file produced per input file.

# Usage
Now you can refer to the list of Jupyter Notebooks for different aspects of the challenge and the datasets.
You can access all of them by :
```bash
jupyter notebook
```
## Available Notebooks
  
* [Baseline Submissions](https://github.com/crowdAI/skip-prediction-challenge-starter-kit/blob/master/baseline_submissions.ipynb)
  
* [Locally test the evaluation function](https://github.com/crowdAI/skip-prediction-challenge-starter-kit/blob/master/local_evaluation.ipynb)   

# Acknowledgements  
We would like to thank our co-organizers from WSDM and CrowdAI for making this challenge possible.
