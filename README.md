# Solution to the Spotify Skip Prediction
<img src="https://dnczkxd1gcfu5.cloudfront.net/images/challenges/image_file/50/spotify.png" alt="Spotify-Logo" width="150"/><img src="https://github.com/crowdAI/crowdai/raw/master/app/assets/images/misc/crowdai-logo-smile.svg?sanitize=true" alt="CrowdAI-Logo" width="300"/>

[The link to the competition](https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge)

# Short summary

The approach chosen was to make a usual ML problem out of the challenge task. 
Independent models are built for each tarck in the second half of the user session.
Thus, such training does not directly optimise the competition metrics, 
but aims to make good predictions for each individual track.

The main issue was the size of the dataset, that is far too large to fit into memory on a local machine.
For preprocessing `dask` was used to allow `pandas`-like processing but perform partitioned operations.
For modelling, no good solution was found within limited timeframe.
There are several frameworks that allow online learning, like:
* [vowpal wabbit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Learning-algorithm), 
* [sklearn has several models, that implement `partial_fit()` method](https://scikit-learn.org/0.15/modules/scaling_strategies.html#incremental-learning),
* [dask-ml has an `Incremental` wrapper around the sklearn models with `partial_fit` to run on a `dask.Array`](https://dask-ml.readthedocs.io/en/latest/incremental.html),
* [XGBoost allows for processing with external memory](https://xgboost.readthedocs.io/en/latest/tutorials/external_memory.html),
* [H2O has checkpointing mechanism, that allows to continue training any H20 model on new data](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/checkpointing-models.html). 
This is slightly different from the other options, as it allows to adjust the model to changing patterns in the data,
instead of finding a pattern in a larger volume of data.

However, all of those have downsides: 
* XGBoost and vowpal wabbit read directly files in specific format, which would mean dumping of processed features on disk with additional large disk space needed;
* most of models with `partial_fit` are quite simple, which affects both dask-ml and sklearn.

Due to lack of time non of those options was tried.
Therefore modelling was restricted to in-memory training with a usual sklearn API in python.
This restricts the amount of data that one can use.
On the private laptop with the full set of features I was not able to go beyond 6 files.

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

The tools to do preprocessing are [Preprocess_Dask.ipynb](Preprocess_Dask.ipynb) and [Preprocess_Dask_AcousticVector.ipynb](Preprocess_Dask_AcousticVector.ipynb). 
There is one file produced byt each of those per input file. The two outputs can be directly concatenated together.
* The first extracts all features except `acoustic_vector`, adjusts feature types and stores output in `HDF`(==`h5`) format.
The format choice was made to optimise readout speed (more than factor 5 faster than `csv.gz`)
at the price of slightly higher disk space (50% more than `csv.gz`, but buch smaller than the plain `csv`).
* The second extracts only acoustic vector. 
This is an atrifact of the original pre-processing, that did not include these data
and a separate production was faster than complete re-run.

There is also [Preprocess_Pandas.ipynb](Preprocess_Pandas.ipynb), which is similar to the dask implementation and does pre-processing recursively.
However, pandas runs in a single thread only, so this version is much slower
and does not scale well to a high-performance cluster.

[Get_y_truth.ipynb](Get_y_truth.ipynb) extracts the competition target for the second halves of the sessions in the training data.
THe format is a `pandas.Series` of lists of `skip_2` values. 
There is one file produced per input file.

# Modelling
For modelling, a GBM implementation in [lightgbm](https://lightgbm.readthedocs.io/en/latest/) was used.
Individual models were build for the 1st, 2nd, ..., 10th track in the second half of the session.
The following features were used by all those models:
* Full information about the last track from the first half of the session;
* Mean, min, max aggregates for all features over the first half;
* For each track feature for a track in the second half there is:
   - the difference calculated wrt mean of that feature for `skip_2==0` and `skip_2==1` tracks in the first half;
   - the significance of that difference is calculated dividing the difference
   with the standard deviation of the feature with the same selection as the difference;
   - the difference between `skip==0` and `skip==1` is calculated for all featured from the previous two steps;
   - the motivation for these features was to evaluate how similar is a new track to those that user skiped and didn't skip.
* For each track the predicted probability (=confidence) of the models for the previous tracks in the session
was calculated and added as features.
This was added predicting directly on the whole dataset for the second halves, i.e. OOF method was not used.
The reason for this compromise was that OOF would have been too slow and model performance on the training
and validation data was very similar, thus no significant bias is expected.

At the end, a classifier was built on the features described above to predict `skip_2`
for each track in the second half of the session.
Logloss objective function was used and binary error rate was used as metric for early stopping criterion.

[Model_Building.ipynb](Model_Building.ipynb) has the primary model-building procedure.
[Model_Building_Iterations.ipynb](Model_Building_Iterations.ipynb) contains a streamlined version to build several independent models
on independent training subsets.
The predictions of such models are be averaged to improve performance.

# Model evaluation

Model evaluation is performed locally on an independent sub-set of files, 
that were not used in training, and was found to give a reliable estimate of the score on the leader board.
[Local_evaluation.ipynb](Local_evaluation.ipynb)  containes the procedure for evaluation outlined.

# Misc

A set of helper functions used in different notebooks is collected in [helpers.py](helpers.py).

# Acknowledgements  
We would like to thank our co-organizers from WSDM and CrowdAI for making this challenge possible.
