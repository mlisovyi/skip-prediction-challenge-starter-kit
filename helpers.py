import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', 500)

import keggler as kg

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import gc
gc.enable()

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import os, psutil
import glob

from collections import OrderedDict

aggs = OrderedDict([('context_switch', ['mean']),
        ('no_pause_before_play', ['mean']),
        ('short_pause_before_play', ['mean']),
        ('long_pause_before_play', ['mean']),
        ('hist_user_behavior_is_shuffle', ['mean']),
        ('duration', ['mean', 'max', 'min']),
        ('us_popularity_estimate', ['mean', 'max', 'min']),
        ('release_year', ['mean', 'max', 'min']),
                   ])

# aggs = {'context_switch': ['mean'],
#         'no_pause_before_play': ['mean'],
#         'short_pause_before_play': ['mean'],
#         'long_pause_before_play': ['mean'],
#         'hist_user_behavior_is_shuffle': ['mean'],
#         'duration': ['mean', 'max', 'min'],
#         'us_popularity_estimate': ['mean', 'max', 'min'],
#         'release_year': ['mean', 'max', 'min'],
#        }
for i in [1,2,3,4]:
    aggs['skip_{}'.format(i)] = ['mean']
    
list_musik_qualities = ['acousticness', 'beat_strength', 'bounciness',
       'danceability', 'dyn_range_mean', 'energy', 'flatness',
       'instrumentalness', 'key', 'liveness', 'loudness', 'mechanism', 'mode',
       'organism', 'speechiness', 'tempo', 'time_signature', 'valence']

col_dtype = {'context_switch': np.uint8,
         'context_type': np.uint8,
         'duration': np.float16,
         'hist_user_behavior_is_shuffle': np.uint8,
         'hist_user_behavior_n_seekback': np.uint8,
         'hist_user_behavior_n_seekfwd': np.uint8,
         'hist_user_behavior_reason_end': np.int8,
         'hist_user_behavior_reason_start': np.int8,
         'hour_of_day': np.int8,
         'long_pause_before_play': np.int8,
         'no_pause_before_play': np.int8,
         'not_skipped': np.int8,
         'premium': np.int8,
         'release_year': np.int16,
        'session_length': np.int8,
         'session_position': np.int8,
         'short_pause_before_play': np.int8,
         'skip_1': np.int8,
         'skip_2': np.int8,
         'skip_3': np.int8,
         'us_popularity_estimate': np.float16,
                 'acousticness': np.float16, 'beat_strength': np.float16, 'bounciness': np.float16,
       'danceability': np.float16, 'dyn_range_mean': np.float16, 'energy': np.float16, 'flatness': np.float16,
       'instrumentalness': np.float16, 'key': np.int16, 'liveness': np.float16, 'loudness': np.float16, 'mechanism': np.float16, 'mode': np.int8,
       'organism': np.float16, 'speechiness': np.float16, 'tempo': np.float16, 'time_signature': np.int16, 'valence': np.float16
                }


aggs_music_qualities = OrderedDict()
for q in list_musik_qualities:
    if q != 'mode':
        aggs_music_qualities[q] = ['mean', 'std', 'min', 'max']
    else:
        aggs_music_qualities[q] = ['mean', 'std']

def evaluate(submission,groundtruth):
    ap_sum = 0.0
    first_pred_acc_sum = 0.0
    counter = 0
    for sub, tru in zip(submission, groundtruth):
        if len(sub) != len(tru):
            raise Exception('Line {} should contain {} predictions, but instead contains '
                            '{}'.format(counter+1,len(tru),len(sub)))
        ap_sum += ave_pre(sub,tru,counter)
        first_pred_acc_sum += sub[0] == tru[0]
        counter+=1
    ap = ap_sum/counter
    first_pred_acc = first_pred_acc_sum/counter
    return ap,first_pred_acc

def ave_pre(submission,groundtruth,counter):
    s = 0.0
    t = 0.0
    c = 1.0
    for x, y in zip(submission, groundtruth):
        if x != 0 and x != 1:
            raise Exception('Invalid prediction in line {}, should be 0 or 1'.format(counter))
        if x==y:
            s += 1.0
            t += s / c
        c += 1
    return t/len(groundtruth)


def evaluate_model(const_preds, y_truth_list):
    tmp_preds_constant_mdl = pd.DataFrame({'pred': const_preds, 'len': y_truth_list.apply(len)})
    preds_lists_stp = tmp_preds_constant_mdl.apply(lambda x: [x['pred']]*x['len'], axis=1)
    del tmp_preds_constant_mdl
    return evaluate(preds_lists_stp.tolist(), y_truth_list.tolist())



def evaluate_set_of_models(list_preds, y_truth_list, i_2fill=-2):
    preds_lists_stp = pred_series_of_lists(list_preds, y_truth_list.apply(len), i_2fill)
#     # transform predictions into a dataframe
#     tmp_preds_constant_mdl = pd.DataFrame({'pred_{}'.format(i): list_preds[i] for i in range(len(list_preds))}).astype(np.uint8)
#     # add a column with the residual desired length of complete session
#     tmp_preds_constant_mdl['len'] = y_truth_list.apply(len).values - len(list_preds)
#     # create a series with lists
#     preds_lists_stp = tmp_preds_constant_mdl.apply(lambda x: x.iloc[:-1].tolist() + [x.iloc[i_2fill]]*x['len'], axis=1)
#     del tmp_preds_constant_mdl
    return evaluate(preds_lists_stp.tolist(), y_truth_list.tolist())


from scipy import stats

def read_log(fin, cols_2read=[]):
    if fin.endswith('.csv') or fin.endswith('.csv.gz'):
        df_ = pd.read_csv(fin, dtype=col_dtype, nrows=None)
    elif fin.endswith('.h5'):
        df_ = pd.read_hdf(fin, key='df')
    else:
        return None
    for c in ['hist_user_behavior_n_seekback', 'hist_user_behavior_n_seekfwd']:
        if c in df_.columns:
            df_.drop(c, axis=1, inplace=True)
    if len(cols_2read) > 0:
        df_ = df_[cols_2read]
    return df_

def fe(df_):
    df_['short_pause_before_play'] = (df_['long_pause_before_play'] - df_['short_pause_before_play']).astype(np.int8)
    df_['hist_user_behavior_reason_end_not_start'] = (df_['hist_user_behavior_reason_end'] == df_['hist_user_behavior_reason_start']).astype(np.uint8)
    df_['skip_2_SUB_1'] = (df_['skip_2'] - df_['skip_1']).astype(np.int8)
    df_['skip_3_SUB_2'] = (df_['skip_3'] - df_['skip_2']).astype(np.int8)
    df_['skip_2_ADD_1'] = (df_['skip_2'] + df_['skip_1']).astype(np.int8)
    df_['skip_3_ADD_2'] = (df_['skip_3'] + df_['skip_2']).astype(np.int8)
    df_['skip_3_ADD_4'] = (df_['skip_3'] + df_['skip_4']).astype(np.int8)
    return df_

def get_halves_split(df_):
    is_first_half = (df_['session_position'] <= 0.5*df_['session_length'])
    return df_[is_first_half], df_[~is_first_half]
    

def get_XY(df_, aggs_, reset_index=False, list_musik_qualities_=[], aggs_music_qualities_={}, i_=0):
    is_tst = False
    if type(df_) == pd.DataFrame:
        df_X, df_y = get_halves_split(df_)
    else:
        df_X, df_y = df_[0], df_[1]
        is_tst = True
    
    # feature engineering
    df_X = fe(df_X)
    
    # Last track info
    X_trn = df_X.groupby('session_id').last()
    # reduce memory
    for c in X_trn.columns:
        X_trn[c] = X_trn[c].astype(df_X[c].dtype)
    # aggregates
    X_agg = df_X.groupby('session_id').agg(aggs_).astype(np.float32)
    X_agg.columns = pd.Index(['AGG_' + e[0] + "_" + e[1].upper() for e in X_agg.columns])
    X_trn = X_trn.merge(X_agg, left_index=True, right_index=True, how='left')
#     display(X_trn.head())
    
    
    if type(i_) is not list and type(i_) is not tuple:
        i_ = [i_]
    y_trn = []
    X_trk = []
    X_trk_agg = {}
    # make track aggregates for skipped and not skipped tracks
    skip_query = OrderedDict([('SKIP0', 'skip_2==0'),
                              ('SKIP1', 'skip_2==1')
                             ])
    for qname, query in skip_query.items():
        # skip and no-skip subsets
        df_tmp = df_X.query(query)
        # track-quality aggs
        X_agg_tmp = df_tmp.groupby('session_id').agg(aggs_music_qualities_).astype(np.float32)
        X_agg_tmp.columns = pd.Index(['AGG_' + e[0] + "_" + e[1].upper() for e in X_agg_tmp.columns])
        # store the dataframe with aggregates
        X_trk_agg[qname] = X_agg_tmp
        del df_tmp
        
    for i__ in i_:
        df_y_nth = df_y.groupby('session_id').nth(i__)
        if not is_tst:
            y_trn.append(df_y_nth['skip_2'])
        elif i__==0:
            y_trn.append(df_y_nth['session_length']-df_y_nth['session_position']+1)
        X_nth_trk = pd.DataFrame(index=X_trn.index)
        # fill in difference of track qualities wrt aggregates for skipped and not tracks
        if list_musik_qualities_:
            for qname, query in skip_query.items():
                X_agg_tmp = X_trk_agg[qname]
                for q in list_musik_qualities_:
                    cols_q_agg = [c for c in X_agg_tmp.columns if c.startswith('AGG_'+q) and not c.endswith('_STD')]
                    for c in cols_q_agg:
                        X_nth_trk['{}_diff_{}'.format(c, qname)] = X_agg_tmp[c] - df_y_nth[q]
                        if c.endswith('_MEAN'):
                            X_nth_trk['{}_sign_{}'.format(c, qname)] = X_nth_trk['{}_diff_{}'.format(c, qname)] / X_agg_tmp[c[:-5]+'_STD']
                del X_agg_tmp
            # add a column with a difference between SKIP0 and SKIP1 aggregate differences
            cols_track_diff = [c for c in X_nth_trk.columns if c.endswith('_SKIP0')]
            for c in cols_track_diff:
                name_base = c[:-6]
                X_nth_trk[name_base+'_SKIPDIFF'] = X_nth_trk[name_base+'_SKIP0'] - X_nth_trk[name_base+'_SKIP1']
                
            X_trk.append(X_nth_trk)
        
#     display(X_trn.head())

#     X_median = (df_X.groupby('session_id')
#                 [
#                  'hist_user_behavior_reason_end'
#                 ].agg(lambda x: stats.mode(x)[0][0])
#                ).rename('AGG_hist_user_behavior_reason_end_MEDIAN')
#     X_trn = pd.concat([X_trn, X_median], axis=1)
#     display(X_median.head())
    
    # drop useless columns
    X_trn.drop(['session_position'], axis=1, inplace=True)
    
    # read out also  the second test target
    
    if reset_index:
        X_trn.index = pd.RangeIndex(start=0, stop=len(X_trn))
        for y in y_trn:
#             if not is_tst:
            y.index = pd.RangeIndex(start=0, stop=len(X_trn))
        for X in X_trk:
            X.index = pd.RangeIndex(start=0, stop=len(X_trn))
    
    del df_y
    
    return X_trn, y_trn, X_trk

def get_y_truth(df_):
    _, df_y = get_halves_split(df_)
    
    ground_truth = []
    #df_y['session_id'] = df_y['session_id'].astype('object')
    # Here we process each session, saving a list containing the targets
    gb = df_y.groupby('session_id',sort=False).groups
    for s_id in tqdm(df_y['session_id'].unique()):
        #print(gb[s_id])
        ground_truth.append(df_y.loc[gb[s_id],'skip_2'].tolist())
        
    return ground_truth

def get_y_length(df_):
    _, df_y = get_halves_split(df_)
    
    first_track = df_y.groupby('session_id',sort=False).first()
    sess_length = first_track['session_length'] - first_track['session_position'] + 1
    return sess_length

def pred_series_of_lists(list_preds, y_length, i_2fill=-2):
    # transform predictions into a dataframe
    tmp_preds_constant_mdl = pd.DataFrame({'pred_{}'.format(i): list_preds[i] for i in range(len(list_preds))}).astype(np.uint8)
    # add a column with the residual desired length of complete session
    tmp_preds_constant_mdl['len'] = y_length.values - len(list_preds)
    # create a series with lists
    series_of_lists = tmp_preds_constant_mdl.apply(lambda x: x.iloc[:-1].tolist() + [x.iloc[i_2fill]]*x['len'], axis=1)
    del tmp_preds_constant_mdl
    return series_of_lists