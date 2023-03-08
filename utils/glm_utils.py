import numpy as np
import pandas as pd

from patsy import dmatrix
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
from nilearn.glm.first_level.hemodynamic_models import spm_hrf, \
spm_time_derivative, spm_dispersion_derivative




def boxcar(event_df, basis_type, tr, resample_tr, 
           n_scans, slicetime_ref, impulse_dur=0.5):
    # get time samples of functional scan based on slicetime reference
    frametimes = np.linspace(slicetime_ref, 
                             (n_scans - 1 + slicetime_ref) * tr, n_scans)
    # Create index based on resampled tr
    h_frametimes = np.arange(0, frametimes[-1]+1, resample_tr) 
    # Grab onsets from event_df
    onsets = event_df.onset.values
    # if basis is spline, create unit impulse, else, use duration values from event df
    if basis_type == 'spline':
        block_dur = impulse_dur
        amp_val = event_df.duration.values
    else:
        block_dur = event_df.duration.values
        amp_val = np.repeat(1, len(block_dur))
    # initialize zero vector for event time course
    event_ts = np.zeros_like(h_frametimes).astype(np.float64)
    tmax = len(h_frametimes)
    # Get samples nearest to onsets
    t_onset = np.minimum(np.searchsorted(h_frametimes, onsets), tmax - 1)
    for t, a in zip(t_onset, amp_val):
        event_ts[t] = a

    t_offset = np.minimum(np.searchsorted(h_frametimes, onsets + block_dur), tmax - 1)
    for t, a in zip(t_offset, amp_val):
        event_ts[t] -= a
        
    event_ts = np.cumsum(event_ts)

    return event_ts, t_onset, frametimes, h_frametimes


def convolve_regressor(event_ts, basis_type, basis, block_dur, 
                       resample_tr, t_onset, event_lag, max_dur):
    # Convolve basis with event time course
    # Modified from nilearn/nilearn/glm/first_level/hemodynamic_models.py 
    if basis_type == 'spline':
        basis_lag = basis['spline']['lag']
        basis_dur = basis['spline']['dur']
        # convert to pandas series
        event_ts = pd.Series(event_ts)
        basis_dur_len = int(max_dur/resample_tr)
        # Don't estimate lags beyond block dur
        max_lag_block = np.ceil((block_dur + event_lag)/resample_tr).astype(int)
        # Put event_ts in duration basis
        event_basis = dmatrix(basis_dur.design_info, {'x': event_ts}, return_type='dataframe')
        # Create vector of lags
        lag_vec = np.arange(basis_dur_len).astype(int)
        # Intialize crossbasis matrix (regressor_mat)
        regressor_mat = np.zeros((event_basis.shape[0], basis_lag.shape[1]*basis_dur.shape[1]))
        # Loop through predictor and lag bases and multiply column pairs
        indx = 0
        for v in np.arange(event_basis.shape[1]):
            lag_mat = pd.concat([event_basis.iloc[:,v].shift(i, fill_value=0) for i in lag_vec], axis=1).values
            for t, max_lag in zip(t_onset, max_lag_block):  
                lag_mat[t:(t+int(basis_dur_len)),max_lag:] = 0
            for l in np.arange(basis_lag.shape[1]):
                regressor_mat[:, indx] = np.dot(lag_mat, basis_lag.iloc[:,l].values)
                indx+=1

    elif basis_type in ['hrf', 'hrf3']:
        if 'hrf' in basis:
            basis_df = basis['hrf']
        elif 'hrf3' in basis:
            basis_df = basis['hrf3']
        regressor_mat = np.array([np.convolve(event_ts, basis_df.iloc[:,h])[:event_ts.size]
                                 for h in range(basis_df.shape[1])]).T

    return regressor_mat


def create_basis(basis_type, tr, resample_tr, max_dur, nknots_dur=4):
    # define basis for creating event regressors
    # Use Natural Cubic spline basis based on Patsy package
    basis = {}
    if basis_type == 'spline':
        basis['spline'] = {}
        # basis along lag dimension
        # spline knot positions are hard-coded (informed by canonical hrf)
        knots_sec = [1, 3, 6, 9, 15, 25] # in secs
        knots_t = [k/resample_tr for k in knots_sec] # convert to samples
        basis_dur_len = int(max_dur/resample_tr) # maximum length of basis set by max_dur (in secs)
        # Create increasing index up to max lag
        lag_vec = np.arange(basis_dur_len).astype(int) 
        # Create Natural Cubic Spline basis for lagged event impulse (using patsy dmatrix)
        basis_lag = dmatrix("cr(x, knots=knots_t) - 1",
                            {"x": lag_vec}, return_type='dataframe')
        basis_lag.columns = [f'Knot{i}' for i in range(basis_lag.shape[1])]
        basis['spline']['lag'] = basis_lag
        # Basis for duration values
        dur_knots = np.linspace(1, 30, nknots_dur) 
        duration_vec = np.arange(60)
        basis_dur = dmatrix('cr(x, knots=dur_knots) - 1', {'x': duration_vec}, 
                            return_type='dataframe')
        basis_dur.columns = [f'Dur_Knot{i}' for i in range(basis_dur.shape[1])]
        basis['spline']['dur'] = basis_dur
    # Canonical HRF 
    elif basis_type == 'hrf':
        basis_hrf = pd.DataFrame({'hrf': spm_hrf(tr, 100)})
        basis['hrf'] = basis
    # Canonical HRF, along w/ its derivative and disperion
    elif basis_type == 'hrf3':
        basis_hrf = pd.DataFrame({
        'hrf': spm_hrf(tr, 100), 
        'hrf_derivative': spm_time_derivative(tr, 100), 
        'hrf_dispersion': spm_dispersion_derivative(tr, 100)
        })
        basis['hrf3'] = basis_hrf 

    # Add metadata to basis dataframe through attrs attribute
    basis['metadata'] = {
        'basis_type': basis_type,
        'tr': tr,
        'resample_tr': resample_tr,
        'max_dur': max_dur # only used for spline basis
    }

    return basis


def create_regressor(event_df, basis_type, tr, n_scans, slicetime_ref, 
                     task_on=True, event_lag = 4, resample_tr = 0.01, 
                     max_dur=30):
    """
    Core function for constructing fMRI regressors from task events

    This function takes a supplied BIDS event dataframe and basis, and construct a
    task fMRI regressor ready for regression

    Much of the code is modified from:
    Modified from nilearn/nilearn/glm/first_level/hemodynamic_models.py 

    Parameters:
    event_df (pd.DataFrame): BIDS compliant event dataframe -
        w/ onsets, duration and trial type, in that order
    basis_type (str): type of basis (can be spline, hrf and hrf3). 'spline' is a natural
        cubic spline basis with prespecified knot locations (N=6). 'hrf' is the canonical
        hrf function. 'hrf3' is the canonical hrf function with its derivative and dispersion
    tr (float): repetition time of functional scan
    n_scans (int): number of time points of functional scan
    slicetime_ref (float): the time between consecutive samples that slices are re-referenced to
    task_on (bool): whether to model all events as the same task
    event_lag (float): how much time (in secs) after the end of an event to model for 
        spline basis (ignored if not spline basis)
    resample_tr (float): sampling rate of high-resolution event regressor 
        for spline basis (ignored if not spline basis)
    max_dur (float): maximum duration of spline basis (ignored if not spline basis)

    Returns:
    pd.DataFrame: dataframe containing task regressors
  
    """
    
    # Create basis 
    basis = create_basis(basis_type, tr, resample_tr, max_dur)
    # if task_on is set to True, set all trial types as one type
    if task_on:
        event_df['trial_type'] = 'task'
    # Loop through trial types and create regressors
    regressor_all = []
    trial_types = event_df.trial_type.unique()
    for trial_t in trial_types:
        event_df_trial = event_df.loc[event_df.trial_type == trial_t].copy()
        # Create boxcar event time course
        event_ts, onsets, frametimes, h_frametimes = boxcar(event_df_trial, basis_type, 
                                                            tr, resample_tr, n_scans, 
                                                            slicetime_ref)
        # Convolve basis with event time course
        block_dur = event_df_trial.duration.values
        regressor = convolve_regressor(event_ts, basis_type, basis, block_dur, 
                                       resample_tr, onsets, event_lag, max_dur)
        regressor_low = interpolate_regressor(regressor, frametimes, h_frametimes)
        regressor_df = name_regressors(regressor_low, basis_type)
        regressor_all.append(regressor_df)

    regressor_all = pd.concat(regressor_all, axis=0)
    return regressor_all, basis


def duration_regressor(ev_concat, ev_df_concat, df_dur_spline=4):
    # initialize zero vector for event time course
    duration_reg = []
    for ev, ev_df in zip(ev_concat, ev_df_concat):
        dur_ts = np.ones(len(ev)) * ev_df.duration.max()
        duration_reg.append(dur_ts)

    duration_reg = np.concatenate(duration_reg)
    ev_concat = pd.concat(ev_concat, axis=0, ignore_index=True)
    duration_basis = dmatrix("cr(x, df=df_dur_spline) - 1", {"x": duration_reg}, 
                             return_type='dataframe')
    duration_basis.columns = [f'duration_reg_{i}' for i in range(duration_basis.shape[1])]
    inter_df = pairwise_interaction_terms(ev_concat, duration_basis)

    return inter_df, duration_basis


def interpolate_regressor(regressor, frametimes, h_frametimes):
    # nilearn/nilearn/glm/first_level/hemodynamic_models/_resample_regressor.py
    f = interp1d(h_frametimes, regressor.T)
    return f(frametimes).T


def name_regressors(regressor, basis_type):
    if basis_type == 'spline':
        cols = [f'Knot{i}' for i in range(regressor.shape[1])]
        regressor_df = pd.DataFrame(regressor, columns=cols)
    elif basis_type == 'hrf':
        regressor_df = pd.DataFrame(regressor, columns=['HRF'])
    elif basis_type == 'hrf3':
        cols=['HRF', 'Derivative', 'Dispersion']
        regressor_df = pd.DataFrame(regressor, columns=cols)
    return regressor_df


def pairwise_interaction_terms(df1, df2):
    cols1 = df1.columns
    cols2 = df2.columns
    inter_list = {f'{c1}_{c2}': df1[c1].mul(df2[c2]) for c1 in cols1 for c2 in cols2}
    inter_df = pd.DataFrame(inter_list)
    inter_df = pd.concat([df1, df2, inter_df], axis=1)
    return inter_df


def spline_model_predict(model, basis, amp=1, eval_sampling_rate=0.25, 
                         pred_impulse_dur=0.5, post_dur_s=6):
    # Evaluate impulse response of model
    # Get meta data
    resample_tr = basis['metadata']['resample_tr']
    max_dur = basis['metadata']['max_dur']
    # Get lag and duration bases
    basis_lag = basis['spline']['lag']
    basis_dur = basis['spline']['dur']
    # Get length of basis 
    basis_dur_len = basis_lag.shape[0]
    # Get increasing index up to length of basis
    lag_vec = np.arange(basis_dur_len)
    # Loop through different possible duration values
    pred_bold = []
    duration_s = [1, 5, 10, 15, 20, 30] # hard coded duration values
    for dur in duration_s:
        # resampled values for prediction
        lag_vec_resamp = np.arange(0, min(dur+post_dur_s, max_dur), 
                                   eval_sampling_rate)/resample_tr
        # Initiliaze impulse predictor
        event_pred = pd.Series(np.zeros(basis_dur_len))
        # Start the impulse at the beginning of the vector and set value as duration
        event_pred[:int(pred_impulse_dur/resample_tr)] = dur
        # Put event_ts in duration basis
        event_pred_basis = dmatrix(basis_dur.design_info, {'x': event_pred}, return_type='dataframe')
        # We must center our predictions around a reference value (set at 10s duration)
        cen = 10
        # Rescale 
        basis_cen = dmatrix(basis_dur.design_info, {'x': cen}, return_type='dataframe')
        event_pred_basis = event_pred_basis.subtract(basis_cen.values, axis=1)
        # Intialize crossbasis predictor matrix (regressor_mat)
        pred_regressor_mat = np.zeros((event_pred_basis.shape[0], basis_lag.shape[1]*basis_dur.shape[1]))
        # Loop through predictor and lag bases and multiply column pairs
        indx = 0
        for v in np.arange(event_pred_basis.shape[1]):
            lag_mat = pd.concat([event_pred_basis.iloc[:,v].shift(i, fill_value=0) for i in lag_vec], axis=1).values
            for l in np.arange(basis_lag.shape[1]):
                pred_regressor_mat[:, indx] = np.dot(lag_mat, basis_lag.iloc[:,l].values)
                indx+=1

        pred_regressor_mat_resamp = interpolate_regressor(pred_regressor_mat, lag_vec_resamp, lag_vec)
        pred = model.predict(pred_regressor_mat_resamp)
        pred_bold.append((dur, pred))

    return pred_bold

    



