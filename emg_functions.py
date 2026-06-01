# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:35:10 2023

@author: Krzysztof
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd


#funkcja usuwająca offset sygnału
def remove_mean(time, emg):
    # process EMG signal: remove mean
    emg_correctmean = emg - np.mean(emg)

    # plot comparison of EMG with offset vs mean-corrected values
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Mean offset present')
    plt.plot(time, emg)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Mean-corrected values')
    plt.plot(time, emg_correctmean)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    fig.tight_layout()
    fig_name = 'fig2.png'
    fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name)
    
    return emg_correctmean

#funkcja do filtracji sygnału
def emg_filter(time, emg_correctmean):
    # create bandpass filter for EMG
    high = 20/(1000/2)
    low = 450/(1000/2)
    b, a = sp.signal.butter(4, [high,low], btype='bandpass')
    
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b, a, emg_correctmean)
    
    # plot comparison of unfiltered vs filtered mean-corrected EMG
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unfiltered EMG')
    plt.plot(time, emg_correctmean)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Filtered EMG')
    plt.plot(time, emg_filtered)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    
    fig.tight_layout()
    fig_name = 'fig3.png'
    fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name)
    
    return emg_filtered

#funkcja do usuwania wartosci ujemnych sygnalu
def emg_rectify(time, emg_filtered):
    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)
    
    # plot comparison of unrectified vs rectified EMG
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unrectified EMG')
    plt.plot(time, emg_filtered)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Rectified EMG')
    plt.plot(time, emg_rectified)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    
    fig.tight_layout()
    fig_name = 'fig4.png'
    fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name)
    
    return emg_rectified

def filteremg(time, emg, low_pass=10, sfreq=1000, high_band=20, low_band=450):
    """
    time: Time data
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """
    
    # normalise cut-off frequencies to sampling frequency
    high_band = high_band/(sfreq/2)
    low_band = low_band/(sfreq/2)
    
    # create bandpass filter for EMG
    b1, a1 = sp.signal.butter(4, [high_band,low_band], btype='bandpass')
    
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b1, a1, emg)    
    
    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)
    
    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass/(sfreq/2)
    b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)
    
    # plot graphs
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.subplot(1, 3, 1).set_title('Unfiltered,' + '\n' + 'unrectified EMG')
    plt.plot(time, emg)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    
    plt.subplot(1, 3, 2)
    plt.subplot(1, 3, 2).set_title('Filtered,' + '\n' + 'rectified EMG: ' + str(int(high_band*sfreq)) + '-' + str(int(low_band*sfreq)) + 'Hz')
    plt.plot(time, emg_rectified)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.plot([0.9, 1.0], [1.0, 1.0], 'r-', lw=5)
    plt.xlabel('Time (sec)')

    plt.subplot(1, 3, 3)
    plt.subplot(1, 3, 3).set_title('Filtered, rectified ' + '\n' + 'EMG envelope: ' + str(int(low_pass*sfreq/2)) + ' Hz')
    plt.plot(time, emg_envelope)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.plot([0.9, 1.0], [1.0, 1.0], 'r-', lw=5)
    plt.xlabel('Time (sec)')
    '''
    plt.subplot(1, 4, 4)
    plt.subplot(1, 4, 4).set_title('Focussed region')
    plt.plot(time[int(0.9*1000):int(1.0*1000)], emg_envelope[int(0.9*1000):int(1.0*1000)])
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlim(0.9, 1.0)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    '''
    fig_name = 'fig_' + str(int(low_pass*sfreq)) + '.png'
    fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name)
    
    return emg_filtered, emg_envelope

def import_data(separator):
    def time_norm(data):
        times = list(data['t'])
        adjusted_times = []
        offset = 0

        for i in range(len(times)):
            if i > 0 and times[i] < times[i - 1]:
                offset += times[i - 1] - times[i] + 1
            adjusted_times.append(times[i] + offset)

        return pd.DataFrame({'emg': data['emg'], 't': adjusted_times})

    column_names = ['emg', 't']
    weights_list = []
    mvc_list = []
    fatigue_list = []

    for i in range(3):
        weights_file = f'Weight{i+1}.txt'
        mvc_file = f'MVC{i+1}.txt'
        fatigue_file = f'Fatigue{i+1}.txt'

        weights_list.append(pd.read_csv(
            weights_file, sep=separator, names=column_names,
            skiprows=50, skipfooter=50, engine='python'
        ))

        mvc_list.append(pd.read_csv(
            mvc_file, sep=separator, names=column_names,
            skiprows=50, skipfooter=50, engine='python'
        ))

        fatigue_list.append(pd.read_csv(
            fatigue_file, sep=separator, names=column_names,
            skiprows=50, skipfooter=50, engine='python'
        ))

    weights_raw = pd.concat(weights_list, ignore_index=True)
    mvc_raw = pd.concat(mvc_list, ignore_index=True)
    fatigue_raw = pd.concat(fatigue_list, ignore_index=True)

    weights = time_norm(weights_raw)
    mvc = time_norm(mvc_raw)
    fatigue = time_norm(fatigue_raw)

    return weights, mvc, fatigue



