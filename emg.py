# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:04:25 2023

@author: Krzysztof
"""
import matplotlib.pyplot as plt
import emg_lib as el
import emg_functions as ef
import numpy as np
import scipy as sp

#wczytanie danych pomiarowych
weights, mvc, fatigue = ef.import_data(',')

'''
plt.figure()
plt.plot(mvc.t, mvc.emg)
plt.figure()
plt.plot(weights.t, weights.emg)
plt.figure()
plt.plot(fatigue.t, fatigue.emg)
'''

#usunięcie offsetu
mvc_correctmean = ef.remove_mean(mvc.t, mvc.emg)
weights_correctmean = ef.remove_mean(weights.t, weights.emg)
fatigue_correctmean = ef.remove_mean(fatigue.t, fatigue.emg)
'''
#filtracja sygnału
emg_filter = ef.emg_filter(mvc.t, emg_correctmean)
#usuwanie wartosci ujemnych sygnału
emg_rectified = ef.emg_rectify(mvc.t, emg_filter)
'''

#filtracja + usuwanie wartosci ujemnych + obwiednia z wszystkich sygnałów
mvc_filt, mvc_env = ef.filteremg(mvc.t, mvc_correctmean, 0.5, 1000, 20, 450)
weights_filt, weights_env = ef.filteremg(weights.t, weights_correctmean, 0.5, 1000, 20, 450)
fatigue_filt, fatigue_env = ef.filteremg(fatigue.t, fatigue_correctmean, 0.5, 1000, 20, 450)


#wyznaczanie przedziałów pracy miesni
mvc_start, mvc_end, weights_start, weights_end, fatigue_start, fatigue_end = el.get_bursts(mvc_filt, weights_filt, fatigue_filt)
#%matplotlib qt

#srednia magnituda MVC z 3 pomiarów
mean_mvc1 = np.mean(mvc_env[mvc_start[0]:mvc_end[0]])
mean_mvc2 = np.mean(mvc_env[mvc_start[1]:mvc_end[1]])
mean_mvc3 = np.mean(mvc_env[mvc_start[2]:mvc_end[2]])
mean_mvc = (mean_mvc1+mean_mvc2+mean_mvc3)/3

#srednia magnitudy dla 3 kolejnych pomiarów obciążenia mięsnia (normalizacja do mvc)
load1 = (np.mean(weights_env[weights_start[0]:weights_end[0]]))/mean_mvc*100
load2 = (np.mean(weights_env[weights_start[1]:weights_end[1]]))/mean_mvc*100
load3 = (np.mean(weights_env[weights_start[2]:weights_end[2]]))/mean_mvc*100

#obliczenie widma mocy sygnału zmęczeniowego EMG
power1, frequencies1 = el.get_power(fatigue_filt[fatigue_start[0]:fatigue_end[0]], 1000)
power2, frequencies2 = el.get_power(fatigue_filt[fatigue_start[1]:fatigue_end[1]], 1000)
power3, frequencies3 = el.get_power(fatigue_filt[fatigue_start[2]:fatigue_end[2]], 1000)
plt.figure()
plt.plot(frequencies1,power1)
plt.plot(frequencies2,power2, 'red')
plt.plot(frequencies3,power3, 'green')

#filtracja widma
plt.figure()
low_pass = 5
low_pass = low_pass/(1000/2)
b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
pow_filt1 = sp.signal.filtfilt(b2, a2, power1)
plt.plot(frequencies1, pow_filt1)
b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
pow_filt2 = sp.signal.filtfilt(b2, a2, power2)
plt.plot(frequencies2, pow_filt2,'red')
b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
pow_filt3 = sp.signal.filtfilt(b2, a2, power3)
plt.plot(frequencies3, pow_filt3,'green')


#obliczanie mediany częstotliwosci
area_freq1 = sp.integrate.cumulative_trapezoid(pow_filt1, frequencies1, initial=0)
total_power1 = area_freq1[-1]
median_freq1 = frequencies1[np.where(area_freq1 >= total_power1 / 2)[0][0]]

area_freq2 = sp.integrate.cumulative_trapezoid(pow_filt2, frequencies2, initial=0)
total_power2 = area_freq2[-1]
median_freq2 = frequencies2[np.where(area_freq2 >= total_power2 / 2)[0][0]]

area_freq3 = sp.integrate.cumulative_trapezoid(pow_filt3, frequencies3, initial=0)
total_power3 = area_freq3[-1]
median_freq3 = frequencies3[np.where(area_freq3 >= total_power3 / 2)[0][0]]
'''
def pseudo_cumtrapz(y, x):
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2
    cum_area = np.concatenate([[0], np.cumsum(avg_y * dx)])
    return cum_area

#obliczanie mediany częstotliwosci
area_freq1 = pseudo_cumtrapz(pow_filt1, frequencies1)
total_power1 = area_freq1[-1]
median_freq1 = frequencies1[np.where(area_freq1 >= total_power1 / 2)[0][0]]
area_freq2 = pseudo_cumtrapz(pow_filt2, frequencies2)
total_power2 = area_freq2[-1]
median_freq2 = frequencies2[np.where(area_freq2 >= total_power2 / 2)[0][0]]
area_freq3 = pseudo_cumtrapz(pow_filt3, frequencies3)
total_power3 = area_freq3[-1]
median_freq3 = frequencies3[np.where(area_freq3 >= total_power3 / 2)[0][0]]
'''

median_freq = [median_freq1, median_freq2, median_freq3]

plt.figure()
plt.plot(median_freq, 'x')