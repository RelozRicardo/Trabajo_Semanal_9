#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Inicialización e importación de módulos
"""
Created on Thu Sep  5 10:04:34 2024

@author: ricardo
"""

# Módulos para Jupyter
import warnings
warnings.filterwarnings('ignore')

# Módulos importantantes
import scipy.signal as sig
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla

fig_sz_x = 10
fig_sz_y = 7
fig_dpi = 100 # dpi

fig_font_size = 16

mpl.rcParams['figure.figsize'] = (fig_sz_x,fig_sz_y)
plt.rcParams.update({'font.size':fig_font_size})

###
## Señal de ECG registrada a 1 kHz, con contaminación de diversos orígenes.
###

# para listar las variables que hay en el archivo
#io.whosmat('ecg.mat')
#mat_struct = sio.loadmat('ecg.mat')

#ecg_one_lead = mat_struct['ecg_lead']
#ecg_one_lead = ecg_one_lead.flatten()
#cant_muestras = len(ecg_one_lead)

fs = 1000 # Hz
nyq_frec = fs / 2

# Plantilla

# filter design
ripple = 0 # dB
atenuacion = 40 # dB

ws1 = 1.0 ; wp1 = 3.0 ; wp2 = 25.0 ; ws2 = 35.0 #Hz

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains/20)




num = sig.firwin2(501 , frecs , gains)
print(num[72:78])

den = 1.0

# muestreo el filtro donde me interesa verlo según la plantilla.
w  = np.append(np.logspace(-1, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w  = np.append(w, np.linspace(110, nyq_frec, 100, endpoint=True) ) / nyq_frec * np.pi

_, hh_win = sig.freqz(num, den, w)

# renormalizo el eje de frecuencia
w = w / np.pi * nyq_frec

plt.plot(w, 20 * np.log10(abs(hh_win)), label='FIR {:d}'.format(num.shape[0]))

plt.title('Filtros diseñados') ; plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]') ; plt.grid() ; plt.axis([0, 100, -60, 5 ])

axes_hdl = plt.gca()
axes_hdl.legend()

plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs)

plt.figure()

plt.subplot(2, 1, 1) ; plt.xlim(0,40) ; plt.grid(True) ; plt.ylabel('Phase [rad]')
plt.plot(w/np.pi , np.angle(hh_win) )
plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

hhRET = np.diff(np.angle(hh_win)) / np.diff(w/np.pi) 
fRET = ( w[:-1] + w[1:] ) /  (2*np.pi)

plt.subplot(2, 1, 2) ; plt.ylabel('Retardo [seg]') ; plt.yticks([-500 , -250 , 0 , 250 , 500])
plt.ylim(-525,525) ; plt.xlim(0,40) ; plt.grid(True) ; plt.xlabel('Frecuencia [Hz]')
plt.plot( fRET , -hhRET )
plt.show()

plt.figure()

impulse = sig.unit_impulse(len(num),'mid')
respuesta = np.convolve(impulse,num,mode='full')
tRES = np.arange(len(respuesta)) / 1000

plt.xlabel('Tiempo [seg]') ; plt.grid(True) ; plt.plot( tRES , respuesta )
plt.ylabel('Impulso')
plt.show()
