"""
Digital Signal Processing 4

Assignment 2: FIR Filters

By Kai Ching Wong (GUID:2143747W)
"""

import numpy as np
import matplotlib.pyplot as plt
from FIR_Fil import FIR_filter as fir

###############################################################################
"""Task 1"""

ecg = np.loadtxt('Ricky_ECG.dat')
fs = 1000 #Sampling Rate

t = ecg[:,0]
amplitude = ecg[:,1] #Only column 1

plt.figure(1)
plt.plot(t,amplitude)
plt.title('Column 1')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim(0,5000)
plt.savefig('Ricky_ECG.svg')

#Converting ECG into Milli Volt
step = (4.096+4.096)/2**12
mV = ((amplitude - 2**(12-1))*step/2000)*1000

plt.figure(2)
plt.plot(t,mV)
plt.title('ECG in Milli Volt')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.xlim(0,5000)
plt.savefig('Ricky_ECG_mV.svg')

#Extracting a heart beat
abeat = mV[2500:3300]

plt.figure(3)
plt.plot(abeat)
plt.title('A Heart Beat')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.savefig('A_Heart_Beat.svg')

#Converitng into Frequency Domain
xfecg = np.fft.fft(mV)
f = np.linspace(0,fs,len(xfecg))

plt.figure(4)
plt.plot(f,abs(xfecg))
plt.title('ECG in Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(-5,500)
plt.savefig('Ricky_ECG_Hz.svg')
###############################################################################

###############################################################################
""" 1 ECG Filtering: Task 3 """

#Creating Impulse Response with analytical calculation
f1 = 45/fs
f2 = 55/fs
n = np.arange(-200,200+1)
h = 2*f1*np.sinc(2*f1*n)-2*f2*np.sinc(2*f2*n)
h[200]=1+(2*(f1-f2))

plt.figure(5)
plt.plot(h)
plt.title('Impulse Response of a 50Hz Notch Filter')
plt.xlabel('Number of Tabs (n)')
plt.ylabel('h(n)')
plt.savefig('50Hz_Notch_Filter_Impulse_Response.svg')

#Frequency Response of the FIR filter with analytical calculation
h1 = h
xfh1 = np.fft.fft(h1)
fh1 = np.linspace(0,fs,len(xfh1))

plt.figure(6)
plt.plot(fh1, 20*np.log10(xfh1)) #Converting frequency response in dB
plt.title('Frequency Response in Decibel')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Decibel (dB)')
plt.savefig('Task_1.3_Frequency_response_dB.svg')

#Using Hamming Window Function
h = h * np.hamming(400+1)

xfh = np.fft.fft(h)
fh = np.linspace(0,fs,len(xfh))

plt.figure(7)
plt.plot(fh, 20*np.log10(xfh))
plt.title('Frequency Response with Hamming Window Function')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Decibel (dB)')
plt.savefig('Frequency_Response_Hamming.svg')

#Filtering ECG with FIR Filter
fil = fir(h)
filecg = np.zeros(len(mV))
for i in range(len(mV)):
    filecg[i] = fil.filter(mV[i])

plt.figure(8)
plt.plot(t,filecg)
plt.title('Filtering ECG with FIR Filter')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.xlim(0,5000)
plt.savefig('ECG_after_FIR.svg')

xfilecg = np.fft.fft(filecg) #Converting into Frequency Domain

plt.figure(9)
plt.plot(f,abs(xfilecg))
plt.title('Filtering ECG with FIR Filter in Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(-5,500)
plt.savefig('ECG_after_FIR_Hz.svg')
###############################################################################

###############################################################################
""" 1 ECG Filtering: Task 4 """

#Creating impulse response with numerical calculation
ntaps =1000 #number of tabs 
f1 =int(45.0/fs*ntaps) #Indice for 45Hz
f2 =int(55.0/fs*ntaps) #Indice for 55Hz
f0 =int(1/fs*ntaps) #Indice for baseline shift

f_resp = np.ones(ntaps)
f_resp[f1:f2+1] = 0 #remove noise
f_resp[ntaps-f2: ntaps- f1+1] = 0 #mirror of f2 and f1
f_resp[ntaps-f0: ntaps- 0+1] = 0 #mirror of baseline shift
f_resp[0:f0+1] = 0 #remove baseline shift

plt.figure(10)
plt.plot(f_resp)
plt.title('Discrete Spectrum')
plt.xlabel('Number of Tabs (n)')
plt.ylabel('Amplitude')
plt.savefig('Discrete_Spectrum.svg')

coeff_tmp = np.fft.ifft(f_resp)

coeff_tmp = np.real(coeff_tmp) #want only real value from complex number
coeff = np.zeros(ntaps) #empty impulse response
coeff[0:int(ntaps/2)] = coeff_tmp[int(ntaps/2):ntaps] #fix the signal position
coeff[int(ntaps/2):ntaps] = coeff_tmp[0:int(ntaps/2)]

plt.figure(11)
plt.plot(coeff)
plt.title('Impulse Response for Numerical Calculation')
plt.xlabel('Number of Tabs (n)')
plt.ylabel('h(n)')
plt.savefig('Impulse_Response_numerical_calculation.svg')

#Using Hamming Window Function
coeff = coeff * np.hamming(1000)

plt.figure(12)
plt.plot(coeff)
plt.title('Impusle Response with Hamming Window Function')
plt.xlabel('Number of Tabs(n)')
plt.ylabel('h(n)')
plt.savefig('Task_1.4_Impulse_Response_Hamming.svg')

fil = fir(coeff)
filecg = np.zeros(len(mV))
for i in range(len(mV)):
    filecg[i] = fil.filter(mV[i])

plt.figure(13)
plt.plot(t,filecg)
plt.title('Filtering ECG with FIR Filter using Numerical Calculation')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.savefig('FIR_ECG_Numerical_Calsulation.svg')

xfilecg = np.fft.fft(filecg) #Converting into Frequency Domain

plt.figure(14)
plt.plot(f,abs(xfilecg))
plt.title('Filtering ECG with FIR Filter using Numerical Calculation')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(-5,500)
plt.savefig('FIR_ECG_Numerical_Calsulation_Hz.svg')
###############################################################################