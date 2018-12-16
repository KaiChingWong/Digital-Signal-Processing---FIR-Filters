""" 2 ECG Heartrate Detection: Task 1 """

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

ecg = np.loadtxt('Participant_run.dat')
fs = 1000 #Sampling Rate

t = ecg[:,0]
amplitude = ecg[:,1]

#Converting ECG into Milli Volt
step = (4.096+4.096)/2**12
mV = ((amplitude - 2**(12-1))*step/2000)*1000

plt.figure(1)
plt.plot(t,mV,linewidth=0.5)
plt.title('ECG after Running in Milli Volt')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.xlim(0,120000)
plt.savefig('Running_ECG_mV.svg')

#Removing DC and 50Hz noise
ntaps =1000 #number of tabs 
f1 =int(45.0/fs*ntaps)
f2 =int(55.0/fs*ntaps)
f0 =int(1/fs*ntaps)

f_resp = np.ones(ntaps)
f_resp[f1:f2+1] = 0
f_resp[ntaps-f2: ntaps- f1+1] = 0
f_resp[ntaps-f0: ntaps- 0+1] = 0
f_resp[0:f0+1] = 0

coeff_tmp = np.fft.ifft(f_resp)

coeff_tmp = np.real(coeff_tmp)
coeff = np.zeros(ntaps)
coeff[0:int(ntaps/2)] = coeff_tmp[int(ntaps/2):ntaps]
coeff[int(ntaps/2):ntaps] = coeff_tmp[0:int(ntaps/2)]

#FIR filtering
beat = signal.lfilter(coeff,1,mV)

plt.figure(2)
plt.plot(t,beat,linewidth=0.5)
plt.title('Filtering Run ECG with FIR filter')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.xlim(0,120000)
plt.savefig('Running_ECG_Removed_50Hz.svg')

#Extract a single heart beat and use as a template
abeat = beat[75400:75900]

plt.figure(3)
plt.plot(abeat)
plt.title('A Heart Beat Template')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.savefig('A_Heart_Beat_2.1.svg')

#Inverting the above template to create a FIR coefficient
h = abeat[::-1]

plt.figure(4)
plt.plot(h)
plt.title('Impulse Response')
plt.xlabel('Number of Tabs (n)')
plt.ylabel('h(n)')
plt.savefig('Inverted_Heart_Beat.svg')

#Plotting a Heart Beat and an inverting Heart Beat on the same graph
plt.figure(5)
plt.plot(abeat)
plt.plot(h)
plt.title('Heart Beat and Inverted Heart Beat')
plt.xlabel('Number of Tabs (n)')
plt.ylabel('h(n)')
plt.savefig('Heart_Beat_&_Inverted_Heart_beat.svg')

#Using the Matched filter to generate stronger detection
ibeat = signal.lfilter(h,1,beat)

plt.figure(6)
plt.plot(t,ibeat,linewidth=0.5)
plt.title('Matched filtering on after exercised ECG')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.xlim(0,120000)
plt.savefig('Matched_filtering_ECG.svg')
###############################################################################

###############################################################################
"""Task 2"""

def BPStoBPM(ibeat,th):
    beatsqu = ibeat**2 #Output from Matched filter
    BPM = [] #empty arry
    counteri = 0
    Threshold = th
    
    for i in range(len(beatsqu)): #output from match filter
        if beatsqu[i] > Threshold: #100 is the threshold point
            dt = (i-counteri) #difference in time in second by 1000
            counteri = i
            bpm = 1/dt*60000 #BPS to BPM
            if(bpm < 200 and bpm > 30): #200bpm to 30bpm as filter
                BPM.append(bpm) #append means plus
    BPM = np.delete(BPM,0)
    return BPM

participant = BPStoBPM(ibeat,2)

plt.figure(7)
plt.plot(participant)
plt.title('Momentary Heart Rate')
plt.xlabel('Time (ms)')
plt.ylabel('Momentary Heart Rate (bpm)')
plt.savefig('Momentary_Heart_Rate.svg')
###############################################################################