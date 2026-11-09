import h5py
import time
import math
import glob
import numpy as np
from scipy.signal import correlate
import scipy.interpolate
norm = np.linalg.norm
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from scipy.signal import welch


def slope_TEC(TEC,timevec,fs):
	times=[]
	add86400=False
	if np.nanmax(timevec)>86400:
		timevec = timevec%86400
		add86400 = True
	elif np.nanmax(timevec)<49:
		if np.nanmax(timevec)>24:
			timevec = timevec%24
		timevec = timevec*3600

	for eachminute in np.arange(0,1440):
		times.append(eachminute*60.0)

	slope_vec=[]
	times_vec=[]

	arr=np.array(timevec)
	for eachminute in times:
		idxarray   = (arr >= eachminute) & (arr < (eachminute+(60.0) ) ) # bool array
		data = TEC[idxarray]
		if len(data)>200:
			nperseg=np.nanmin([len(data),4000])
			f,pxx = welch(data-np.mean(data),fs=fs,nperseg=nperseg,window='hamming')
			mask = (f>=0.05) & (f<=20)
			f_mask = f[mask]
			pxx_mask = pxx[mask]
			slope,intercept = np.polyfit(np.log10(f_mask),np.log10(pxx_mask),1)
			slope_vec.append(slope)
			times_vec.append(eachminute)

			fig2, axs = pyplot.subplots(1, 2,figsize=(2, 2),dpi=300)
			axs[0].loglog(f_mask,pxx_mask,label='psd')
			axs[0].loglog(f_mask,10**(intercept)*f_mask**(slope),'--k',label='fit')
			axs[0].set_xlabel('Hz')
			axs[0].set_ylabel('PSD')
			axs[0].set_title('minute: %d '%(eachminute))
			axs[0].text(0.1,-0.9, "%.2f"%slope)

			axs[1].plot(arr[idxarray],data,'--ob')
			pyplot.savefig(r"/Volumes/scintpi3_data/JM_DOCUMENTS/windowsPC/oct11_2024/sc3/figs/n_min%d"%(eachminute),transparent=True)
			pyplot.close()
		else:
			slope_vec.append(-1)
			times_vec.append(eachminute)

		# fig2, axs = pyplot.subplots(1, 1,figsize=(2, 2),dpi=300)
		# axs.loglog(f_mask,pxx_mask,label='psd')
		# axs.loglog(f_mask,10**(intercept)*f_mask**(slope),'--k',label='fit')
		# axs.set_xlabel('Hz')
		# axs.set_ylabel('PSD')
		# axs.set_title('minute: %d '%(eachminute))
		# axs.text(0.1,0.9, "%.2f"%slope)
		# pyplot.savefig(r"C:\Users\JmGomezs\Downloads\oct11_2024\sc3\min%d"%(eachminute),transparent=True)
		# pyplot.close()
	if add86400:
		times_vec = np.array(times_vec) + np.ones((len(times_vec)))*86400
	return slope_vec,times_vec

def slope_SNR(SNR,timevec,fs):
	times=[]
	add86400=False
	if np.nanmax(timevec)>86400:
		timevec = timevec%86400
		add86400 = True
	elif np.nanmax(timevec)<49:
		if np.nanmax(timevec)>24:
			timevec = timevec%24
		timevec = timevec*3600

	for eachminute in np.arange(0,1440):
		times.append(eachminute*60.0)

	slope_vec=[]
	times_vec=[]

	arr=np.array(timevec)
	for eachminute in times:
		idxarray   = (arr >= eachminute) & (arr < (eachminute+(60.0) ) ) # bool array
		data = SNR[idxarray]
		if len(data)>200:
			nperseg=np.nanmin([len(data),4000])
			f,pxx = welch(data-np.mean(data),fs=fs,nperseg=nperseg,window='hamming')
			mask = (f>=0.05) & (f<=20)
			f_mask = f[mask]
			pxx_mask = pxx[mask]
			slope,intercept = np.polyfit(np.log10(f_mask),np.log10(pxx_mask),1)
			slope_vec.append(slope)
			times_vec.append(eachminute)
		else:
			slope_vec.append(-1)
			times_vec.append(eachminute)

		# fig2, axs = pyplot.subplots(1, 1,figsize=(2, 2),dpi=300)
		# axs.loglog(f_mask,pxx_mask,label='psd')
		# axs.loglog(f_mask,10**(intercept)*f_mask**(slope),'--k',label='fit')
		# axs.set_xlabel('Hz')
		# axs.set_ylabel('PSD')
		# axs.set_title('minute: %d '%(eachminute))
		# axs.text(0.1,0.9, "%.2f"%slope)
		# pyplot.savefig(r"C:\Users\JmGomezs\Downloads\oct11_2024\sc3\min%d"%(eachminute),transparent=True)
		# pyplot.close()
	if add86400:
		times_vec = np.array(times_vec) + np.ones((len(times_vec)))*86400
	return slope_vec,times_vec

def dTEC_std(dTEC,timevec):
	add86400=False
	if np.nanmax(timevec)>86400:
		timevec = timevec%86400
		add86400 = True

	sigmaPhiList=[]
	sigmaPhitime=[]
	times=[]
	frame =1
	for eachminute in range(0,1440):
		times.append(eachminute*60.0)
	arr=np.array(timevec)
	for eachminute in times:
		# print (eachminute,eachminute+60 )
		idxarray   = (arr >= eachminute) & (arr < (eachminute+(60.0) ) ) # bool array
		if len(timevec[idxarray])>0:
			# print ("len(idxarray):",len(timevec[idxarray]) )
			tmp_time = timevec[idxarray]
			phasedatafil = dTEC[idxarray]
			if len(timevec[idxarray])>50:
				sigmaPhi= np.nanstd(phasedatafil)
				sigmaPhiList.append(sigmaPhi)
			else :
				sigmaPhiList.append(float("nan"))
		else :
			sigmaPhiList.append(float("nan"))
		sigmaPhitime.append(eachminute+60.0)

	if add86400:
		sigmaPhitime = np.array(sigmaPhitime) + np.ones((len(sigmaPhitime)))*86400

	return sigmaPhiList,sigmaPhitime

# from matplotlib import pyplot
#Version 000
def butter_highpass(cutoff, fs, order=2):
	"""
	Design a highpass filter.

	Args:
		- cutoff (float) : the cutoff frequency of the filter.
		- fs     (float) : the sampling rate.
		- order    (int) : order of the filter, by default defined to 5.
	"""
	# calculate the Nyquist frequency
	nyq = 0.5 * fs

	# design filter
	high = cutoff / nyq
	b, a = butter(order, high, btype='high', analog=False)

	# returns the filter coefficients: numerator and denominator
	return b, a

def butter_highpass_filter(data, cutoff, fs, order=2):
	times = order//2
	b, a = butter_highpass(cutoff, fs, order=2)
	for i in range(0,times):
		# print(i)
		y = lfilter(b, a, data)
		data = y
	return y

def fastdownsampling_1sec(timevec,SNR):
	round_timevec = timevec.astype(int)
	unique_round_timevec,indexvector = np.unique(round_timevec, return_index=True)
	return unique_round_timevec,SNR[indexvector]

def fastdownsampling_1secSNR(timevec,SNR):
	round_timevec = timevec.astype(int)
	unique_round_timevec,indexvector = np.unique(round_timevec, return_index=True)
	return unique_round_timevec,SNR[indexvector]

def fastdownsampling_1sec_30secTEC(timevec,tec):
	round_timevec = timevec.astype(int)
	unique_round_timevec,indexvector = np.unique(round_timevec, return_index=True)
	return unique_round_timevec,tec[indexvector],unique_round_timevec[::30],tec[indexvector][::30]

def read_CLK(clk_path,clk_file,sat_string):
	Ts = []
	Xs = []
	Ys = []
	clk = []
	sat = []
	# clk_file = "IAC0MGXFIN_20241310000_01D_30S_CLK.CLK"
	# clk_path = r"C:\Users\JmGomezs\Downloads\super_geostorm"
	file = "%s\%s"%(clk_path,clk_file)
	with open(file) as fd:
		for n, line in enumerate(fd):
			if line[0:2]=='AS' :
				sat.append(line[3:6])
				Ts.append(int(line[19:21])*3600+int(line[22:25])*60+float(line[25:30]))
				clk.append(float(line[-40:-20].replace('D','e')) + float(line[-20:-1].replace('D','e') ) ) #

	sat = np.array(sat)
	Ts = np.array(Ts)
	clk = np.array(clk)
	gxx_idx = sat==sat_string
	c = 299792458
	return Ts[gxx_idx],clk[gxx_idx]*c

def getTEC_data(inputs,data):
	timevec0 = data['tow']
	gnssvec0 = data['gnssid']
	signvec0 = data['signalid']
	#TODO save data and load it again to avoid wait 6 minutes each time. lol
	idxarray = (data['gnssid'] == inputs[0])# helps alot with process time
	idxarray2 = (data['signalid'][idxarray] == inputs[1])# helps alot with process time
	idxarray3 = (data['signalid'][idxarray] == inputs[2])# helps alot with process time
	testime0  = timevec0[idxarray][idxarray2]
	testime0L5  = timevec0[idxarray][idxarray3]
	testpha0  = data['cphase'][idxarray][idxarray2]
	testpha20  = data['cphase'][idxarray][idxarray3]
	testsnr0  = data['snr'][idxarray][idxarray2]
	testprang0= data['prang'][idxarray][idxarray2]
	rphase10 = remove_cycleslips(testpha0,1000)
	rphase20 = remove_cycleslips(testpha20,2000)
	final_time0, L1_ind0, L2_ind0 = np.intersect1d(testime0, testime0L5, assume_unique=True, return_indices=True)#WOW super fast
	#So one value is added
	rphase10=np.hstack((rphase10,rphase10[-1]))
	rphase20=np.hstack((rphase20,rphase20[-1]))

	GNSSid=inputs[3]
	L1_wlen, L2_wlen = getwavelengths(GNSSid)
	L1_wlen, L2_wlen = getwavelengths(GNSSid)
	tec0 = ((-rphase20[L2_ind0])*L2_wlen+(rphase10[L1_ind0])*L1_wlen)*get_TEC_cte(GNSSid,inputs[4])
	tec0 = remove_spikes(tec0,0.125)
	tec0=np.hstack((tec0,tec0[-1]))

	return final_time0,tec0,rphase10[L1_ind0],rphase20[L2_ind0]


def get_phase_dopp_data(inputs,data):
	timevec0 = data['tow']
	gnssvec0 = data['gnssid']
	signvec0 = data['signalid']
	#TODO save data and load it again to avoid wait 6 minutes each time. lol
	idxarray = (data['gnssid'] == inputs[0])# helps alot with process time
	idxarray2 = (data['signalid'][idxarray] == inputs[1])# helps alot with process time
	idxarray3 = (data['signalid'][idxarray] == inputs[2])# helps alot with process time
	testime0  = timevec0[idxarray][idxarray2]
	testime0L5  = timevec0[idxarray][idxarray3]
	testpha0  = data['cphase'][idxarray][idxarray2]
	testpha20  = data['cphase'][idxarray][idxarray3]
	testsnr0  = data['snr'][idxarray][idxarray2]
	testprang0= data['prang'][idxarray][idxarray2]
	testdopp = data['dop'][idxarray][idxarray2]
	testdopp2 = data['dop'][idxarray][idxarray3]
	rphase10 = remove_cycleslips(testpha0,1000)
	rphase20 = remove_cycleslips(testpha20,2000)
	final_time0, L1_ind0, L2_ind0 = np.intersect1d(testime0, testime0L5, assume_unique=True, return_indices=True)#WOW super fast
	#So one value is added
	rphase10=np.hstack((rphase10,rphase10[-1]))
	rphase20=np.hstack((rphase20,rphase20[-1]))

	return final_time0,rphase10[L1_ind0],rphase20[L2_ind0],testdopp[L1_ind0],testdopp2[L2_ind0]

def SEP_read_raw10Hz(datafolder,doy):
	raw_data_files=[]
	raw_data_files = glob.glob("%s/cssm*%s*.24__Messages.txt"%(datafolder,doy))
	print (raw_data_files.sort())
	cols2read = [1,2,3,4,6,7,8,9]
	'''
	SBF Meas,TOW [s],WNc [w],SVID,SignalType,Antenna ID,PR [m],L [cyc],Doppler [Hz],C/N0 [dB-Hz],LockTime [s]
	---------------------------------------------------------------------------------------------------------
	SBF Meas,518400.000,2148,G06,GPS_L1CA,Main,23075129.28000,121260684.24625,-2140.76880,42.00000,14613
	'''
	start_time = time.time()
	data0 = np.loadtxt(open(raw_data_files[0],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data1 = np.loadtxt(open(raw_data_files[1],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data2 = np.loadtxt(open(raw_data_files[2],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data3 = np.loadtxt(open(raw_data_files[3],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	full_data = np.hstack((data0,data1,data2,data3))
	return full_data


def MOS_read_raw1Hz_wLockT(datafolder,doy):
	raw_data_files=[]
	raw_data_files = glob.glob("%s/cssm*%s*.24__Messages.txt"%(datafolder,doy))
	print (raw_data_files.sort())
	cols2read = [1,2,3,4,6,7,8,9]
	'''
	SBF Meas,TOW [s],WNc [w],SVID,SignalType,Antenna ID,PR [m],L [cyc],Doppler [Hz],C/N0 [dB-Hz],LockTime [s]
	---------------------------------------------------------------------------------------------------------
	SBF Meas,518400.000,2148,G06,GPS_L1CA,Main,23075129.28000,121260684.24625,-2140.76880,42.00000,14613
	'''
	start_time = time.time()
	data0 = np.loadtxt(open(raw_data_files[0],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data1 = np.loadtxt(open(raw_data_files[1],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data2 = np.loadtxt(open(raw_data_files[2],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data3 = np.loadtxt(open(raw_data_files[3],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	full_data = np.hstack((data0,data1,data2,data3))
	return full_data


def SEP_read_raw1Hz(datafolder,doy):
	raw_data_files=[]
	raw_data_files = glob.glob("%s/cssm*%s*.24__Messages.txt"%(datafolder,doy))
	print (raw_data_files.sort())
	cols2read = [1,2,3,4,6,7,8,9]
	'''
	SBF Meas,TOW [s],WNc [w],SVID,SignalType,Antenna ID,PR [m],L [cyc],Doppler [Hz],C/N0 [dB-Hz],LockTime [s]
	---------------------------------------------------------------------------------------------------------
	SBF Meas,518400.000,2148,G06,GPS_L1CA,Main,23075129.28000,121260684.24625,-2140.76880,42.00000,14613
	'''
	start_time = time.time()
	data0 = np.loadtxt(open(raw_data_files[0],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data1 = np.loadtxt(open(raw_data_files[1],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data2 = np.loadtxt(open(raw_data_files[2],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data3 = np.loadtxt(open(raw_data_files[3],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	full_data = np.hstack((data0,data1,data2,data3))
	return full_data

def TSEP_read_raw1Hz(datafolder,doy):
	raw_data_files=[]
	raw_data_files = glob.glob("%s/CSS*%s*.24__Messages.txt"%(datafolder,doy))
	print (raw_data_files.sort())
	cols2read = [1,2,3,4,6,7,8,9]
	'''CSS_1312.24__Messages
	SBF Meas,TOW [s],WNc [w],SVID,SignalType,Antenna ID,PR [m],L [cyc],Doppler [Hz],C/N0 [dB-Hz],LockTime [s]
	---------------------------------------------------------------------------------------------------------
	SBF Meas,518400.000,2148,G06,GPS_L1CA,Main,23075129.28000,121260684.24625,-2140.76880,42.00000,14613
	'''
	start_time = time.time()
	data0 = np.loadtxt(open(raw_data_files[0],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data1 = np.loadtxt(open(raw_data_files[1],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data2 = np.loadtxt(open(raw_data_files[2],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	data3 = np.loadtxt(open(raw_data_files[3],'rt').readlines(),delimiter=',',usecols=cols2read,converters={7: lambda s: float(s.strip() or 'Nan')} ,dtype={'names':('tow','week','gnssid','signalid','prang','cphase','dop','snr'),'formats': (np.float64, np.float64,'S5','S15',np.float64,np.float64,np.float64,np.float64)} )
	# print(time.time()-start_time)
	full_data = np.hstack((data0,data1,data2,data3))
	return full_data

def PRN(GNSSid,num):
	if GNSSid == '00':
		return num
	elif GNSSid == '01' and (num>=31) and (num<=35):
		return num+100 #same
	elif GNSSid == '02':
		return num+70
	elif GNSSid == '03':
		return num+140
	elif GNSSid == '06':
		if num>0 and num<25:
			return num+37
		elif num>24 and num<31:
			return num+38
	else:
		return num-999

def readingISMR_TOW(SAT,FILENAME):
	data = open(FILENAME,'r')
	data_lines= data.readline()
	data_lines= data.readlines()
	data.close()
	selected_sat=int(SAT)
	elev=[]
	azit=[]
	timerx5=[]
	timevec=[]
	s4=[]
	s42=[]
	phi=[]
	phi2=[]
	GPSfromUTC = (datetime(1980,1,6) - datetime(1970,1,1)).total_seconds()
	# print ('GPSfromUTC:',GPSfromUTC)
	for line in data_lines:
		split_data= line.split(',')
		len_split = len(split_data)
		if (int(split_data[2])==selected_sat):
			try:
				week=int(split_data[0])
				sec = int(split_data[1])
				timestamp= int(sec)
				elev.append(float(split_data[5]))
				azit.append(float(split_data[4]))
				s4.append(float(split_data[7]))
				s42.append(float(split_data[32]))
				phi.append(float(split_data[13]))
				phi2.append(float(split_data[38]))
				timevec.append(timestamp%86400)#return seconds of day
			except Exception as e:
				print ('e:',e)
	return timevec,azit,s4,s42,elev,phi2,phi

def interpolate_to_1sec(rangf,time):
	trained_rangf = scipy.interpolate.interp1d(time/24.0*86400.0,rangf,kind='quadratic')
	new_time = np.arange(np.nanmin(time/24.0*86400.0),np.nanmax(time/24.0*86400.0))
	new_rangf = trained_rangf(new_time) # any of these values should be nan
	return new_rangf,new_time


def interpolate_to_10Hz(rangf,time):
	trained_rangf = scipy.interpolate.interp1d(time/24.0*86400.0,rangf,kind='quadratic')
	new_time = np.arange(np.nanmin(time/24.0*86400.0),np.nanmax(time/24.0*86400.0),0.1)
	new_rangf = trained_rangf(new_time) # any of these values should be nan
	return new_rangf,new_time


def get_TEC_cte(conste,prn):
	if conste == '00':
		return 9.529101453519065
	elif conste == '01':
		return 7.771372607668402
	elif conste == '02':
		return 8.766203502852782
	elif conste == '03':
		return 9.002159172072213
	elif conste == '06':
		if prn == 14 or prn == 10 :   #k=-7
			return 9.71314456
		elif prn == 2 or prn == 6 :   #k=-4
			return 9.73366889
		elif prn == 18 or prn == 22 : #k=-3
			return 9.74051515
		elif prn == 13 or prn == 9 :  #k=-2
			return 9.74736382
		elif prn == 12 or prn == 16 : #k=-1
			return 9.75421489
		elif prn == 15 or prn == 11 : #k=0
			return 9.76106837
		elif prn == 5 or prn == 1 :   #k=+1
			return 9.76792425
		elif prn == 20 or prn == 24 : #k=+2
			return 9.77478255
		elif prn == 19 or prn == 23 : #k=+3
			return 9.78164325
		elif prn == 21 or prn == 17 : #k=+4
			return 9.78164325
		elif prn == 3 or prn == 7 :   #k=+5
			return 9.79537187
		elif prn == 4 or prn == 8 :   #k=+6
			return 9.80223979
		else:
			print ('Error with PRN')
			return 0
	else :
		print ('Error with Constellation')
		return 0

def remove_cycleslips(carrierphase,threshold):
	diff_phase = np.diff(carrierphase)
	righavg = leftavg = np.zeros((len(diff_phase)))
	leftavg[1: ]= diff_phase[:-1]
	righavg[:-1]= diff_phase[1:]
	avg=leftavg*0.5+righavg*0.5
	diff_phase_removed = np.where((-threshold<diff_phase) & (diff_phase<threshold),diff_phase,avg)
	diff_phase_removed=np.hstack((0,diff_phase_removed))
	return np.nancumsum(diff_phase_removed)

def getwavelengths(conste):
	#c = 299 792 458 m / s
	if conste == '00':
		return 0.19029367,0.24421021
	elif conste == '01':
		return 0.1902936728,0.2548280488 #This values work for SBAS L1,L5
	elif conste == '02':
		return 0.19029367,0.24834937 #This values work for GALILEO L1,L2 #0.25482805
	elif conste == '03':
		return 0.19203949,0.24834937 #0.19203949 , 0.24834937
	elif conste == '06':
		return 0.18713637,0.2406039
	else :
		print ('Error with Constellation')
		return 0,0

def get_IPP(rx_lat,rx_long,elevaData,azimtData,h_desired):

	#input in radians
	current_lat=rx_lat
	current_long=rx_long
	#h_desired in KM
	# print(current_lat,current_long)
	Re = np.ones((len(elevaData)))*6378.137
	h  = np.ones((len(elevaData)))*h_desired/1000.0
	wpp = np.ones((len(elevaData)))*np.pi/2.0 - elevaData - np.arcsin(((Re)/(Re+h))*np.cos(elevaData))

	ipplat  = np.arcsin(np.sin(current_lat/180.0*np.pi)*np.cos(wpp) + np.cos(current_lat/180.0*np.pi)*np.sin(wpp)*np.cos(azimtData) )
	ipplong = np.ones((len(elevaData)))*current_long/180.0*np.pi + np.arcsin(np.sin(wpp)*np.sin(azimtData)/np.cos(ipplat) )
	ipplat	= ipplat/np.pi*180.0
	ipplong = ipplong/np.pi*180.0

	return ipplat,ipplong

def dec_angle(latipp,longipp):
	'''
		INPUTS: latipp , longipp and heiipp: coordinates of the IPPs, satellite elevation and azimuth
		round lat and long using only one decimal ex: -7.39 deg lat to -7.4
	'''
	#1. Check if lat and long have the same lenghts
	#2. extract Bx, By and Bz at 350Km, be sure that all the latitudes are inside the region
	#TODO : load in cache memory to avoid open the file many times
	datafolder = r"C:\Users\JmGomezs\Documents\ScintPi_Scripts\analysis\pyIGRF\release"
	with open("%s/Bmag_2023.npy"%(datafolder), 'rb') as f:
		Bxmat =np.load(f)
		Bymat =np.load(f)
		Bzmat =np.load(f)
		lat_vecv=np.load(f)
		lon_vec =np.load(f)

	#TODO: restric the lat and long to the min and max of the Bmag file
	latipp  = np.where((latipp>-15)&(latipp<=-0.1),latipp,float("nan") )
	longipp = np.where((longipp>-43)&(longipp<=-28.1),longipp,float("nan") )
	#3. round vectors with 1 decimal replace, TODO: verify if 0.1 res is enought
	latipp = np.round(latipp,decimals=1)
	longipp = np.round(longipp,decimals=1)
	#4. be sure the height is 350Km
	# print("latipp,longipp:",latippf,longippf)
	#Clean values from nans

	zip_object = zip(latipp, longipp)
	Bx=[]
	By=[]
	Bz=[]
	for lat,long in zip_object:
		y_idx = np.where(lat_vecv == lat)
		x_idx = np.where(lon_vec == long)
		print('lat,long:',lat,long)
		if (np.isnan(lat)) or (np.isnan(long)):
			Bx.append(float('nan'))
			By.append(float('nan'))
			Bz.append(float('nan'))
		else:
			Bx.append(Bxmat[y_idx][0][x_idx][0])
			By.append(Bymat[y_idx][0][x_idx][0])
			Bz.append(Bzmat[y_idx][0][x_idx][0])

	Bx = np.array(Bx)
	By = np.array(By)
	Bz = np.array(Bz)

	arg2=By/Bx
	dec= np.arctan(arg2)

	return dec

def dip_angle(latipp,longipp):
	'''
		INPUTS: latipp , longipp and heiipp: coordinates of the IPPs, satellite elevation and azimuth
		round lat and long using only one decimal ex: -7.39 deg lat to -7.4
	'''
	#1. Check if lat and long have the same lenghts
	#2. extract Bx, By and Bz at 350Km, be sure that all the latitudes are inside the region
	#TODO : load in cache memory to avoid open the file many times
	datafolder = r"C:\Users\JmGomezs\Documents\ScintPi_Scripts\analysis\pyIGRF\release"
	with open("%s/Bmag_2023.npy"%(datafolder), 'rb') as f:
		Bxmat =np.load(f)
		Bymat =np.load(f)
		Bzmat =np.load(f)
		lat_vecv=np.load(f)
		lon_vec =np.load(f)
	#TODO: restric the lat and long to the min and max of the Bmag file
	latipp  = np.where((latipp>-15)&(latipp<=-0.1),latipp,float("nan") )
	longipp = np.where((longipp>-43)&(longipp<=-28.1),longipp,float("nan") )
	# print(len(latipp),len(longipp))
	latipp = np.round(latipp,decimals=1)
	longipp = np.round(longipp,decimals=1)
	#4. be sure the height is 350Km
	# print("latipp,longipp:",latippf,longippf)
	#Clean values from nans

	zip_object = zip(latipp, longipp)
	Bx=[]
	By=[]
	Bz=[]
	for lat,long in zip_object:
		print('lat,long:',lat,long)
		y_idx = np.where(lat_vecv == lat)
		x_idx = np.where(lon_vec == long)
		if (np.isnan(lat)) or (np.isnan(long)):
			Bx.append(float('nan'))
			By.append(float('nan'))
			Bz.append(float('nan'))
		else:
			Bx.append(Bxmat[y_idx][0][x_idx][0])
			By.append(Bymat[y_idx][0][x_idx][0])
			Bz.append(Bzmat[y_idx][0][x_idx][0])

	Bx = np.array(Bx)
	By = np.array(By)
	Bz = np.array(Bz)

	arg = Bz/np.sqrt(Bx**2.0 + By**2.0)
	Dip = np.arctan(arg)

	return Dip

def remove_spikes(data,threshold):
	diff_phase = np.diff(data)
	diff_phase_removed = np.where((-threshold<diff_phase) & (diff_phase<threshold),diff_phase,float("nan"))
	diff_phase_removed=np.hstack((0,diff_phase_removed))
	return np.nancumsum(diff_phase_removed)

def write_vscint_h5(h5filename,DIC_OUT,out_fields,header):
	gnssdic={0:'GPS',1:'SBS',2:'GAL',3:'BDS',6:'GLO'}
	gnsslist=['00','01','02','03','06']

	fileh5 = h5py.File(h5filename,'w')
	for GNSSid in gnsslist:
		group = fileh5.create_group("%s"%(header[GNSSid][0]))
		for eachsat in header[GNSSid][1]:
			key_to_test =  "%s_%03d_%s"%(GNSSid,eachsat,out_fields[0])
			if (key_to_test in DIC_OUT):
				rows=len(DIC_OUT["%s_%03d_%s"%(GNSSid,eachsat,out_fields[0])])
				if rows>0:
					sub_group = fileh5.create_group("/%s/SVID%03d"%(header[GNSSid][0],eachsat))
					for field in out_fields:
						datatype= type(DIC_OUT["%s_%03d_%s"%(GNSSid,eachsat,field)][0])
						veclengh= len(DIC_OUT["%s_%03d_%s"%(GNSSid,eachsat,field)])
						dataset = sub_group.create_dataset("%s"%(field), (1,veclengh), dtype =datatype)
						dataset[...] = DIC_OUT["%s_%03d_%s"%(GNSSid,eachsat,field)]
	fileh5.close()
	return 1
def read_scintpi_h5(filename):
	dic={}
	gnsslist=['00','01','02','03','06']
	gpslist=[]
	gallist=[]
	bdslist=[]
	sbslist=[]
	glolist=[]
	gnssdic={'00':['GPS',gpslist],'01':['SBS',sbslist],'02':['GAL',gallist],'03':['BDS',bdslist],'06':['GLO',glolist]}
	maxsats=0
	h5file = h5py.File(filename,'r+')
	for conste in h5file.keys():
		if conste == 'GPS':
			gnssid = '00'
		elif conste == 'SBS':
			gnssid = '01'
		elif conste == 'GAL':
			gnssid = '02'
		elif conste == 'BDS':
			gnssid = '03'
		elif conste == 'GLO':
			gnssid = '06'

		groups = h5file.get(conste)
		for member in groups.items():
			maxsats = maxsats + 1
			svid = member[0].replace('SVID','')
			gnssdic[gnssid][1].append(int(svid))
			for each_param in groups.get(member[0]).keys():
				dic["%s_%03d_%s"%(gnssid,int(svid),each_param)] = groups.get(member[0]).get(each_param)[0]
	h5file.close()

	return dic,gnssdic

def cross_corr_amp(rx1_powerData,rx1_timevec,rx2_powerData,rx2_timevec,cross_corr_space,seconds_to_correlate,sampling,vec_per):
	"""
	https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where
	INPUTS :-rx1_powerData : Receivers Amplitude Signals in dB
			-rx2_powerData : Receivers Amplitude Signals in dB
			-rx1_timevec	: Timestamps in Seconds of the day between 0 to 86399.
			-rx2_timevec	: Timestamps in Seconds of the day between 0 to 86399.
	"""
	rx1_powerData	=np.array(rx1_powerData)#+np.ones([len(timevec)])*(18.0/3600.0) # #SEPTENTRIO USES DATA from 1 MINUTE GPS TIME
	rx1_timevec  	=np.array(rx1_timevec)
	rx2_powerData	=np.array(rx2_powerData)#+np.ones([len(timevec)])*(18.0/3600.0) # #SEPTENTRIO USES DATA from 1 MINUTE GPS TIME
	rx2_timevec  	=np.array(rx2_timevec)


	'''Defining mid points in time, cross correlations will be performed around the midpoints +/- seconds_to_correlate '''
	midpoints = np.arange(0,86399,cross_corr_space)
	'''Defining outputs'''
	rhos		= np.zeros((len(midpoints)),dtype=np.float32)
	offset		= np.zeros((len(midpoints)),dtype=np.float32)
	rx1_s4_vec	= np.zeros((len(midpoints)),dtype=np.float32)
	rx2_s4_vec	= np.zeros((len(midpoints)),dtype=np.float32)
	rx1_tau_vec	= np.zeros((len(midpoints)),dtype=np.float32)
	rx2_tau_vec	= np.zeros((len(midpoints)),dtype=np.float32)
	output_idx = 0
	print("# WARNING:  full mode correlation")
	for eachmidpoint in midpoints:
		rx1_idx_vec  = (rx1_timevec >= (eachmidpoint-(seconds_to_correlate//2)) ) & (rx1_timevec < (eachmidpoint+(seconds_to_correlate//2)) )
		rx2_idx_vec  = (rx2_timevec >= (eachmidpoint-(seconds_to_correlate//2)) ) & (rx2_timevec < (eachmidpoint+(seconds_to_correlate//2)) )
		rx1_tmp_ampdB = rx1_powerData[rx1_idx_vec]
		rx2_tmp_ampdB = rx2_powerData[rx2_idx_vec]


		#TODO : if all the vector is nan, dont process
		samples_threshold = seconds_to_correlate*sampling*(vec_per/100.0)
		allnans1 = len(rx1_tmp_ampdB[~np.isnan(rx1_tmp_ampdB)])
		allnans2 = len(rx2_tmp_ampdB[~np.isnan(rx2_tmp_ampdB)])
		if (len(rx1_tmp_ampdB) < samples_threshold)  or (len(rx2_tmp_ampdB) < samples_threshold) or (allnans1==0) or (allnans2==0):
			rhos[output_idx]=float("nan")
			offset[output_idx]=float("nan")
			rx1_s4_vec[output_idx]=float("nan")
			rx2_s4_vec[output_idx]=float("nan")
			rx1_tau_vec[output_idx]=float("nan")
			rx2_tau_vec[output_idx] = float("nan")
			output_idx=output_idx+1
			continue
		else:
			# if np.abs(diff)>=1:
				# rx1_tmp_ampdB = rx1_tmp_ampdB + np.ones((len(rx1_tmp_ampdB)))*diff #Normalize the mean to do an easy correlation
			# diff=np.nanmean(rx1_tmp_ampdB)-np.nanmean(rx2_tmp_ampdB)
			rx1_tmp_amp_lin=np.power(10,np.array(rx1_tmp_ampdB)/10.0)
			rx2_tmp_amp_lin=np.power(10,np.array(rx2_tmp_ampdB)/10.0)
			rx1_s4 = np.nanstd(rx1_tmp_amp_lin,ddof=1) / np.nanmean(rx1_tmp_amp_lin)
			rx2_s4 = np.nanstd(rx2_tmp_amp_lin,ddof=1) / np.nanmean(rx2_tmp_amp_lin)

			right_time = np.round(np.arange((eachmidpoint-(seconds_to_correlate//2)),(eachmidpoint+(seconds_to_correlate//2)),1/sampling),decimals=2)

			rx1_tmp_timevec = rx1_timevec[rx1_idx_vec]
			rx2_tmp_timevec = rx2_timevec[rx2_idx_vec]

			right_time_rx1 = np.where(right_time	>np.nanmin(rx1_tmp_timevec),right_time,float("nan"))
			right_time_rx1 = np.where(right_time_rx1<np.nanmax(rx1_tmp_timevec),right_time_rx1,float("nan"))
			right_time_rx2 = np.where(right_time	>np.nanmin(rx2_tmp_timevec),right_time,float("nan"))
			right_time_rx2 = np.where(right_time_rx2<np.nanmax(rx2_tmp_timevec),right_time_rx2,float("nan"))

			rx1_amp_dB_intf = scipy.interpolate.interp1d(rx1_tmp_timevec,rx1_tmp_ampdB,kind='linear')
			rx2_amp_dB_intf = scipy.interpolate.interp1d(rx2_tmp_timevec,rx2_tmp_ampdB,kind='linear')

			rx1_amp_dB_int = rx1_amp_dB_intf(right_time_rx1) # any of these values should be nan
			rx2_amp_dB_int = rx2_amp_dB_intf(right_time_rx2) # complete the rest of the values with 0?

			rx1_amp_lin_int = np.power(10,np.array(rx1_amp_dB_int)/10.0)
			rx2_amp_lin_int = np.power(10,np.array(rx2_amp_dB_int)/10.0)

			rx12_tmp_timevec, rx1_idx, rx2_idx = np.intersect1d(right_time_rx1, right_time_rx2, assume_unique=True, return_indices=True) #Only interpolate same number of samples per rx

			joint_mean = np.nanmean(rx1_amp_lin_int[rx1_idx])*0.5 + np.nanmean(rx2_amp_lin_int[rx2_idx])*0.5
			#This is trying to send everything around 0
			rx1_amp_lin_int[rx1_idx] = rx1_amp_lin_int[rx1_idx] - np.ones(len(rx1_amp_lin_int[rx1_idx]))*joint_mean
			rx2_amp_lin_int[rx2_idx] = rx2_amp_lin_int[rx2_idx] - np.ones(len(rx2_amp_lin_int[rx2_idx]))*joint_mean


			norm = np.linalg.norm
			corr = correlate(rx1_amp_lin_int[rx1_idx], rx2_amp_lin_int[rx2_idx],mode='full')
			mid = len(corr)/2.0
			rhos[output_idx]		= np.nanmax(corr/(norm(rx1_amp_lin_int[rx1_idx])*norm(rx2_amp_lin_int[rx2_idx])))
			offset[output_idx]		= np.argmax(corr)-mid
			# corr2 = correlate(rx1_amp_lin_int[rx1_idx], np.roll(rx2_amp_lin_int[rx2_idx],np.round(offset[output_idx],decimals=0)))
			# mid2 = len(corr2)/2.0
			rx1_s4_vec[output_idx]	=rx1_s4
			rx2_s4_vec[output_idx]	=rx2_s4

			if (rx1_s4_vec[output_idx]>=0.2) and (rx2_s4_vec[output_idx]>=0.2):

				midpoint  = len(rx1_amp_lin_int[rx1_idx])
				midpoint2 = len(rx2_amp_lin_int[rx2_idx])
				uu = correlate(rx1_amp_lin_int[rx1_idx], rx1_amp_lin_int[rx1_idx])
				uu2= correlate(rx2_amp_lin_int[rx2_idx], rx2_amp_lin_int[rx2_idx])
				var = np.nanvar(rx1_amp_lin_int[rx1_idx])
				var2= np.nanvar(rx2_amp_lin_int[rx2_idx])
				uunorm = uu/var/midpoint
				uunorm2= uu2/var2/midpoint2
				uumax = np.nanmax(uunorm)
				uumax2 = np.nanmax(uunorm2)
				samples =0
				samples2=0
				db3 = False
				db32= False
				for eachvalue in uunorm[midpoint:]:
					# print(eachvalue,uumax-3)
					if eachvalue<=(uumax-0.5*uumax):
						db3 =True
						break
					else:
						samples=samples+1

				for eachvalue in uunorm2[midpoint2:]:
					# print(eachvalue,uumax-3)
					if eachvalue<=(uumax2-0.5*uumax2):
						db32 =True
						break
					else:
						samples2=samples2+1

				#what happens if the -3db never reachs?
				if (samples< (midpoint-1)) & db3:
					rx1_tau_vec[output_idx]=samples*(1.0/20.0)
				if (samples2<(midpoint-1)) & db32:
					rx2_tau_vec[output_idx]=samples2*(1.0/20.0)
					# print("entras step3")
					# fig2 = pyplot.figure()
					# axe = fig2.add_subplot(111)
					# axe.plot(uulog,'--ok')
					# axe.plot([len(uulog)//2,len(uulog)//2],[0,np.nanmin(uulog)],'--r')
					# axe.plot([len(uunorm)//2+samples,len(uunorm)//2+samples],[0,np.nanmin(uulog)],'-r')
					# axe.set_title("%04d - S4: %f tau: %2.2f"%(midpoint,rx1_s4,samples )  )
					# axe.legend()
					# pyplot.show()
				else:
					rx1_tau_vec[output_idx] = float("nan")
					rx2_tau_vec[output_idx] = float("nan")
			else:
				rx1_tau_vec[output_idx] = float("nan")
				rx2_tau_vec[output_idx] = float("nan")

			output_idx=output_idx+1

	return midpoints,rx1_s4_vec,rx2_s4_vec,rhos,offset,rx1_tau_vec,rx2_tau_vec

def decorrelation_time_50Hz(rx1_powerData,rx1_timevec,seconds_to_correlate,sampling,vec_per):
	"""
	https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where
	INPUTS :-rx1_powerData : Receivers Amplitude Signals in dB
			-rx1_timevec	: Timestamps in Seconds of the day between 0 to 86399.
	"""
	rx1_powerData	=np.array(rx1_powerData)#+np.ones([len(timevec)])*(18.0/3600.0) # #SEPTENTRIO USES DATA from 1 MINUTE GPS TIME
	rx1_timevec  	=np.array(rx1_timevec)

	'''Defining mid points in time, cross correlations will be performed around the midpoints +/- seconds_to_correlate '''
	midpoints = np.arange(0,86399,60)#EVERY ,INUTE
	'''Defining outputs'''
	rhos		= np.zeros((len(midpoints)),dtype=np.float32)
	offset		= np.zeros((len(midpoints)),dtype=np.float32)
	rx1_s4_vec	= np.zeros((len(midpoints)),dtype=np.float32)
	rx1_tau_vec	= np.zeros((len(midpoints)),dtype=np.float32)
	output_idx = 0
	for eachmidpoint in midpoints:
		rx1_idx_vec  = (rx1_timevec >= (eachmidpoint-(seconds_to_correlate//2)) ) & (rx1_timevec < (eachmidpoint+(seconds_to_correlate//2)) )
		rx1_tmp_ampdB = rx1_powerData[rx1_idx_vec]
		#TODO : if all the vector is nan, dont process
		samples_threshold = seconds_to_correlate*sampling*(vec_per/100.0)
		if (len(rx1_tmp_ampdB) < samples_threshold):
			rhos[output_idx]=float("nan")
			offset[output_idx]=float("nan")
			rx1_s4_vec[output_idx]=float("nan")
			rx1_tau_vec[output_idx]=float("nan")
			output_idx=output_idx+1
			continue
		else:
			# if np.abs(diff)>=1:
				# rx1_tmp_ampdB = rx1_tmp_ampdB + np.ones((len(rx1_tmp_ampdB)))*diff #Normalize the mean to do an easy correlation
			# diff=np.nanmean(rx1_tmp_ampdB)-np.nanmean(rx2_tmp_ampdB)
			rx1_tmp_amp_lin=np.power(10,np.array(rx1_tmp_ampdB)/10.0)
			rx1_s4 = np.nanstd(rx1_tmp_amp_lin,ddof=1) / np.nanmean(rx1_tmp_amp_lin)

			right_time = np.arange((eachmidpoint-(seconds_to_correlate//2)),(eachmidpoint+(seconds_to_correlate//2)),1/sampling)

			rx1_tmp_timevec = rx1_timevec[rx1_idx_vec]

			right_time_rx1 = np.where(right_time	>np.nanmin(rx1_tmp_timevec),right_time,float("nan"))
			right_time_rx1 = np.where(right_time_rx1<np.nanmax(rx1_tmp_timevec),right_time_rx1,float("nan"))

			rx1_amp_dB_intf = scipy.interpolate.interp1d(rx1_tmp_timevec,rx1_tmp_ampdB,kind='linear')
			rx1_amp_dB_int = rx1_amp_dB_intf(right_time_rx1) # any of these values should be nan
			rx1_amp_lin_int = np.power(10,np.array(rx1_amp_dB_int)/10.0)

			# corr2 = correlate(rx1_amp_lin_int[rx1_idx], np.roll(rx2_amp_lin_int[rx2_idx],np.round(offset[output_idx],decimals=0)))
			# mid2 = len(corr2)/2.0
			rx1_s4_vec[output_idx]	=rx1_s4

			if (rx1_s4_vec[output_idx]>=0.05):
				midpoint  = len(rx1_amp_lin_int)
				uu = correlate(rx1_amp_lin_int[~np.isnan(rx1_amp_lin_int)], rx1_amp_lin_int[~np.isnan(rx1_amp_lin_int)])
				# uu2= correlate(rx2_amp_lin_int[rx2_idx], rx2_amp_lin_int[rx2_idx])
				var = np.nanvar(rx1_amp_lin_int[~np.isnan(rx1_amp_lin_int)])
				# var2= np.nanvar(rx2_amp_lin_int[rx2_idx])
				uunorm = uu/var/midpoint
				# uunorm2= uu2/var2/midpoint2
				uumax = np.nanmax(uunorm)
				# uumax2 = np.nanmax(uunorm2)
				samples =0
				# samples2=0
				db3 = False
				# db32= False
				for eachvalue in uunorm[midpoint:]:
					# print(eachvalue,uumax-3)
					if eachvalue<=(uumax-0.5*uumax):
						db3 =True
						break
					else:
						samples=samples+1

				# for eachvalue in uunorm2[midpoint2:]:
				# 	# print(eachvalue,uumax-3)
				# 	if eachvalue<=(uumax2-0.5*uumax2):
				# 		db32 =True
				# 		break
				# 	else:
				# 		samples2=samples2+1

				#what happens if the -3db never reachs?
				if (samples< (midpoint-1)) & db3:
					rx1_tau_vec[output_idx]=samples*(1.0/50.0)
					#
					# # print("entras step3")
					# fig2 = pyplot.figure()
					# axe = fig2.add_subplot(211)
					# axe.plot(uunorm,'--k',ms=0.5)
					# axe.plot([len(uunorm)//2,len(uunorm)//2],[0,np.nanmax(uunorm)],'--r')
					# axe.plot([len(uunorm)//2+samples,len(uunorm)//2+samples],[0,np.nanmax(uunorm)],'-r')
					# axe.set_title("%04d - S4: %f tau: %2.2f"%(midpoint,rx1_s4,samples )  )
					# # axe.legend()
					# axe = fig2.add_subplot(212)
					# axe.plot(rx1_amp_lin_int[~np.isnan(rx1_amp_lin_int)])
					# pyplot.savefig(r"C:\Users\JmGomezs\Box Sync\2023_Josemaria_TXScintillation\Data\SC3_utd\%05d.png"%eachmidpoint)
					# pyplot.close()
				else:
					rx1_tau_vec[output_idx] = float("nan")
			else:
				rx1_tau_vec[output_idx] = float("nan")

			output_idx=output_idx+1

	return midpoints,rx1_s4_vec,rx1_tau_vec

def decorrelation_time(rx1_powerData,rx1_timevec,seconds_to_correlate,sampling,vec_per):
	"""
	https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where
	INPUTS :-rx1_powerData : Receivers Amplitude Signals in dB
			-rx1_timevec	: Timestamps in Seconds of the day between 0 to 86399.
	"""
	rx1_powerData	=np.array(rx1_powerData)#+np.ones([len(timevec)])*(18.0/3600.0) # #SEPTENTRIO USES DATA from 1 MINUTE GPS TIME
	rx1_timevec  	=np.array(rx1_timevec)

	'''Defining mid points in time, cross correlations will be performed around the midpoints +/- seconds_to_correlate '''
	midpoints = np.arange(0,86399,60)#EVERY ,INUTE
	'''Defining outputs'''
	rhos		= np.zeros((len(midpoints)),dtype=np.float32)
	offset		= np.zeros((len(midpoints)),dtype=np.float32)
	rx1_s4_vec	= np.zeros((len(midpoints)),dtype=np.float32)
	rx1_tau_vec	= np.zeros((len(midpoints)),dtype=np.float32)
	output_idx = 0
	for eachmidpoint in midpoints:
		rx1_idx_vec  = (rx1_timevec >= (eachmidpoint-(seconds_to_correlate//2)) ) & (rx1_timevec < (eachmidpoint+(seconds_to_correlate//2)) )
		rx1_tmp_ampdB = rx1_powerData[rx1_idx_vec]
		#TODO : if all the vector is nan, dont process
		samples_threshold = seconds_to_correlate*sampling*(vec_per/100.0)
		allnans1 = len(rx1_tmp_ampdB[~np.isnan(rx1_tmp_ampdB)])
		if (len(rx1_tmp_ampdB) < samples_threshold)  or (allnans1==0) :
			rhos[output_idx]=float("nan")
			offset[output_idx]=float("nan")
			rx1_s4_vec[output_idx]=float("nan")
			rx1_tau_vec[output_idx]=float("nan")
			output_idx=output_idx+1
			continue
		else:
			# if np.abs(diff)>=1:
				# rx1_tmp_ampdB = rx1_tmp_ampdB + np.ones((len(rx1_tmp_ampdB)))*diff #Normalize the mean to do an easy correlation
			# diff=np.nanmean(rx1_tmp_ampdB)-np.nanmean(rx2_tmp_ampdB)
			rx1_tmp_amp_lin=np.power(10,np.array(rx1_tmp_ampdB)/10.0)
			rx1_s4 = np.nanstd(rx1_tmp_amp_lin,ddof=1) / np.nanmean(rx1_tmp_amp_lin)

			right_time = np.arange((eachmidpoint-(seconds_to_correlate//2)),(eachmidpoint+(seconds_to_correlate//2)),1/sampling)

			rx1_tmp_timevec = rx1_timevec[rx1_idx_vec]

			right_time_rx1 = np.where(right_time	>np.nanmin(rx1_tmp_timevec),right_time,float("nan"))
			right_time_rx1 = np.where(right_time_rx1<np.nanmax(rx1_tmp_timevec),right_time_rx1,float("nan"))

			rx1_amp_dB_intf = scipy.interpolate.interp1d(rx1_tmp_timevec,rx1_tmp_ampdB,kind='linear')
			rx1_amp_dB_int = rx1_amp_dB_intf(right_time_rx1) # any of these values should be nan
			rx1_amp_lin_int = np.power(10,np.array(rx1_amp_dB_int)/10.0)

			# corr2 = correlate(rx1_amp_lin_int[rx1_idx], np.roll(rx2_amp_lin_int[rx2_idx],np.round(offset[output_idx],decimals=0)))
			# mid2 = len(corr2)/2.0
			rx1_s4_vec[output_idx]	=rx1_s4

			if (rx1_s4_vec[output_idx]>=0.20):
				midpoint  = len(rx1_amp_lin_int)
				uu = correlate(rx1_amp_lin_int[~np.isnan(rx1_amp_lin_int)], rx1_amp_lin_int[~np.isnan(rx1_amp_lin_int)])
				# uu2= correlate(rx2_amp_lin_int[rx2_idx], rx2_amp_lin_int[rx2_idx])
				var = np.nanvar(rx1_amp_lin_int[~np.isnan(rx1_amp_lin_int)])
				# var2= np.nanvar(rx2_amp_lin_int[rx2_idx])
				uunorm = uu/var/midpoint
				# uunorm2= uu2/var2/midpoint2
				uumax = np.nanmax(uunorm)
				# uumax2 = np.nanmax(uunorm2)
				samples =0
				# samples2=0
				db3 = False
				# db32= False
				for eachvalue in uunorm[midpoint:]:
					# print(eachvalue,uumax-3)
					if eachvalue<=(uumax-0.5*uumax):
						db3 =True
						break
					else:
						samples=samples+1

				# for eachvalue in uunorm2[midpoint2:]:
				# 	# print(eachvalue,uumax-3)
				# 	if eachvalue<=(uumax2-0.5*uumax2):
				# 		db32 =True
				# 		break
				# 	else:
				# 		samples2=samples2+1

				#what happens if the -3db never reachs?
				if (samples< (midpoint-1)) & db3:
					rx1_tau_vec[output_idx]=samples*(1.0/20.0)
					#
					# # print("entras step3")
					# fig2 = pyplot.figure()
					# axe = fig2.add_subplot(211)
					# axe.plot(uunorm,'--k',ms=0.5)
					# axe.plot([len(uunorm)//2,len(uunorm)//2],[0,np.nanmax(uunorm)],'--r')
					# axe.plot([len(uunorm)//2+samples,len(uunorm)//2+samples],[0,np.nanmax(uunorm)],'-r')
					# axe.set_title("%04d - S4: %f tau: %2.2f"%(midpoint,rx1_s4,samples )  )
					# # axe.legend()
					# axe = fig2.add_subplot(212)
					# axe.plot(rx1_amp_lin_int[~np.isnan(rx1_amp_lin_int)])
					# pyplot.savefig(r"C:\Users\JmGomezs\Box Sync\2023_Josemaria_TXScintillation\Data\SC3_utd\%05d.png"%eachmidpoint)
					# pyplot.close()
				else:
					rx1_tau_vec[output_idx] = float("nan")
			else:
				rx1_tau_vec[output_idx] = float("nan")

			output_idx=output_idx+1

	return midpoints,rx1_s4_vec,rx1_tau_vec
def s4_cross_rdic(powerDatadBe,timevece,powerDatadBw,timevecw):
	s4_times=[]
	n_lags=[]
	s4_valuese=[]
	s4_samplese=[]
	cr_samplese=[]
	s4_valuesw=[]
	s4_samplesw=[]
	cr_samplesw=[]
	crossamps = []
	tau_valuee=[]
	tau_valuew=[]
	t0_valuew=[]
	t0_valuee=[]
	minNsamples = 600
	arre=np.round(np.array(timevece),decimals=3)
	arrw=np.round(np.array(timevecw),decimals=3)

#     for eachminute in range(0,1440*6):#every 10 seconds
#         s4_times.append(eachminute*10.0)

	for eachminute in range(0,1440):#every minute
		s4_times.append(eachminute*60.0)

	for eachminute in s4_times:
		idxarraye   = (arre >= eachminute) & (arre < (eachminute+(60.0)) )# bool array
		idxarrayw   = (arrw >= eachminute) & (arrw < (eachminute+(60.0)) )# bool array

		amp_dBe = powerDatadBe[idxarraye]
		amp_lne = np.power(10,np.array(amp_dBe)/10.0).astype(int)
		amp_nue = len(amp_lne)
		if amp_nue>minNsamples:
			s4e=np.nanstd(amp_lne,ddof=1)/np.nanmean(amp_lne)
			s4_valuese.append(s4e)
		else:
			s4_valuese.append(np.NaN)
		s4_samplese.append(amp_nue)


		amp_dBw = powerDatadBw[idxarrayw]
		amp_lnw = np.power(10,np.array(amp_dBw)/10.0).astype(int)
		amp_nuw = len(amp_lnw)
		if amp_nuw>minNsamples:
			s4w=np.nanstd(amp_lnw,ddof=1)/np.nanmean(amp_lnw)
			s4_valuesw.append(s4w)
		else:
			s4_valuesw.append(np.NaN)
		s4_samplesw.append(amp_nuw)

		if (amp_nuw>=minNsamples) and (amp_nue>=minNsamples) and (s4e>=0.05) and (s4w>=0.05):
#             n_lags.append(28)
			#this compesates the cases where one vector is shorter than the another
			mintime_e = np.nanmin(arre[idxarraye])
			mintime_w = np.nanmin(arrw[idxarrayw])
			mintime_t = np.nanmax([mintime_e,mintime_w])
			#get the shortest time in the other side
			maxtime_e = np.nanmax(arre[idxarraye])
			maxtime_w = np.nanmax(arrw[idxarrayw])
			maxtime_t = np.nanmin([maxtime_e,maxtime_w])

			#commom times
			tarre  = arre[idxarraye]
			tarrw  = arrw[idxarrayw]
			#We reduce the vectors to equal time intervals
			ctime_e = np.where( (tarre>=mintime_t) & (tarre<=maxtime_t),tarre,float("nan") )
			ctime_w = np.where( (tarrw>=mintime_t) & (tarrw<=maxtime_t),tarrw,float("nan") )
			itime_t = np.round(np.arange(eachminute,eachminute+60,0.02),decimals=2)
			#This corrects the time values that have offset ( seems to be due to the clock's jitter)
			itime_t = np.where( (itime_t>=mintime_t) & (itime_t<=maxtime_t),itime_t,float("nan") )
			#we are not going to interpolate values
			if_e = scipy.interpolate.interp1d(tarre,amp_dBe,kind='nearest')#train with the originals time vectors
			if_w = scipy.interpolate.interp1d(tarrw,amp_dBw,kind='nearest')#train with the originals time vectors
			idB_e = if_e(itime_t)
			idB_w = if_w(itime_t)

			# Just to avoid interpolation over the out-of-lock regions
			dtarre = np.diff(ctime_e)
			dtarrw = np.diff(ctime_w)
			#delete points of data that are not recorded in commom, this reduces coherence
			ngapse=len(dtarre[dtarre>=1]) # Number of time jumps over 1.0 sec
			ngapsw=len(dtarrw[dtarrw>=1]) # Number of time jumps over 1.0 sec

			boundse= []
			temp_dtarre = dtarre
			for idx in range(0,ngapse):
				maxidx = np.argmax(temp_dtarre)
				boundse.append([maxidx,maxidx+1])
				temp_dtarre[maxidx] = np.NaN
				pair = [maxidx,maxidx+1]
#                 cctime_e = np.where( (ctime_e>=boundse[pair[0]]) & ((ctime_e<=boundse[pair[1]])),np.NaN,ctime_e)
				itime_t  = np.where( (itime_t>=ctime_e[pair[0]]) & ((itime_t<=ctime_e[pair[1]])),float("nan"),itime_t)

			boundsw= []
			temp_dtarrw = dtarrw
			for idx in range(0,ngapsw):
				maxidx = np.argmax(temp_dtarrw)
				boundsw.append([maxidx,maxidx+1])
				temp_dtarrw[maxidx] = np.NaN
				pair = [maxidx,maxidx+1]
#                 cctime_w = np.where( (ctime_w>=boundsw[pair[0]]) & ((ctime_w<=boundsw[pair[1]])),np.NaN,ctime_w)
				itime_t  = np.where( (itime_t>=ctime_w[pair[0]]) & ((itime_t<=ctime_w[pair[1]])),float("nan"),itime_t)

			z_idB_e = idB_e - np.nanmean(idB_e)
			z_idB_w = idB_w - np.nanmean(idB_w)
			nanmask = ~np.isnan(itime_t)
			nz_idB_e   = z_idB_e[nanmask]
			nz_idB_w   = z_idB_w[nanmask]
			nz_itime_t = itime_t[nanmask]

			#First cross-correlation
			#this correlation will give you the mid-point where to start to count,
			#we do this because no all the vectors have 1200 samples in a minute
			corr = correlate(nz_idB_w,nz_idB_e)/(norm(nz_idB_w)*norm(nz_idB_e))
			maxidx = np.argmax(corr)
			# second cross-correlation with shifted lag delay
			# this compesates for the distintic vector lengths
			yff = np.roll(nz_idB_w,-maxidx)
			corr2 = correlate(nz_idB_e,yff)/(norm(nz_idB_e)*norm(yff))
			maxidx2 = np.argmax(corr2)

			#third cross-correlation
			# only worth it if coherence is > 0.5
			drops = maxidx-maxidx2-1
			if (np.nanmax(corr2) > 0.5) and (s4e>=0.25) and (s4w>=0.25) and (drops!=0):
				corr3 = correlate(nz_idB_e[:-drops],yff[:-drops])/(norm(nz_idB_e[:-drops])*norm(yff[:-drops]))
			else:
				corr3 = corr2

			crossamps.append(np.nanmax(corr3))
			n_lags.append(maxidx-maxidx2-1)


			acorr= np.correlate(nz_idB_e,nz_idB_e,mode='full')
			idxmax = np.argmax(acorr)
			valmax = np.max(acorr)
			ccorrmax = np.nanmax(corr3)
			samples=0
			db3 = False
			uff= acorr[idxmax:]
			for eachvalue in uff:
				if eachvalue<=(valmax*0.5):
					db3 =True
					break
				else:
					samples=samples+1
			tau_valuee.append(samples)

			samples=0
			db3 = False
			uffn = uff/valmax
			for eachvalue in uffn:
				if eachvalue<=(ccorrmax):
					db3 =True
					break
				else:
					samples=samples+1
			t0_valuee.append(samples)


			acorr= np.correlate(nz_idB_w,nz_idB_w,mode='full')
			idxmax = np.argmax(acorr)
			valmax = np.max(acorr)
			samples=0
			db3 = False
			uff= acorr[idxmax:]
			for eachvalue in uff:
				if eachvalue<=(valmax*0.5):
					db3 =True
					break
				else:
					samples=samples+1
			tau_valuew.append(samples)

			samples=0
			db3 = False
			uffn = uff/valmax
			for eachvalue in uffn:
				if eachvalue<=(ccorrmax):
					db3 =True
					break
				else:
					samples=samples+1
			t0_valuew.append(samples)
			cr_samplese.append(len(ctime_e))
			cr_samplesw.append(len(ctime_w))

		else:
			n_lags.append(float("nan"))
			cr_samplese.append(float("nan"))
			cr_samplesw.append(float("nan"))
			crossamps.append(float("nan"))
			tau_valuee.append(float("nan"))
			tau_valuew.append(float("nan"))
			t0_valuew.append(float("nan"))
			t0_valuee.append(float("nan"))
		#get the minimun time in the 1 min vector values

		returndic = {}
		returndic["utsec"] = s4_times
		returndic["s4rx1"] = s4_valuese
		returndic["s4rx2"] = s4_valuesw
		returndic["nsrx1"] = s4_samplese
		returndic["nsrx2"] = s4_samplesw
		returndic["nlags"] = n_lags
		returndic["norcc"] = n_lags
		returndic["taur1"] = tau_valuee
		returndic["taur2"] = tau_valuew
		returndic["ts0r1"] = t0_valuee
		returndic["ts0r2"] = t0_valuew
		#s4_times,s4_valuese,s4_samplese,s4_valuesw,s4_samplesw,n_lags,cr_samplese,cr_samplesw,crossamps,tau_valuee,tau_valuew,t0_valuee,t0_valuew
	return returndic

def s4_cross(powerDatadBe,timevece,powerDatadBw,timevecw):
	s4_times=[]
	n_lags=[]
	s4_valuese=[]
	s4_samplese=[]
	cr_samplese=[]
	s4_valuesw=[]
	s4_samplesw=[]
	cr_samplesw=[]
	crossamps = []
	tau_valuee=[]
	tau_valuew=[]
	t0_valuew=[]
	t0_valuee=[]
	minNsamples = 600
	arre=np.round(np.array(timevece),decimals=3)
	arrw=np.round(np.array(timevecw),decimals=3)
	for eachminute in range(0,1440*6):#every 10 seconds
		s4_times.append(eachminute*10.0)

	for eachminute in s4_times: # cam I do 1440 in parallel or split in 10 cores?
		idxarraye   = (arre >= eachminute) & (arre < (eachminute+(60.0)) )# bool array
		idxarrayw   = (arrw >= eachminute) & (arrw < (eachminute+(60.0)) )# bool array

		amp_dBe = powerDatadBe[idxarraye]
		amp_lne = np.power(10,np.array(amp_dBe)/10.0).astype(int)
		amp_nue = len(amp_lne)
		if amp_nue>minNsamples:
			s4e=np.nanstd(amp_lne,ddof=1)/np.nanmean(amp_lne)
			s4_valuese.append(s4e)
		else:
			s4e = -1
			s4_valuese.append(np.NaN)
		s4_samplese.append(amp_nue)


		amp_dBw = powerDatadBw[idxarrayw]
		amp_lnw = np.power(10,np.array(amp_dBw)/10.0).astype(int)
		amp_nuw = len(amp_lnw)
		if amp_nuw>minNsamples:
			s4w=np.nanstd(amp_lnw,ddof=1)/np.nanmean(amp_lnw)
			s4_valuesw.append(s4w)
		else:
			s4w = -1
			s4_valuesw.append(np.NaN)
		s4_samplesw.append(amp_nuw)

		S4_TRESHOLD = 0.125
		if (amp_nuw>=minNsamples) and (amp_nue>=minNsamples) and (s4e>=S4_TRESHOLD) and (s4w>=S4_TRESHOLD):
#             n_lags.append(28)
			mintime_e = np.nanmin(arre[idxarraye])
			mintime_w = np.nanmin(arrw[idxarrayw])
			mintime_t = np.nanmax([mintime_e,mintime_w])
			#get the shortest time in the other side
			maxtime_e = np.nanmax(arre[idxarraye])
			maxtime_w = np.nanmax(arrw[idxarrayw])
			maxtime_t = np.nanmin([maxtime_e,maxtime_w])

			#commom times
			tarre  = arre[idxarraye]
			tarrw  = arrw[idxarrayw]

			ctime_e = np.where( (tarre>=mintime_t) & (tarre<=maxtime_t),tarre,float("nan") )
			ctime_w = np.where( (tarrw>=mintime_t) & (tarrw<=maxtime_t),tarrw,float("nan") )
			itime_t = np.round(np.arange(eachminute,eachminute+60,0.05),decimals=2)
			itime_t = np.where( (itime_t>=mintime_t) & (itime_t<=maxtime_t),itime_t,float("nan") )
			#we are not going to interpolate values
			if_e = scipy.interpolate.interp1d(tarre,amp_dBe,kind='nearest')#train with the originals time vectors
			if_w = scipy.interpolate.interp1d(tarrw,amp_dBw,kind='nearest')#train with the originals time vectors
			idB_e = if_e(itime_t)
			idB_w = if_w(itime_t)
			# Just to avoid interpolation over the out-of-lock regions

			dtarre = np.diff(ctime_e)
			dtarrw = np.diff(ctime_w)

			ngapse=len(dtarre[dtarre>=1]) # Number of time jumps over 1.0 sec
			ngapsw=len(dtarrw[dtarrw>=1]) # Number of time jumps over 1.0 sec
			boundse= []
			temp_dtarre = dtarre
			for idx in range(0,ngapse):
				maxidx = np.argmax(temp_dtarre)
				boundse.append([maxidx,maxidx+1])
				temp_dtarre[maxidx] = np.NaN
				pair = [maxidx,maxidx+1]
#                 cctime_e = np.where( (ctime_e>=boundse[pair[0]]) & ((ctime_e<=boundse[pair[1]])),np.NaN,ctime_e)
				itime_t  = np.where( (itime_t>=ctime_e[pair[0]]) & ((itime_t<=ctime_e[pair[1]])),float("nan"),itime_t)

			boundsw= []
			temp_dtarrw = dtarrw
			for idx in range(0,ngapsw):
				maxidx = np.argmax(temp_dtarrw)
				boundsw.append([maxidx,maxidx+1])
				temp_dtarrw[maxidx] = np.NaN
				pair = [maxidx,maxidx+1]
#                 cctime_w = np.where( (ctime_w>=boundsw[pair[0]]) & ((ctime_w<=boundsw[pair[1]])),np.NaN,ctime_w)
				itime_t  = np.where( (itime_t>=ctime_w[pair[0]]) & ((itime_t<=ctime_w[pair[1]])),float("nan"),itime_t)

			z_idB_e = idB_e - np.nanmean(idB_e)
			z_idB_w = idB_w - np.nanmean(idB_w)
			nanmask = ~np.isnan(itime_t)
			nz_idB_e   = z_idB_e[nanmask]
			nz_idB_w   = z_idB_w[nanmask]
			nz_itime_t = itime_t[nanmask]

			corr = correlate(nz_idB_w,nz_idB_e)/(norm(nz_idB_w)*norm(nz_idB_e))
			crossamps.append(np.nanmax(corr))
			maxidx = np.argmax(corr)
			yff = np.roll(nz_idB_w,-maxidx)
			corr2 = correlate(nz_idB_e,yff)/(norm(nz_idB_e)*norm(yff))
			maxidx2 = np.argmax(corr2)
#             print('delta samples :',maxidx-maxidx2-1)
			drops = maxidx-maxidx2-1
			if (np.nanmax(corr2) > 0.5) and (s4e>=0.2) and (s4w>=0.2) and (drops!=0):
				corr3 = correlate(nz_idB_e[:-drops],yff[:-drops])/(norm(nz_idB_e[:-drops])*norm(yff[:-drops]))
				maxidx3 = np.argmax(corr3)
				# print("maxidx3:",maxidx3)
			else:
				corr3 = corr2

			n_lags.append(maxidx-maxidx2-1)

			acorr= np.correlate(nz_idB_e,nz_idB_e,mode='full')
			idxmax = np.argmax(acorr)
			valmax = np.max(acorr)
			ccorrmax = np.nanmax(corr3)
			samples=0
			db3 = False
			uff= acorr[idxmax:]
			for eachvalue in uff:
				if eachvalue<=(valmax*0.5):
					db3 =True
					break
				else:
					samples=samples+1
			tau_valuee.append(samples*0.05)

			samples=0
			db3 = False
			uffn = uff/valmax
			for eachvalue in uffn:
				if eachvalue<=(ccorrmax):
					db3 =True
					break
				else:
					samples=samples+1
			t0_valuee.append(samples*0.05)

			acorr= np.correlate(nz_idB_w,nz_idB_w,mode='full')
			idxmax = np.argmax(acorr)
			valmax = np.max(acorr)
			samples=0
			db3 = False
			uff= acorr[idxmax:]
			for eachvalue in uff:
				if eachvalue<=(valmax*0.5):
					db3 =True
					break
				else:
					samples=samples+1
			tau_valuew.append(samples*0.05)

			samples=0
			db3 = False
			uffn = uff/valmax
			for eachvalue in uffn:
				if eachvalue<=(ccorrmax):
					db3 =True
					break
				else:
					samples=samples+1
			t0_valuew.append(samples*0.05)

			cr_samplese.append(len(ctime_e))
			cr_samplesw.append(len(ctime_w))

		else:
			n_lags.append(float("nan"))
			cr_samplese.append(float("nan"))
			cr_samplesw.append(float("nan"))
			crossamps.append(float("nan"))
			tau_valuee.append(float("nan"))
			tau_valuew.append(float("nan"))
			t0_valuew.append(float("nan"))
			t0_valuee.append(float("nan"))
		#get the minimun time in the 1 min vector values
	return s4_times,s4_valuese,s4_samplese,s4_valuesw,s4_samplesw,n_lags,cr_samplese,cr_samplesw,crossamps,tau_valuee,tau_valuew,t0_valuee,t0_valuew

def s4_v2(powerDatadBe,timevece):
	add86400=False
	if np.nanmax(timevece)>86400:
		timevece = timevece%86400
		add86400 = True

	s4_times=[]
	n_lags=[]
	s4_valuese=[]
	s4_samplese=[]
	cr_samplese=[]
	s4_valuesw=[]
	s4_samplesw=[]
	cr_samplesw=[]
	crossamps = []
	tau_valuee=[]
	tau_valuew=[]
	t0_valuew=[]
	t0_valuee=[]
	minNsamples = 1000
	arre=np.round(np.array(timevece),decimals=2)
	for eachminute in range(0,1440):#every 10 seconds
		s4_times.append(eachminute*60.0)

	for eachminute in s4_times:
		idxarraye   = (arre >= eachminute) & (arre < (eachminute+(60.0)) )# bool array

		amp_dBe = powerDatadBe[idxarraye]

		num_of_SNRvalues = len(np.unique(powerDatadBe[idxarraye]))
		amp_lne = np.power(10,np.array(amp_dBe)/10.0).astype(float)
		amp_nue = len(amp_lne)
		if ((amp_nue>minNsamples) and (num_of_SNRvalues>=2)):
			# print(np.unique(powerDatadBe[idxarraye]))
			s4e=np.nanstd(amp_lne,ddof=1)/np.nanmean(amp_lne)
			s4_valuese.append(s4e)
			amp_dBe_zo = amp_dBe - np.nanmean(amp_dBe)
			uu= np.correlate(amp_dBe_zo,amp_dBe_zo,mode='full')
			uuargmax = np.nanargmax(uu)
			uumax = np.nanmax(uu)
			uuLHALF = np.where(uu>(uumax*0.5),uu,float("nan"))
			uuLHALFmax = np.nanmax(uuLHALF)
			uuLHALFargmax = np.nanargmax(uuLHALF)
			uuLHALFargmin = np.nanargmin(uuLHALF[uuLHALFargmax:])
			tau_valuee.append((uuLHALFargmax+uuLHALFargmin-uuargmax)*0.02)
		else:
			s4e = -1
			s4_valuese.append(np.NaN)
			tau_valuee.append(np.NaN)
		s4_samplese.append(amp_nue)
	if add86400:
		s4_times = np.array(s4_times)+np.ones((len(s4_times)))*86400
	return s4_times,s4_valuese,s4_samplese,tau_valuee

def s4_50Hz(powerDatadBe,timevece):
	if np.nanmax(timevece)>86400:
		print("ERROR: timestamp required 0-86400 sec range")

	s4_times=[]
	n_lags=[]
	s4_valuese=[]
	s4_samplese=[]
	cr_samplese=[]
	s4_valuesw=[]
	s4_samplesw=[]
	cr_samplesw=[]
	crossamps = []
	tau_valuee=[]
	tau_valuew=[]
	t0_valuew=[]
	t0_valuee=[]
	minNsamples = 2500
	arre=np.round(np.array(timevece),decimals=2)
	for eachminute in range(0,1440):#every 10 seconds
		s4_times.append(eachminute*60.0)

	for eachminute in s4_times:
		idxarraye   = (arre >= eachminute) & (arre < (eachminute+(60.0)) )# bool array

		amp_dBe = powerDatadBe[idxarraye]
		amp_lne = np.power(10,np.array(amp_dBe)/10.0).astype(float)
		amp_nue = len(amp_lne)
		if (amp_nue>minNsamples):
			s4e=np.nanstd(amp_lne,ddof=1)/np.nanmean(amp_lne)
			s4_valuese.append(s4e)
			amp_dBe_zo = amp_dBe - np.nanmean(amp_dBe)
			uu= np.correlate(amp_dBe_zo,amp_dBe_zo,mode='full')
			uuargmax = np.nanargmax(uu)
			uumax = np.nanmax(uu)
			uuLHALF = np.where(uu>(uumax*0.5),uu,float("nan"))
			uuLHALFmax = np.nanmax(uuLHALF)
			uuLHALFargmax = np.nanargmax(uuLHALF)
			uuLHALFargmin = np.nanargmin(uuLHALF[uuLHALFargmax:])
			tau_valuee.append((uuLHALFargmax+uuLHALFargmin-uuargmax)*0.02)
		else:
			s4e = -1
			s4_valuese.append(np.NaN)
			tau_valuee.append(np.NaN)
		s4_samplese.append(amp_nue)
	return s4_times,s4_valuese,s4_samplese,tau_valuee

def s4_cross_50Hz(powerDatadBe,timevece,powerDatadBw,timevecw):
	s4_times=[]
	n_lags=[]
	s4_valuese=[]
	s4_samplese=[]
	cr_samplese=[]
	s4_valuesw=[]
	s4_samplesw=[]
	cr_samplesw=[]
	crossamps = []
	tau_valuee=[]
	tau_valuew=[]
	t0_valuew=[]
	t0_valuee=[]
	minNsamples = 600
	arre=np.round(np.array(timevece),decimals=2)
	arrw=np.round(np.array(timevecw),decimals=2)
	for eachminute in range(0,1440*6):#every 10 seconds
		s4_times.append(eachminute*10.0)

	for eachminute in s4_times:
		idxarraye   = (arre >= eachminute) & (arre < (eachminute+(60.0)) )# bool array
		idxarrayw   = (arrw >= eachminute) & (arrw < (eachminute+(60.0)) )# bool array

		amp_dBe = powerDatadBe[idxarraye]
		amp_lne = np.power(10,np.array(amp_dBe)/10.0).astype(int)
		amp_nue = len(amp_lne)
		if amp_nue>minNsamples:
			s4e=np.nanstd(amp_lne,ddof=1)/np.nanmean(amp_lne)
			s4_valuese.append(s4e)
		else:
			s4e = -1
			s4_valuese.append(np.NaN)
		s4_samplese.append(amp_nue)


		amp_dBw = powerDatadBw[idxarrayw]
		amp_lnw = np.power(10,np.array(amp_dBw)/10.0).astype(int)
		amp_nuw = len(amp_lnw)
		if amp_nuw>minNsamples:
			s4w=np.nanstd(amp_lnw,ddof=1)/np.nanmean(amp_lnw)
			s4_valuesw.append(s4w)
		else:
			s4w = -1
			s4_valuesw.append(np.NaN)
		s4_samplesw.append(amp_nuw)

		S4_TRESHOLD = 0.1
		if (amp_nuw>=minNsamples) and (amp_nue>=minNsamples) and (s4e>=S4_TRESHOLD) and (s4w>=S4_TRESHOLD):
#             n_lags.append(28)
			mintime_e = np.nanmin(arre[idxarraye])
			mintime_w = np.nanmin(arrw[idxarrayw])
			mintime_t = np.nanmax([mintime_e,mintime_w])
			#get the shortest time in the other side
			maxtime_e = np.nanmax(arre[idxarraye])
			maxtime_w = np.nanmax(arrw[idxarrayw])
			maxtime_t = np.nanmin([maxtime_e,maxtime_w])

			#commom times
			tarre  = arre[idxarraye]
			tarrw  = arrw[idxarrayw]

			ctime_e = np.where( (tarre>=mintime_t) & (tarre<=maxtime_t),tarre,float("nan") )
			ctime_w = np.where( (tarrw>=mintime_t) & (tarrw<=maxtime_t),tarrw,float("nan") )
			itime_t = np.round(np.arange(eachminute,eachminute+60,0.02),decimals=2)
			itime_t = np.where( (itime_t>=mintime_t) & (itime_t<=maxtime_t),itime_t,float("nan") )
			#we are not going to interpolate values
			if_e = scipy.interpolate.interp1d(tarre,amp_dBe,kind='nearest')#train with the originals time vectors
			if_w = scipy.interpolate.interp1d(tarrw,amp_dBw,kind='nearest')#train with the originals time vectors
			idB_e = if_e(itime_t)
			idB_w = if_w(itime_t)
			# Just to avoid interpolation over the out-of-lock regions

			dtarre = np.diff(ctime_e)
			dtarrw = np.diff(ctime_w)

			ngapse=len(dtarre[dtarre>=1]) # Number of time jumps over 1.0 sec
			ngapsw=len(dtarrw[dtarrw>=1]) # Number of time jumps over 1.0 sec
			boundse= []
			temp_dtarre = dtarre
			for idx in range(0,ngapse):
				maxidx = np.argmax(temp_dtarre)
				boundse.append([maxidx,maxidx+1])
				temp_dtarre[maxidx] = np.NaN
				pair = [maxidx,maxidx+1]
#                 cctime_e = np.where( (ctime_e>=boundse[pair[0]]) & ((ctime_e<=boundse[pair[1]])),np.NaN,ctime_e)
				itime_t  = np.where( (itime_t>=ctime_e[pair[0]]) & ((itime_t<=ctime_e[pair[1]])),float("nan"),itime_t)

			boundsw= []
			temp_dtarrw = dtarrw
			for idx in range(0,ngapsw):
				maxidx = np.argmax(temp_dtarrw)
				boundsw.append([maxidx,maxidx+1])
				temp_dtarrw[maxidx] = np.NaN
				pair = [maxidx,maxidx+1]
#                 cctime_w = np.where( (ctime_w>=boundsw[pair[0]]) & ((ctime_w<=boundsw[pair[1]])),np.NaN,ctime_w)
				itime_t  = np.where( (itime_t>=ctime_w[pair[0]]) & ((itime_t<=ctime_w[pair[1]])),float("nan"),itime_t)

			z_idB_e = idB_e - np.nanmean(idB_e)
			z_idB_w = idB_w - np.nanmean(idB_w)
			nanmask = ~np.isnan(itime_t)
			nz_idB_e   = z_idB_e[nanmask]
			nz_idB_w   = z_idB_w[nanmask]
			nz_itime_t = itime_t[nanmask]

			corr = correlate(nz_idB_w,nz_idB_e)/(norm(nz_idB_w)*norm(nz_idB_e))
			crossamps.append(np.nanmax(corr))
			maxidx = np.argmax(corr)
			yff = np.roll(nz_idB_w,-maxidx)
			corr2 = correlate(nz_idB_e,yff)/(norm(nz_idB_e)*norm(yff))
			maxidx2 = np.argmax(corr2)
#             print('delta samples :',maxidx-maxidx2-1)
			drops = maxidx-maxidx2-1
			if (np.nanmax(corr2) > 0.5) and (s4e>=0.1) and (s4w>=0.1) and (drops!=0):
				corr3 = correlate(nz_idB_e[:-drops],yff[:-drops])/(norm(nz_idB_e[:-drops])*norm(yff[:-drops]))
				maxidx3 = np.argmax(corr3)
				# print("maxidx3:",maxidx3)
			else:
				corr3 = corr2

			n_lags.append(maxidx-maxidx2-1)

			acorr= np.correlate(nz_idB_e,nz_idB_e,mode='full')
			idxmax = np.argmax(acorr)
			valmax = np.max(acorr)
			ccorrmax = np.nanmax(corr3)
			samples=0
			db3 = False
			uff= acorr[idxmax:]
			for eachvalue in uff:
				if eachvalue<=(valmax*0.5):
					db3 =True
					break
				else:
					samples=samples+1
			tau_valuee.append(samples*0.02)

			samples=0
			db3 = False
			uffn = uff/valmax
			for eachvalue in uffn:
				if eachvalue<=(ccorrmax):
					db3 =True
					break
				else:
					samples=samples+1
			t0_valuee.append(samples*0.02)

			acorr= np.correlate(nz_idB_w,nz_idB_w,mode='full')
			idxmax = np.argmax(acorr)
			valmax = np.max(acorr)
			samples=0
			db3 = False
			uff= acorr[idxmax:]
			for eachvalue in uff:
				if eachvalue<=(valmax*0.5):
					db3 =True
					break
				else:
					samples=samples+1
			tau_valuew.append(samples*0.02)

			samples=0
			db3 = False
			uffn = uff/valmax
			for eachvalue in uffn:
				if eachvalue<=(ccorrmax):
					db3 =True
					break
				else:
					samples=samples+1
			t0_valuew.append(samples*0.02)

			cr_samplese.append(len(ctime_e))
			cr_samplesw.append(len(ctime_w))

		else:
			n_lags.append(float("nan"))
			cr_samplese.append(float("nan"))
			cr_samplesw.append(float("nan"))
			crossamps.append(float("nan"))
			tau_valuee.append(float("nan"))
			tau_valuew.append(float("nan"))
			t0_valuew.append(float("nan"))
			t0_valuee.append(float("nan"))
		#get the minimun time in the 1 min vector values
	return s4_times,s4_valuese,s4_samplese,s4_valuesw,s4_samplesw,n_lags,cr_samplese,cr_samplesw,crossamps,tau_valuee,tau_valuew,t0_valuee,t0_valuew

def snr_cross(powerDatadBe,timevece,powerDatadBw,timevecw):
	s4_times=[]
	n_lags=[]
	s4_valuese=[]
	s4_samplese=[]
	cr_samplese=[]
	s4_valuesw=[]
	s4_samplesw=[]
	cr_samplesw=[]
	crossamps = []
	tau_valuee=[]
	tau_valuew=[]
	t0_valuew=[]
	t0_valuee=[]
	minNsamples = 600
	arre=np.round(np.array(timevece),decimals=3)
	arrw=np.round(np.array(timevecw),decimals=3)
	for eachminute in range(0,1440*6):#every 10 seconds
		s4_times.append(eachminute*10.0)

	for eachminute in s4_times:
		idxarraye   = (arre >= eachminute) & (arre < (eachminute+(480.0)) )# bool array
		idxarrayw   = (arrw >= eachminute) & (arrw < (eachminute+(480.0)) )# bool array

		amp_dBe = powerDatadBe[idxarraye]
		amp_lne = np.power(10,np.array(amp_dBe)/10.0).astype(int)
		amp_nue = len(amp_lne)
		if amp_nue>minNsamples:
			s4e=np.nanstd(amp_lne,ddof=1)/np.nanmean(amp_lne)
			s4_valuese.append(s4e)
		else:
			s4e = -1
			s4_valuese.append(np.NaN)
		s4_samplese.append(amp_nue)


		amp_dBw = powerDatadBw[idxarrayw]
		amp_lnw = np.power(10,np.array(amp_dBw)/10.0).astype(int)
		amp_nuw = len(amp_lnw)
		if amp_nuw>minNsamples:
			s4w=np.nanstd(amp_lnw,ddof=1)/np.nanmean(amp_lnw)
			s4_valuesw.append(s4w)
		else:
			s4w = -1
			s4_valuesw.append(np.NaN)
		s4_samplesw.append(amp_nuw)

		S4_TRESHOLD = 0.1
		if (amp_nuw>=minNsamples) and (amp_nue>=minNsamples) and (s4e>=S4_TRESHOLD) and (s4w>=S4_TRESHOLD):
#             n_lags.append(28)
			mintime_e = np.nanmin(arre[idxarraye])
			mintime_w = np.nanmin(arrw[idxarrayw])
			mintime_t = np.nanmax([mintime_e,mintime_w])
			#get the shortest time in the other side
			maxtime_e = np.nanmax(arre[idxarraye])
			maxtime_w = np.nanmax(arrw[idxarrayw])
			maxtime_t = np.nanmin([maxtime_e,maxtime_w])

			#commom times
			tarre  = arre[idxarraye]
			tarrw  = arrw[idxarrayw]

			ctime_e = np.where( (tarre>=mintime_t) & (tarre<=maxtime_t),tarre,float("nan") )
			ctime_w = np.where( (tarrw>=mintime_t) & (tarrw<=maxtime_t),tarrw,float("nan") )
			itime_t = np.round(np.arange(eachminute,eachminute+480,0.05),decimals=2)
			itime_t = np.where( (itime_t>=mintime_t) & (itime_t<=maxtime_t),itime_t,float("nan") )
			#we are not going to interpolate values
			if_e = scipy.interpolate.interp1d(tarre,amp_dBe,kind='nearest')#train with the originals time vectors
			if_w = scipy.interpolate.interp1d(tarrw,amp_dBw,kind='nearest')#train with the originals time vectors
			idB_e = if_e(itime_t)
			idB_w = if_w(itime_t)
			# Just to avoid interpolation over the out-of-lock regions

			dtarre = np.diff(ctime_e)
			dtarrw = np.diff(ctime_w)

			ngapse=len(dtarre[dtarre>=1]) # Number of time jumps over 1.0 sec
			ngapsw=len(dtarrw[dtarrw>=1]) # Number of time jumps over 1.0 sec
			boundse= []
			temp_dtarre = dtarre
			for idx in range(0,ngapse):
				maxidx = np.argmax(temp_dtarre)
				boundse.append([maxidx,maxidx+1])
				temp_dtarre[maxidx] = np.NaN
				pair = [maxidx,maxidx+1]
#                 cctime_e = np.where( (ctime_e>=boundse[pair[0]]) & ((ctime_e<=boundse[pair[1]])),np.NaN,ctime_e)
				itime_t  = np.where( (itime_t>=ctime_e[pair[0]]) & ((itime_t<=ctime_e[pair[1]])),float("nan"),itime_t)

			boundsw= []
			temp_dtarrw = dtarrw
			for idx in range(0,ngapsw):
				maxidx = np.argmax(temp_dtarrw)
				boundsw.append([maxidx,maxidx+1])
				temp_dtarrw[maxidx] = np.NaN
				pair = [maxidx,maxidx+1]
#                 cctime_w = np.where( (ctime_w>=boundsw[pair[0]]) & ((ctime_w<=boundsw[pair[1]])),np.NaN,ctime_w)
				itime_t  = np.where( (itime_t>=ctime_w[pair[0]]) & ((itime_t<=ctime_w[pair[1]])),float("nan"),itime_t)

			z_idB_e = idB_e - np.nanmean(idB_e)
			z_idB_w = idB_w - np.nanmean(idB_w)
			nanmask = ~np.isnan(itime_t)
			nz_idB_e   = z_idB_e[nanmask]
			nz_idB_w   = z_idB_w[nanmask]
			nz_itime_t = itime_t[nanmask]

			corr = correlate(nz_idB_w,nz_idB_e)/(norm(nz_idB_w)*norm(nz_idB_e))
			crossamps.append(np.nanmax(corr))
			maxidx = np.argmax(corr)
			yff = np.roll(nz_idB_w,-maxidx)
			corr2 = correlate(nz_idB_e,yff)/(norm(nz_idB_e)*norm(yff))
			maxidx2 = np.argmax(corr2)
#             print('delta samples :',maxidx-maxidx2-1)
			drops = maxidx-maxidx2-1
			if (np.nanmax(corr2) > 0.5) and (s4e>=0.0) and (s4w>=0.0) and (drops!=0):
				corr3 = correlate(nz_idB_e[:-drops],yff[:-drops])/(norm(nz_idB_e[:-drops])*norm(yff[:-drops]))
				maxidx3 = np.argmax(corr3)
				# print("maxidx3:",maxidx3)
			else:
				corr3 = corr2

			n_lags.append(maxidx-maxidx2-1)

			acorr= np.correlate(nz_idB_e,nz_idB_e,mode='full')
			idxmax = np.argmax(acorr)
			valmax = np.max(acorr)
			ccorrmax = np.nanmax(corr3)
			samples=0
			db3 = False
			uff= acorr[idxmax:]
			for eachvalue in uff:
				if eachvalue<=(valmax*0.5):
					db3 =True
					break
				else:
					samples=samples+1
			tau_valuee.append(samples*0.05)

			samples=0
			db3 = False
			uffn = uff/valmax
			for eachvalue in uffn:
				if eachvalue<=(ccorrmax):
					db3 =True
					break
				else:
					samples=samples+1
			t0_valuee.append(samples*0.05)

			acorr= np.correlate(nz_idB_w,nz_idB_w,mode='full')
			idxmax = np.argmax(acorr)
			valmax = np.max(acorr)
			samples=0
			db3 = False
			uff= acorr[idxmax:]
			for eachvalue in uff:
				if eachvalue<=(valmax*0.5):
					db3 =True
					break
				else:
					samples=samples+1
			tau_valuew.append(samples*0.05)

			samples=0
			db3 = False
			uffn = uff/valmax
			for eachvalue in uffn:
				if eachvalue<=(ccorrmax):
					db3 =True
					break
				else:
					samples=samples+1
			t0_valuew.append(samples*0.05)

			cr_samplese.append(len(ctime_e))
			cr_samplesw.append(len(ctime_w))

		else:
			n_lags.append(float("nan"))
			cr_samplese.append(float("nan"))
			cr_samplesw.append(float("nan"))
			crossamps.append(float("nan"))
			tau_valuee.append(float("nan"))
			tau_valuew.append(float("nan"))
			t0_valuew.append(float("nan"))
			t0_valuee.append(float("nan"))
		#get the minimun time in the 1 min vector values
	return s4_times,s4_valuese,s4_samplese,s4_valuesw,s4_samplesw,n_lags,cr_samplese,cr_samplesw,crossamps,tau_valuee,tau_valuew,t0_valuee,t0_valuew
