import numpy as np
import scipy.interpolate
import datetime
from datetime import timedelta
from os.path import exists
from matplotlib import pyplot

dt = np.dtype('float64')

def read_GPS(sp3path,sp3name,sv):
	Ts = []
	Xs = []
	Ys = []
	Zs = []

	file = "%s\%s"%(sp3path,sp3name)
	with open(file) as fd:
		for n, line in enumerate(fd):
			if line[0]=='*' :
				Ts.append(int(line[14:16])*3600+int(line[17:19])*60+float(line[21:24]))
				# print(line[14:16],line[17:19],line[21:24])
			if (line[0]=='P') and (line[1]=='G') and int(line[2:4])==sv:
				# print(line)
				#oint(self._eval(line[4:18]), self._eval(line[18:32]), self._eval(line[32:46])),
				Xs.append(float(line[4:18]))
				Ys.append(float(line[18:32]))
				Zs.append(float(line[32:46]))
	#TODO: Add the last point only for interpolation purposes
	return Ts[:-1],Xs[:-1],Ys[:-1],Zs[:-1]

def read_GLO(sp3path,sp3name,sv):
	Ts = []
	Xs = []
	Ys = []
	Zs = []

	file = "%s\%s"%(sp3path,sp3name)
	with open(file) as fd:
		for n, line in enumerate(fd):
			if line[0]=='*' :
				Ts.append(int(line[14:16])*3600+int(line[17:19])*60+float(line[21:24]))
				# print(line[14:16],line[17:19],line[21:24])
			if (line[0]=='P') and (line[1]=='R') and int(line[2:4])==sv:
				# print(line)
				#oint(self._eval(line[4:18]), self._eval(line[18:32]), self._eval(line[32:46])),
				Xs.append(float(line[4:18]))
				Ys.append(float(line[18:32]))
				Zs.append(float(line[32:46]))
	#TODO: Add the last point only for interpolation purposes
	return Ts[:-1],Xs[:-1],Ys[:-1],Zs[:-1]

def read_GAL(sp3path,sp3name,sv):
	Ts = []
	Xs = []
	Ys = []
	Zs = []

	file = "%s\%s"%(sp3path,sp3name)
	# print(file)
	with open(file) as fd:
		for n, line in enumerate(fd):
			if line[0]=='*' :
				Ts.append(int(line[14:16])*3600+int(line[17:19])*60+float(line[21:24]))
				# print(line[14:16],line[17:19],line[21:24])
			if (line[0]=='P') and (line[1]=='E') and int(line[2:4])==sv:
				# print('line[2:4]:',int(line[2:4]))
				#oint(self._eval(line[4:18]), self._eval(line[18:32]), self._eval(line[32:46])),
				Xs.append(float(line[4:18]))
				Ys.append(float(line[18:32]))
				Zs.append(float(line[32:46]))
	#TODO: Add the last point only for interpolation purposes
	return Ts[:-1],Xs[:-1],Ys[:-1],Zs[:-1]

def read_BDS(sp3path,sp3name,sv):
	Ts = []
	Xs = []
	Ys = []
	Zs = []

	file = "%s\%s"%(sp3path,sp3name)
	# print(file)
	with open(file) as fd:
		for n, line in enumerate(fd):
			if line[0]=='*' :
				Ts.append(int(line[14:16])*3600+int(line[17:19])*60+float(line[21:24]))
				# print(line[14:16],line[17:19],line[21:24])
			if (line[0]=='P') and (line[1]=='C') and int(line[2:4])==sv:
				# print('line[2:4]:',int(line[2:4]))
				#oint(self._eval(line[4:18]), self._eval(line[18:32]), self._eval(line[32:46])),
				Xs.append(float(line[4:18]))
				Ys.append(float(line[18:32]))
				Zs.append(float(line[32:46]))
	#TODO: Add the last point only for interpolation purposes
	return Ts[:-1],Xs[:-1],Ys[:-1],Zs[:-1]

def read(sp3path,sp3name,sv):
	Ts = []
	Xs = []
	Ys = []
	Zs = []

	file = "%s\%s"%(sp3path,sp3name)
	with open(file) as fd:
		for n, line in enumerate(fd):
			if line[0]=='*' :
				Ts.append(int(line[14:16])*3600+int(line[17:19])*60+float(line[21:24]))
				# print(line[14:16],line[17:19],line[21:24])
			if (line[0]=='P'):
				if int(line[2:4])==sv:
					#oint(self._eval(line[4:18]), self._eval(line[18:32]), self._eval(line[32:46])),
					Xs.append(float(line[4:18]))
					Ys.append(float(line[18:32]))
					Zs.append(float(line[32:46]))
	#TODO: Add the last point only for interpolation purposes
	return Ts,Xs,Ys,Zs
			# print(line[2:4],line[5:19],line[19:32],line[35:47])

def wgs2xyz(lam,phi,h):

	'''
	function [x,y,z] = wgs2xyz(lam,phi,h)

	% WGS2XYZ   Converts lam(longitude) phi(latitude) ellipsoidal coordinates
	%           from WGS-84 to ECEF cartesian coordinates. Vectorized.
	%
	% Input:  lam (longitude), phi (latitude), h can be vectors
	%         lon, lat in decimal degrees
	%         h in meters above ellipsoid
	%
	% Output: x,y,z in meters
	%
	% Call: [x,y,z] = wgs2xyz(longitude,latitude,h)
	%
	% eric.calais@ens.fr
	% semimajor axis axis and flattening for WGS-84
	'''
	a = 6378137.0000;
	f = 1.0/298.257223563;
	# semiminor axis (should be 6356752.3142);
	b = a * (1-f);
	# eccentricity
	ecc = 2*f - f**2;

	# degrees to radians
	lam = lam*np.pi/180.0;
	phi = phi*np.pi/180.0;

	# radius of curvature in prime vertical
	N = a/ np.sqrt(1-(np.sin(phi))**2*ecc);
	#%N = a^2 / sqrt((cos(phi)).^2*a^2 + (sin(phi)).^2*b^2);

	x = np.cos(phi)*np.cos(lam)*(N+h);
	y = np.cos(phi)*np.sin(lam)*(N+h);
	z = np.sin(phi)*(N*(b**2/a**2) + h);
	return x,y,z

#TODO test azelle
def azelle(Sx,Sy,Sz,Rx,Ry,Rz):
	'''
	% AZELLE	Computes elevation angle, range, and azimuth
	%        of satellites from a ground station.
	%
	%        Input:
	%          Sx,Sy,Sz = ECEF satellite coordinates (m), n x 3 matrix
	%          Rx,Ry,Rz = ground station coordinates, ECEF (m) (vector)
	%        Output:
	%          AZ = azimuth (radians CW from north)
	%          EL = elevation angle (radians)
	%          LE = range (meters)
	%
	%        Usage:
	%          [AZ,EL,LE] = azelle(Sx,Sy,Sz,Rx,Ry,Rz);
	%
	'''
	# compute ground-sat vector in ECEF coordinates
	Rgsx = Sx - np.ones(len(Sx))*Rx
	Rgsy = Sy - np.ones(len(Sy))*Ry
	Rgsz = Sz - np.ones(len(Sz))*Rz
	Rgsx=np.array(Rgsx, dtype=dt)
	Rgsy=np.array(Rgsy, dtype=dt)
	Rgsz=np.array(Rgsz, dtype=dt)
	# print('************Rgsx')
	# print(Rgsx[3])
	# convert to unit vector
	rang = np.sqrt(Rgsx**2.0+Rgsy**2.0+Rgsz**2.0);
	Ru = np.array([Rgsx/rang,Rgsy/rang,Rgsz/rang], dtype=dt)#NORMALIZED VECTOR

	# dummy stdev and correlation for xyz2neu
	SV = np.zeros((Ru.shape[1],Ru.shape[1]));
	COR = np.zeros((Ru.shape[1],3));
	[neu,Cneu,LLH] = xyz2neu([Rx,Ry,Rz],Ru,SV,COR);
	neu =np.array(neu, dtype=dt)
	Cneu=np.array(Cneu, dtype=dt)
	# convert neu to azimuth and elevation angle
	LE = np.sqrt(neu[:,0]**2.0+neu[:,1]**2.0);
	#EL = (pi/2) - atan2(LE,neu(:,3));
	EL = np.arctan2(LE,neu[:,2]);
	AZ = np.arctan2(neu[:,1],neu[:,0]);

	return AZ,EL,LE,rang,neu

def neu2xyz(O,V,SV,COR):
	'''
	% XYZ2NEU Convert local topocentric into ECEF
	%
	% 	Input:
	% 	  - O = origin vector in ellipsoidal coordinates (lat lon height) only 3 floats
	% 	  - V = position or velocity vector in NEU frame (m or m/yr)
	% 	  - SV = stdev in NEU frame (m or m/yr)
	% 	  - COR = correlations, NE NU EU (m^2 or m/yr^2)
	% 	    (NOTE: O, V, SV, COR can be n x 3 matrices, n = # of sites)
	%
	% 	Output:
	% 	  - XYZ = output in ECEF Cartesian frame (m)
	% 	  - CXYZ = associated covariance (m), format is:
	% 	           Cxx Cxy Cxz Cyy Cyz Czz
	% 	    (NOTE: XYZ and CXYZ will be matrices with n rows)
	%
	% 	Call: [XYZ,CXYZ] = neu2xyz(O,V,SV,COR);
	'''
	# if O is a single point, make it the same size as V
	O= np.array(O, dtype=dt)
	if len(O.shape)==1:
		lat = np.ones((V.shape[0]))*O[0]
		lon = np.ones((V.shape[0]))*O[1]
		h   = np.ones((V.shape[0]))*O[2]
	else :
		lat = O[0]
		lon = O[1]
		h   = O[2]

	# read rest of input
	V = np.array(V, dtype=dt)
	SV = np.array(SV, dtype=dt)
	COR = np.array(COR, dtype=dt)

	vn = V[:,0]; ve = V[:,1]; vu = V[:,2];
	#TODO: DOUBLE CHECK THE FOLLOWING 2 LINES MAY BE WRONG
	svn = SV[0]; sve = SV[1]; svu = SV[2];
	cne = COR[:,0]; cnu = COR[:,1]; ceu = COR[:,2];

	# % convert position(s) of origin to ECEF
	[XR,YR,ZR] = wgs2xyz(lon,lat,h);
	# print('XR,YR,ZR:',XR[0],YR[0],ZR[0])
	# % compute sines and cosines
	cp = np.cos(lon*np.pi/180.0);
	sp = np.sin(lon*np.pi/180.0);
	cl = np.cos(lat*np.pi/180.0);
	sl = np.sin(lat*np.pi/180.0);

	# % initiate outputs
	XYZ = [];
	CXYZ = [];

	# % for each site
	for i in range(0,V.shape[0]):
		# % build the rotation matrix
		R = [[-sl[i]*cp[i], -sl[i]*sp[i], cl[i] ],
			 [-sp[i]      ,   cp[i]     , 0.0],
			 [cl[i]*cp[i] ,  cl[i]*sp[i], sl[i]  ]]
		R = np.array(R, dtype=dt)
		# % apply the rotation
		XYZi = np.dot(np.array(R, dtype=dt).T,np.array([vn[i],ve[i],vu[i]], dtype=dt).T)

		# % svu cannot be zero or R'*CVi*R may return negative variances
		if (svu[i]==0):
			svu[i]= np.mean([svn[i],sve[i]])
			# % build covariance for that site

		CVi = [	[svn[i]**2.0,  cne[i]   	,cnu[i]   ],
				[cne[i]     ,  sve[i]**2.0	,ceu[i]   ],
				[cnu[i]     ,  ceu[i]   	,svu[i]**2.0] ];
		# % propagate covariance
		CXYZi = np.dot(np.dot(R.T,CVi),R);
		# % increment result matrices
		XYZ.append(XYZi)
		CXYZ.append(np.array([CXYZi[0][0],CXYZi[0][1],CXYZi[0][2],CXYZi[1][1],CXYZi[1][2],CXYZi[2][2]], dtype=dt))

	return XYZ,CXYZ


def xyz2neu(O,V,SV,COV):
	'''
	function [NEU,CNEU,LLH] = xyz2neu(O,V,SV,COV);

	% XYZ2NEU  Convert ECEF coordinates into local (topocentric) frame.
	%
	%     Input:
	%       O = origin vector in ECEF frame (m)
	%       V = position or velocity vector in ECEF frame (m or m/yr)
	%       SV = stdev in ECEF frame (m or m/yr)
	%       COV = covariances, XY XZ YZ (m^2 or m^2/yr^2)
	%       (NOTE: O, V, SV, COV can be n x 3 matrices, n = # of sites)
	%
	%     Output:
	%       NEU  = output on NEU frame (m or m/yr)
	%       CNEU = associated covariance (m or m^2/yr^2), format is:
	%                    Cnn Cne Cnu Cee Ceu Cuu
	%       (NOTE: NEU and CNEU will be matrices with n rows)
	%       LLH  = coordinates of origin vector in WGS84
	%              time lon lat elevation
	%
	%     Call: [NEU,CNEU,LLH] = xyz2neu(O,V,SV,COV);
	'''
# if O is a single point, make it the same size as V
	dt = np.dtype('float64')
	O= np.array(O, dtype=dt)
	# print("O.Shape:",O.shape)
	# print("len(O.Shape):",len(O.shape))
	# print("V,shape",V.shape)
	if len(O.shape)==1:
		XR = np.ones((V.shape[1]))*O[0]
		YR = np.ones((V.shape[1]))*O[1]
		ZR = np.ones((V.shape[1]))*O[2]
	else :
		XR = O[0]
		YR = O[1]
		ZR = O[2]
	# read rest of input
	V = np.array(V, dtype=dt)
	SV = np.array(SV, dtype=dt)
	COV = np.array(COV, dtype=dt)

	vx = V[0]; vy = V[1]; vz = V[2];
	svx = SV[0]; svy = SV[1]; svz = SV[2];
	cxy = COV[:,0]; cxz = COV[:,1]; cyz = COV[:,2];
	# convert origin vector to ellipsoidal coordinates
	T = np.zeros((XR.shape[0],1));
	LLH = xyz2wgs([T, XR, YR, ZR])
	# compute sines and cosines
	cp = np.cos(LLH[1]*np.pi/180.0);
	sp = np.sin(LLH[1]*np.pi/180.0); #% longitude
	cl = np.cos(LLH[2]*np.pi/180.0);
	sl = np.sin(LLH[2]*np.pi/180.0); #% latitude

	NEU = [];
	CNEU = [];
	for i in range(0,V.shape[1]):
		# build the rotation matrix
		R = [[-sl[i]*cp[i],-sl[i]*sp[i],cl[i]],
			  [-sp[i]     , cp[i]	   ,  0.0],
			  [cl[i]*cp[i], cl[i]*sp[i],sl[i]]]
		R = np.array(R, dtype=dt)
		# apply the rotation
		NEUi = np.dot(np.array(R, dtype=dt),np.array([vx[i],vy[i],vz[i]], dtype=dt).T);
		# build covariance for that site
		CVi = [[svx[i]**2.0,  cxy[i]   ,cxz[i]   ],
			   [cxy[i]   ,  svy[i]**2.0,cyz[i]   ],
			   [cxz[i]   ,	cyz[i]   ,svz[i]**2.0] ];
		#% propagate covariance
		CNEUi = np.dot(np.dot(R,CVi),R.T);

		NEU.append(NEUi)
		CNEU.append(np.array([CNEUi[0][0],CNEUi[0][1],CNEUi[0][2],CNEUi[1][1],CNEUi[1][2],CNEUi[2][2]], dtype=dt))

	return NEU,CNEU,LLH


#######################################
def xyz2wgs(S):
	'''
	function R = xyz2wgs(S)

	% XYZ2WGS  Converts cartesian coordinates (x,y,z) into
	%          ellipsoidal coordinates (lat,lon,alt) on WGS-84
	%          according to a non iterative method (Bowring 76,
	%          see also GPS Theory and Practice, p. 258).
	%
	% Input:
	%   S = nx4 matrix with time, X, Y, Z
	%   A! first column of S is time but can be dummy.
	%
	% Output:
	%   R = nx4 matrix with time, lon (lam), lat (phi), elevation
	%       (lon,lat in decimal degrees, elevation in meters above ellipsoid)
	%
	% Call: R = xyz2wgs(S)
	%
	% eric.calais@ens.fr
	'''
	# semimajor axis axis and flattening for WGS-84
	a = 6378137.0000;
	f = 1.0/298.257223563;
	# semiminor axis (should be 6356752.3142);
	b = a*(1.0-f);
	# eccentricity
	ecc = 2.0*f - f**2.0;

	# second numerical eccentricity
	e1 = (a**2.0-b**2.0)/(b**2.0);

	# read data
	t = S[0];
	x = S[1];
	y = S[2];
	z = S[3];
	# print("t,shape",t.shape)
	# auxiliary quantities
	p = np.sqrt(x**2.0+y**2.0);
	theta = np.arctan2(z*a,p*b);
	# longitude
	lam = np.arctan2(y,x);

	# latitude
	phi = np.arctan2(np.ones(len(theta))*z + (np.sin(theta))**3.0*e1*b , np.ones(len(theta))*p - (np.cos(theta))**3.0*(ecc**2.0)*a);

	# radius of curvature in prime vertical
	N = a/ np.sqrt(np.ones(len(phi))-(np.sin(phi))**2*ecc);
	#N = a / sqrt((cos(phi)).^2*a^2 + (sin(phi)).^2*b^2);
	# geocentric (?) altitude
	alt_g = (p / np.cos(phi)) - N;

	# ellipsoidal altitude
	alt = p*np.cos(phi) + z*np.sin(phi) - a*np.sqrt(np.ones(len(phi)) - ecc*np.sin(phi)**2.0);
	# fill out result matrix
	R = [t,lam*180.0/np.pi,phi*180.0/np.pi,alt];

	return R
# def Lagrange(Lx, Ly):
# 	x=sympy.symbols('x')
# 	if  len(Lx)!= len(Ly):
# 		return 1
# 	y=0
# 	for k in range ( len(Lx) ):
# 		t=1
# 		for j in range ( len(Lx) ):
# 			if j != k:
# 				t=t* ( (x-Lx[j]) /(Lx[k]-Lx[j]) )
# 		y+= t*Ly[k]
# 	return y

def interpolate_fromsp3(sp3path,date,sv,cutoffelev,geolong,geolat,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	weekn,dow = gnsscal.date2gpswd(datedata)
	sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	fullfile = "%s\%s"%(sp3path,sp3name)
	if(exists(fullfile)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		Ts,Xs,Ys,Zs = read(sp3path,sp3name,sv)
		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)

		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		XR,YR,ZR = wgs2xyz(-35.9061,-7.2122266,552.50323)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		#NEU TO XYZ
		# NEUB = neu2xyz()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])
		V_UPDW_350 = V_UPDW_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev;

		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,float("nan") );
		azim = np.where(zen_ang>cutoff_rad,azim,float("nan"))#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,float("nan"))#greater than 30 deg

		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		# ### JUST FOR HIGHER SAMPLING)
		# azimf    = scipy.interpolate.interp1d(Tstmp,azim,kind='linear')
		# zen_angf = scipy.interpolate.interp1d(Tstmp,zen_ang,kind='linear')
		# midpoints = np.arange(np.nanmin(Tstmp),np.nanmax(Tstmp),cross_corr_space)
		# azimi = azimf(midpoints) # any of these values should be nan
		# zen_angi = zen_angf(midpoints) # complete the rest of the values with 0?

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi[1:]
def interpolate_fromsp3_IAC(sp3path,date,sv,cutoffelev,geolong,geolat,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))

	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	# print(fullfile)
	if(exists(fullfile)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		Ts,Xs,Ys,Zs = read_GPS(sp3path,sp3name,sv)
		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)

		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		print("Warning: Receiver coordinates are forced!")
		XR,YR,ZR = wgs2xyz(-35.9061,-7.2122266,552.50323)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		#NEU TO XYZ
		# NEUB = neu2xyz()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])
		V_UPDW_350 = V_UPDW_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev;

		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,float("nan") );
		azim = np.where(zen_ang>cutoff_rad,azim,float("nan"))#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,float("nan"))#greater than 30 deg

		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi[1:]

def interpolate_fromsp3_IAC_new(sp3path,date,sv,cutoffelev,geolong,geolat,rx_hei,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	print("Reading:",fullfile)
	if(exists(fullfile)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		print("Warning reading GPS")
		Ts,Xs,Ys,Zs = read_GPS(sp3path,sp3name,sv)
		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)

		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(geolong,geolat,rx_hei)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		#NEU TO XYZ
		# NEUB = neu2xyz()
		Z_sat = NEU[:,2]*rang
		# pyplot.plot(Z_sat)
		# pyplot.show()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*geoh/(NEU[:,2][1:]*rang[1:])#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*geoh/(NEU[:,2][1:]*rang[1:])
		V_UPDW_350 = V_UPDW_SAT*geoh/(NEU[:,2][1:]*rang[1:])
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev;

		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,float("nan") );
		azim = np.where(zen_ang>cutoff_rad,azim,float("nan"))#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,float("nan"))#greater than 30 deg

		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi[1:]


def interpolate_fromsp3_IAC_2days_ZSAT_BACKUP(sp3path,date,sv,cutoffelev,geolong,geolat,rx_hei,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	newdate = datedata + timedelta(days=1)
	DOY_N = int(newdate.strftime('%j')) #depends on the time when this script is executed.
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	#TODO: THIS WILL NOT WORK FOR DECEMBER 31ST TO JAN 1, SICNE THE YEAR CHANGE
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	sp3name_N = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY_N)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	fullfile_N = "%s\%s"%(sp3path,sp3name_N)
	# print("Reading:",fullfile)
	# print("Reading:",fullfile_N)
	if(exists(fullfile)==False) | (exists(fullfile_N)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		# print("Warning reading GPS")
		Ts,Xs,Ys,Zs = read_GPS(sp3path,sp3name,sv)
		Ts_N,Xs_N,Ys_N,Zs_N = read_GPS(sp3path,sp3name_N,sv)
		Ts_N = np.array(Ts_N) + 86400
		#ONLY ADD THE 4 HOURS FROM THE NEXT DAY.
		Ts_N = np.where(Ts_N<=100800,Ts_N,float("nan"))
		Xs_N = np.where(Ts_N<=100800,Xs_N,float("nan"))
		Ys_N = np.where(Ts_N<=100800,Ys_N,float("nan"))
		Zs_N = np.where(Ts_N<=100800,Zs_N,float("nan"))

		Ts_N = Ts_N[~np.isnan(Ts_N)]
		Xs_N = Xs_N[~np.isnan(Xs_N)]
		Ys_N = Ys_N[~np.isnan(Ys_N)]
		Zs_N = Zs_N[~np.isnan(Zs_N)]

		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Ts = np.hstack((np.array(Ts),np.array(Ts_N)))
		Xs = np.hstack((np.array(Xs),np.array(Xs_N)))
		Ys = np.hstack((np.array(Ys),np.array(Ys_N)))
		Zs = np.hstack((np.array(Zs),np.array(Zs_N)))

		# print("1 len(Ts):",len(Ts))

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)
		# print("2 len(Tsi):",len(Tsi))
		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		# print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(geolong,geolat,rx_hei)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		# print('len(azim),len(elev),len(NEU[:,0]):',len(azim),len(elev),len(NEU[:,0]))
		# print("max elev:",np.nanmin(elev))
		#NEU TO XYZ
		# NEUB = neu2xyz()
		Z_sat = NEU[:,2]*rang
		# pyplot.plot(Z_sat)
		# pyplot.show()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*geoh/(Z_sat[1:])#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*geoh/(Z_sat[1:])#TODO: MULTIPLY BY QY/QX
		V_UPDW_350 = V_UPDW_SAT*geoh/(Z_sat[1:])#TODO: MULTIPLY BY QZ/QX
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev;
		# print("max elev:",np.nanmax(zen_ang))
		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,float("nan") );
		azim = np.where(zen_ang>cutoff_rad,azim,float("nan"))#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,float("nan"))#greater than 30 deg
		print('len(azim),len(zen_ang),len(V_NORT_350):',len(azim),len(zen_ang),len(V_NORT_350))
		# print("3 len(Tstmp):",len(Tstmp))
		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]
		# print("4 len(Tstmp):",len(Tstmp))

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi[1:],Z_sat[1:]

def interpolate_fromsp3_IAC_2days_ZSAT(sp3path,date,sv,cutoffelev,geolong,geolat,rx_hei,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	newdate = datedata + timedelta(days=1)
	DOY_N = int(newdate.strftime('%j')) #depends on the time when this script is executed.
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	#TODO: THIS WILL NOT WORK FOR DECEMBER 31ST TO JAN 1, SICNE THE YEAR CHANGE
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	sp3name_N = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY_N)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	fullfile_N = "%s\%s"%(sp3path,sp3name_N)
	# print("Reading:",fullfile)
	# print("Reading:",fullfile_N)
	if(exists(fullfile)==False) | (exists(fullfile_N)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		# print("Warning reading GPS")
		Ts,Xs,Ys,Zs = read_GPS(sp3path,sp3name,sv)
		Ts_N,Xs_N,Ys_N,Zs_N = read_GPS(sp3path,sp3name_N,sv)
		Ts_N = np.array(Ts_N) + 86400
		#ONLY ADD THE 4 HOURS FROM THE NEXT DAY.
		Ts_N = np.where(Ts_N<=100800,Ts_N,float("nan"))
		Xs_N = np.where(Ts_N<=100800,Xs_N,float("nan"))
		Ys_N = np.where(Ts_N<=100800,Ys_N,float("nan"))
		Zs_N = np.where(Ts_N<=100800,Zs_N,float("nan"))

		Ts_N = Ts_N[~np.isnan(Ts_N)]
		Xs_N = Xs_N[~np.isnan(Xs_N)]
		Ys_N = Ys_N[~np.isnan(Ys_N)]
		Zs_N = Zs_N[~np.isnan(Zs_N)]

		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Ts = np.hstack((np.array(Ts),np.array(Ts_N)))
		Xs = np.hstack((np.array(Xs),np.array(Xs_N)))
		Ys = np.hstack((np.array(Ys),np.array(Ys_N)))
		Zs = np.hstack((np.array(Zs),np.array(Zs_N)))

		# print("1 len(Ts):",len(Ts))

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)
		# print("2 len(Tsi):",len(Tsi))
		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		# print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(geolong,geolat,rx_hei)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev; #THIS IS THE REAL ELEVATION, WE NEED TO CLEAN THE CODE AND RENAMEIT
		Z_sat = NEU[:,2]*rang

		# print("max elev:",np.nanmin(elev))
		#NEU TO XYZ
		# NEUB = neu2xyz()

		# pyplot.plot(Z_sat)
		# pyplot.show()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)

		V_NORT_SAT=np.hstack((V_NORT_SAT,V_NORT_SAT[-1]))
		V_EAST_SAT=np.hstack((V_EAST_SAT,V_EAST_SAT[-1]))
		V_UPDW_SAT=np.hstack((V_UPDW_SAT,V_UPDW_SAT[-1]))
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#TODO: MULTIPLY BY QY/QX
		V_UPDW_350 = V_UPDW_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#TODO: MULTIPLY BY QZ/QX
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		# print("max elev:",np.nanmax(zen_ang))
		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,0 ); #just changed float("nan") by 0
		azim = np.where(zen_ang>cutoff_rad,azim,0)#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,0)#greater than 30 deg
		# print("3 len(Tstmp):",len(Tstmp))
		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi,Z_sat


def interpolate_fromsp3_IAC_2days_GPS(sp3path,date,sv,cutoffelev,geolong,geolat,rx_hei,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	newdate = datedata + timedelta(days=1)
	DOY_N = int(newdate.strftime('%j')) #depends on the time when this script is executed.
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	#TODO: THIS WILL NOT WORK FOR DECEMBER 31ST TO JAN 1, SICNE THE YEAR CHANGE
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	sp3name_N = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY_N)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	fullfile_N = "%s\%s"%(sp3path,sp3name_N)
	# print("Reading:",fullfile)
	# print("Reading:",fullfile_N)
	if(exists(fullfile)==False) | (exists(fullfile_N)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		# print("Warning reading GPS")
		Ts,Xs,Ys,Zs = read_GPS(sp3path,sp3name,sv)
		Ts_N,Xs_N,Ys_N,Zs_N = read_GPS(sp3path,sp3name_N,sv)
		Ts_N = np.array(Ts_N) + 86400
		#ONLY ADD THE 4 HOURS FROM THE NEXT DAY.
		Ts_N = np.where(Ts_N<=100800,Ts_N,float("nan"))
		Xs_N = np.where(Ts_N<=100800,Xs_N,float("nan"))
		Ys_N = np.where(Ts_N<=100800,Ys_N,float("nan"))
		Zs_N = np.where(Ts_N<=100800,Zs_N,float("nan"))

		Ts_N = Ts_N[~np.isnan(Ts_N)]
		Xs_N = Xs_N[~np.isnan(Xs_N)]
		Ys_N = Ys_N[~np.isnan(Ys_N)]
		Zs_N = Zs_N[~np.isnan(Zs_N)]

		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Ts = np.hstack((np.array(Ts),np.array(Ts_N)))
		Xs = np.hstack((np.array(Xs),np.array(Xs_N)))
		Ys = np.hstack((np.array(Ys),np.array(Ys_N)))
		Zs = np.hstack((np.array(Zs),np.array(Zs_N)))

		# print("1 len(Ts):",len(Ts))

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)
		# print("2 len(Tsi):",len(Tsi))
		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		# print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(geolong,geolat,rx_hei)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev; #THIS IS THE REAL ELEVATION, WE NEED TO CLEAN THE CODE AND RENAMEIT
		Z_sat = NEU[:,2]*rang

		# print("max elev:",np.nanmin(elev))
		#NEU TO XYZ
		# NEUB = neu2xyz()

		# pyplot.plot(Z_sat)
		# pyplot.show()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)

		V_NORT_SAT=np.hstack((V_NORT_SAT,V_NORT_SAT[-1]))
		V_EAST_SAT=np.hstack((V_EAST_SAT,V_EAST_SAT[-1]))
		V_UPDW_SAT=np.hstack((V_UPDW_SAT,V_UPDW_SAT[-1]))
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#TODO: MULTIPLY BY QY/QX
		V_UPDW_350 = V_UPDW_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#TODO: MULTIPLY BY QZ/QX
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		# print("max elev:",np.nanmax(zen_ang))
		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,0 ); #just changed float("nan") by 0
		azim = np.where(zen_ang>cutoff_rad,azim,0)#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,0)#greater than 30 deg
		# print("3 len(Tstmp):",len(Tstmp))
		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi,Z_sat

def interpolate_fromsp3_IAC_2days_rang2(sp3path,date,sv,cutoffelev,geolong,geolat,rx_hei,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	newdate = datedata + timedelta(days=1)
	DOY_N = int(newdate.strftime('%j')) #depends on the time when this script is executed.
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	#TODO: THIS WILL NOT WORK FOR DECEMBER 31ST TO JAN 1, SICNE THE YEAR CHANGE

	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	sp3name_N = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY_N)
	print(sp3name)
	print(sp3name_N)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	fullfile_N = "%s\%s"%(sp3path,sp3name_N)
	# print("Reading:",fullfile)
	# print("Reading:",fullfile_N)
	if(exists(fullfile)==False) | (exists(fullfile_N)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		# print("Warning reading GPS")
		Ts,Xs,Ys,Zs = read_GPS(sp3path,sp3name,sv)
		Ts_N,Xs_N,Ys_N,Zs_N = read_GPS(sp3path,sp3name_N,sv)
		Ts_N = np.array(Ts_N) + 86400
		#ONLY ADD THE 4 HOURS FROM THE NEXT DAY.
		Ts_N = np.where(Ts_N<=100800,Ts_N,float("nan"))
		Xs_N = np.where(Ts_N<=100800,Xs_N,float("nan"))
		Ys_N = np.where(Ts_N<=100800,Ys_N,float("nan"))
		Zs_N = np.where(Ts_N<=100800,Zs_N,float("nan"))

		Ts_N = Ts_N[~np.isnan(Ts_N)]
		Xs_N = Xs_N[~np.isnan(Xs_N)]
		Ys_N = Ys_N[~np.isnan(Ys_N)]
		Zs_N = Zs_N[~np.isnan(Zs_N)]

		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Ts = np.hstack((np.array(Ts),np.array(Ts_N)))
		Xs = np.hstack((np.array(Xs),np.array(Xs_N)))
		Ys = np.hstack((np.array(Ys),np.array(Ys_N)))
		Zs = np.hstack((np.array(Zs),np.array(Zs_N)))

		# print("1 len(Ts):",len(Ts))

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)
		# print("2 len(Tsi):",len(Tsi))
		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		# print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(geolong,geolat,rx_hei)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		return Tsi,rang

def interpolate_fromsp3_IAC_2days_GLO(sp3path,date,sv,cutoffelev,geolong,geolat,rx_hei,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	newdate = datedata + timedelta(days=1)
	DOY_N = int(newdate.strftime('%j')) #depends on the time when this script is executed.
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	#TODO: THIS WILL NOT WORK FOR DECEMBER 31ST TO JAN 1, SICNE THE YEAR CHANGE
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	sp3name_N = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY_N)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	fullfile_N = "%s\%s"%(sp3path,sp3name_N)
	# print("Reading:",fullfile)
	# print("Reading:",fullfile_N)
	if(exists(fullfile)==False) | (exists(fullfile_N)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		# print("Warning reading GPS")
		Ts,Xs,Ys,Zs = read_GLO(sp3path,sp3name,sv)
		Ts_N,Xs_N,Ys_N,Zs_N = read_GLO(sp3path,sp3name_N,sv)
		Ts_N = np.array(Ts_N) + 86400
		#ONLY ADD THE 4 HOURS FROM THE NEXT DAY.
		Ts_N = np.where(Ts_N<=100800,Ts_N,float("nan"))
		Xs_N = np.where(Ts_N<=100800,Xs_N,float("nan"))
		Ys_N = np.where(Ts_N<=100800,Ys_N,float("nan"))
		Zs_N = np.where(Ts_N<=100800,Zs_N,float("nan"))

		Ts_N = Ts_N[~np.isnan(Ts_N)]
		Xs_N = Xs_N[~np.isnan(Xs_N)]
		Ys_N = Ys_N[~np.isnan(Ys_N)]
		Zs_N = Zs_N[~np.isnan(Zs_N)]

		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Ts = np.hstack((np.array(Ts),np.array(Ts_N)))
		Xs = np.hstack((np.array(Xs),np.array(Xs_N)))
		Ys = np.hstack((np.array(Ys),np.array(Ys_N)))
		Zs = np.hstack((np.array(Zs),np.array(Zs_N)))

		# print("1 len(Ts):",len(Ts))

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)
		# print("2 len(Tsi):",len(Tsi))
		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		# print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(geolong,geolat,rx_hei)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev; #THIS IS THE REAL ELEVATION, WE NEED TO CLEAN THE CODE AND RENAMEIT
		Z_sat = NEU[:,2]*rang

		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)

		V_NORT_SAT=np.hstack((V_NORT_SAT,V_NORT_SAT[-1]))
		V_EAST_SAT=np.hstack((V_EAST_SAT,V_EAST_SAT[-1]))
		V_UPDW_SAT=np.hstack((V_UPDW_SAT,V_UPDW_SAT[-1]))
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#TODO: MULTIPLY BY QY/QX
		V_UPDW_350 = V_UPDW_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#TODO: MULTIPLY BY QZ/QX

		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,0 ); #just changed float("nan") by 0
		azim = np.where(zen_ang>cutoff_rad,azim,0)#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,0)#greater than 30 deg

		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi,Z_sat


def interpolate_fromsp3_IAC_2days_BDS(sp3path,date,sv,cutoffelev,geolong,geolat,rx_hei,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	newdate = datedata + timedelta(days=1)
	DOY_N = int(newdate.strftime('%j')) #depends on the time when this script is executed.
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	#TODO: THIS WILL NOT WORK FOR DECEMBER 31ST TO JAN 1, SICNE THE YEAR CHANGE
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	sp3name_N = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY_N)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	fullfile_N = "%s\%s"%(sp3path,sp3name_N)
	# print("Reading:",fullfile)
	# print("Reading:",fullfile_N)
	if(exists(fullfile)==False) | (exists(fullfile_N)==False):
		#if not here download!
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		# print("Warning reading GPS")
		Ts,Xs,Ys,Zs = read_BDS(sp3path,sp3name,sv)
		Ts_N,Xs_N,Ys_N,Zs_N = read_BDS(sp3path,sp3name_N,sv)
		Ts_N = np.array(Ts_N) + 86400
		#ONLY ADD THE 4 HOURS FROM THE NEXT DAY.
		Ts_N = np.where(Ts_N<=100800,Ts_N,float("nan"))
		Xs_N = np.where(Ts_N<=100800,Xs_N,float("nan"))
		Ys_N = np.where(Ts_N<=100800,Ys_N,float("nan"))
		Zs_N = np.where(Ts_N<=100800,Zs_N,float("nan"))

		Ts_N = Ts_N[~np.isnan(Ts_N)]
		Xs_N = Xs_N[~np.isnan(Xs_N)]
		Ys_N = Ys_N[~np.isnan(Ys_N)]
		Zs_N = Zs_N[~np.isnan(Zs_N)]

		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Ts = np.hstack((np.array(Ts),np.array(Ts_N)))
		Xs = np.hstack((np.array(Xs),np.array(Xs_N)))
		Ys = np.hstack((np.array(Ys),np.array(Ys_N)))
		Zs = np.hstack((np.array(Zs),np.array(Zs_N)))

		# print("1 len(Ts):",len(Ts))

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)
		# print("2 len(Tsi):",len(Tsi))
		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		# print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(geolong,geolat,rx_hei)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev; #THIS IS THE REAL ELEVATION, WE NEED TO CLEAN THE CODE AND RENAMEIT
		Z_sat = NEU[:,2]*rang

		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)

		V_NORT_SAT=np.hstack((V_NORT_SAT,V_NORT_SAT[-1]))
		V_EAST_SAT=np.hstack((V_EAST_SAT,V_EAST_SAT[-1]))
		V_UPDW_SAT=np.hstack((V_UPDW_SAT,V_UPDW_SAT[-1]))
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#TODO: MULTIPLY BY QY/QX
		V_UPDW_350 = V_UPDW_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#TODO: MULTIPLY BY QZ/QX

		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,0 ); #just changed float("nan") by 0
		azim = np.where(zen_ang>cutoff_rad,azim,0)#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,0)#greater than 30 deg

		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi,Z_sat

def interpolate_fromsp3_IAC_2days_GAL(sp3path,date,sv,cutoffelev,geolong,geolat,rx_hei,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	newdate = datedata + timedelta(days=1)
	DOY_N = int(newdate.strftime('%j')) #depends on the time when this script is executed.
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	#TODO: THIS WILL NOT WORK FOR DECEMBER 31ST TO JAN 1, SICNE THE YEAR CHANGE
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	sp3name_N = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY_N)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	fullfile_N = "%s\%s"%(sp3path,sp3name_N)
	# print("Reading:",fullfile)
	# print("Reading:",fullfile_N)
	if(exists(fullfile)==False) | (exists(fullfile_N)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		# print("Warning reading GPS")
		Ts,Xs,Ys,Zs = read_GAL(sp3path,sp3name,sv)
		Ts_N,Xs_N,Ys_N,Zs_N = read_GAL(sp3path,sp3name_N,sv)
		Ts_N = np.array(Ts_N) + 86400
		#ONLY ADD THE 4 HOURS FROM THE NEXT DAY.
		Ts_N = np.where(Ts_N<=100800,Ts_N,float("nan"))
		Xs_N = np.where(Ts_N<=100800,Xs_N,float("nan"))
		Ys_N = np.where(Ts_N<=100800,Ys_N,float("nan"))
		Zs_N = np.where(Ts_N<=100800,Zs_N,float("nan"))

		Ts_N = Ts_N[~np.isnan(Ts_N)]
		Xs_N = Xs_N[~np.isnan(Xs_N)]
		Ys_N = Ys_N[~np.isnan(Ys_N)]
		Zs_N = Zs_N[~np.isnan(Zs_N)]

		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Ts = np.hstack((np.array(Ts),np.array(Ts_N)))
		Xs = np.hstack((np.array(Xs),np.array(Xs_N)))
		Ys = np.hstack((np.array(Ys),np.array(Ys_N)))
		Zs = np.hstack((np.array(Zs),np.array(Zs_N)))

		# print("1 len(Ts):",len(Ts))

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)
		# print("2 len(Tsi):",len(Tsi))
		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		# print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(geolong,geolat,rx_hei)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev; #THIS IS THE REAL ELEVATION, WE NEED TO CLEAN THE CODE AND RENAMEIT
		Z_sat = NEU[:,2]*rang

		#SUPER WARNING, IS THIS OR ediff1d
		print("try:ediff1d insead of diff")
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)

		V_NORT_SAT=np.hstack((V_NORT_SAT,V_NORT_SAT[-1]))
		V_EAST_SAT=np.hstack((V_EAST_SAT,V_EAST_SAT[-1]))
		V_UPDW_SAT=np.hstack((V_UPDW_SAT,V_UPDW_SAT[-1]))
		#TODO: get the real height at 350 from the ground, no radial

		V_NORT_350 = V_NORT_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#TODO: MULTIPLY BY QY/QX
		V_UPDW_350 = V_UPDW_SAT*geoh/(Z_sat-np.ones(len(Z_sat))*geoh)#TODO: MULTIPLY BY QZ/QX

		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,0 ); #just changed float("nan") by 0
		azim = np.where(zen_ang>cutoff_rad,azim,0)#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,0)#greater than 30 deg

		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi,Z_sat

def interpolate_fromsp3_IAC_2days(sp3path,date,sv,cutoffelev,geolong,geolat,rx_hei,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	newdate = datedata + timedelta(days=1)
	DOY_N = int(newdate.strftime('%j')) #depends on the time when this script is executed.
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	#TODO: THIS WILL NOT WORK FOR DECEMBER 31ST TO JAN 1, SICNE THE YEAR CHANGE
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	sp3name_N = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY_N)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	fullfile_N = "%s\%s"%(sp3path,sp3name_N)
	print("Reading:",fullfile)
	print("Reading:",fullfile_N)
	if(exists(fullfile)==False) | (exists(fullfile_N)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		print("Warning reading GPS")
		Ts,Xs,Ys,Zs = read_GPS(sp3path,sp3name,sv)
		Ts_N,Xs_N,Ys_N,Zs_N = read_GPS(sp3path,sp3name_N,sv)
		Ts_N = np.array(Ts_N) + 86400
		#ONLY ADD THE 4 HOURS FROM THE NEXT DAY.
		Ts_N = np.where(Ts_N<=100800,Ts_N,float("nan"))
		Xs_N = np.where(Ts_N<=100800,Xs_N,float("nan"))
		Ys_N = np.where(Ts_N<=100800,Ys_N,float("nan"))
		Zs_N = np.where(Ts_N<=100800,Zs_N,float("nan"))

		Ts_N = Ts_N[~np.isnan(Ts_N)]
		Xs_N = Xs_N[~np.isnan(Xs_N)]
		Ys_N = Ys_N[~np.isnan(Ys_N)]
		Zs_N = Zs_N[~np.isnan(Zs_N)]

		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Ts = np.hstack((np.array(Ts),np.array(Ts_N)))
		Xs = np.hstack((np.array(Xs),np.array(Xs_N)))
		Ys = np.hstack((np.array(Ys),np.array(Ys_N)))
		Zs = np.hstack((np.array(Zs),np.array(Zs_N)))

		# print("1 len(Ts):",len(Ts))

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)
		# print("2 len(Tsi):",len(Tsi))
		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(geolong,geolat,rx_hei)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		# print("max elev:",np.nanmin(elev))
		#NEU TO XYZ
		# NEUB = neu2xyz()
		Z_sat = NEU[:,2]*rang
		# pyplot.plot(Z_sat)
		# pyplot.show()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*geoh/(Z_sat[1:])#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*geoh/(Z_sat[1:])#TODO: MULTIPLY BY QY/QX
		V_UPDW_350 = V_UPDW_SAT*geoh/(Z_sat[1:])#TODO: MULTIPLY BY QZ/QX
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev;
		# print("max elev:",np.nanmax(zen_ang))
		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,float("nan") );
		azim = np.where(zen_ang>cutoff_rad,azim,float("nan"))#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,float("nan"))#greater than 30 deg
		# print("3 len(Tstmp):",len(Tstmp))
		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]
		# print("4 len(Tstmp):",len(Tstmp))

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi[1:]

def interpolate_fromsp3_IAC_GLO(sp3path,date,sv,cutoffelev,geolong,geolat,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	print("Reading:",fullfile)
	if(exists(fullfile)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		print("Warning reading GLO")
		Ts,Xs,Ys,Zs = read_GLO(sp3path,sp3name,sv)
		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)

		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(-35.9061,-7.2122266,552.50323)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		#NEU TO XYZ
		# NEUB = neu2xyz()
		Z_sat = NEU[:,2]*rang
		# pyplot.plot(Z_sat)
		# pyplot.show()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])
		V_UPDW_350 = V_UPDW_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev;

		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,float("nan") );
		azim = np.where(zen_ang>cutoff_rad,azim,float("nan"))#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,float("nan"))#greater than 30 deg

		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi[1:]

def interpolate_fromsp3_IAC_GAL(sp3path,date,sv,cutoffelev,geolong,geolat,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	print("Reading:",fullfile)
	if(exists(fullfile)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		print("Warning reading GAL")
		Ts,Xs,Ys,Zs = read_GAL(sp3path,sp3name,sv)
		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)

		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(-35.9061,-7.2122266,552.50323)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		#NEU TO XYZ
		# NEUB = neu2xyz()
		Z_sat = NEU[:,2]*rang
		# pyplot.plot(Z_sat)
		# pyplot.show()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])
		V_UPDW_350 = V_UPDW_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev;

		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,float("nan") );
		azim = np.where(zen_ang>cutoff_rad,azim,float("nan"))#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,float("nan"))#greater than 30 deg

		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi[1:]

def interpolate_fromsp3_IAC_BDS(sp3path,date,sv,cutoffelev,geolong,geolat,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))
	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	sp3name = 'IAC0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	print("Reading:",fullfile)
	if(exists(fullfile)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		print("Warning reading BDS")
		Ts,Xs,Ys,Zs = read_BDS(sp3path,sp3name,sv)
		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)

		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		print("Warning: Receiver coordinates are forced to PARAIBA!")
		XR,YR,ZR = wgs2xyz(-35.9061,-7.2122266,552.50323)
		# XR,YR,ZR = wgs2xyz(-96.7572,32.9918,80)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		#NEU TO XYZ
		# NEUB = neu2xyz()
		Z_sat = NEU[:,2]*rang
		# pyplot.plot(Z_sat)
		# pyplot.show()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*geoh*1e3/(NEU[:,2][1:]*rang[1:])#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*geoh*1e3/(NEU[:,2][1:]*rang[1:])
		V_UPDW_350 = V_UPDW_SAT*geoh*1e3/(NEU[:,2][1:]*rang[1:])
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev;

		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,float("nan") );
		azim = np.where(zen_ang>cutoff_rad,azim,float("nan"))#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,float("nan"))#greater than 30 deg

		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi[1:]


def interpolate_fromsp3_JAX_new(sp3path,date,sv,cutoffelev,geolong,geolat,geoh,cross_corr_space):
	#TODO: INCLUDE ONE POINT FROM NEXT FILE
	datedata = datetime.date(int(date[0:4]),int(date[4:6]),int(date[6:8]) )
	year = int(date[0:4])
	DOY = int(datedata.strftime('%j'))

	# weekn,dow = gnsscal.date2gpswd(datedata)
	# sp3name = 'igr%04d%01d.sp3'%(weekn,dow)
	sp3name = 'JAX0MGXFIN_%04d%03d0000_01D_05M_ORB.SP3'%(year,DOY)
	# print(year,DOY,sp3name)
	fullfile = "%s\%s"%(sp3path,sp3name)
	# print(fullfile)
	if(exists(fullfile)==False):
		return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
	else:
		dt = np.dtype('float64')
		print("Reading gps from SP3")
		Ts,Xs,Ys,Zs = read_GPS(sp3path,sp3name,sv)
		# Ts,Xs,Ys,Zs = read_GAL(sp3path,sp3name,sv)
		if len(np.array(Xs))==0:
			return [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

		Xs = np.array(Xs, dtype=dt)
		Ys = np.array(Ys, dtype=dt)
		Zs = np.array(Zs, dtype=dt)

		Tsi = np.arange(np.nanmin(Ts),np.nanmax(Ts),cross_corr_space)

		Xsif= scipy.interpolate.interp1d(Ts,Xs,kind='cubic')
		Ysif= scipy.interpolate.interp1d(Ts,Ys,kind='cubic')
		Zsif= scipy.interpolate.interp1d(Ts,Zs,kind='cubic')
		Xsi = Xsif(Tsi)
		Ysi = Ysif(Tsi)
		Zsi = Zsif(Tsi)

		S = np.array([Xsi*1000.0, Ysi*1000.0, Zsi*1000.0]);
		#RECEUVER COORDINATES ARE FORCED
		print("Warning: Receiver coordinates are forced!")
		XR,YR,ZR = wgs2xyz(-35.9061,-7.2122266,552.50323)
		azim,elev,hlen,rang,NEU= azelle(Xsi*1000, Ysi*1000, Zsi*1000,XR,YR,ZR)
		#NEU TO XYZ
		# NEUB = neu2xyz()
		Z_sat = NEU[:,2]*rang
		# pyplot.plot(Z_sat)
		# pyplot.show()
		V_NORT_SAT = np.diff(NEU[:,0]*rang)/np.diff(Tsi)
		V_EAST_SAT = np.diff(NEU[:,1]*rang)/np.diff(Tsi)
		V_UPDW_SAT = np.diff(NEU[:,2]*rang)/np.diff(Tsi)
		#TODO: get the real height at 350 from the ground, no radial
		V_NORT_350 = V_NORT_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])#-np.ones(len(NEU[:,2][1:]))*350e3
		V_EAST_350 = V_EAST_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])
		V_UPDW_350 = V_UPDW_SAT*350.0e3/(NEU[:,2][1:]*rang[1:])
		# pyplot.plot(Tsi[1:],V_NORT_350,'^k')
		# pyplot.plot(Tsi[1:],V_EAST_350,'or')
		# pyplot.show()
		zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev;

		cutoff=5.0
		cutoff_rad = cutoff * np.pi/180.0;
		zen_ang = np.where(zen_ang>cutoff_rad,zen_ang,float("nan") );
		azim = np.where(zen_ang>cutoff_rad,azim,float("nan"))#greater than 30 deg
		azim = np.where(azim<0,azim+2*np.pi,azim)
		Tstmp = np.where(zen_ang>cutoff_rad,Tsi,float("nan"))#greater than 30 deg

		# pyplot.plot(Tsi,azim,'--ok')
		azim = azim[~np.isnan(azim)]
		zen_ang = zen_ang[~np.isnan(zen_ang)]
		Tstmp= Tstmp[~np.isnan(Tstmp)]

		return Tstmp,zen_ang,azim,V_NORT_350,V_EAST_350,V_UPDW_350,Tsi[1:]
