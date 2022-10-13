"""
Goal: Metropolis evalution of I = \int_{[a,b]} (sin(pi*X * X/2 ))^2 * exp[ - X * X  / (sigma)], 4-dimension 
The expected values with Sigma = 0.1, 1, 10 are [0.0125, 5.35, 493] for [a=-inf, b=inf]
f(X) = (sin(pi*X * X/2 ))^2 * exp[ - X * X  / (sigma)]]
P(X) = exp[ - X * X  / (sigma)]
g(X) =(sin(pi*X * X/2 ))^2 
"""
import numpy as np 	# mostly for quicker operations
import pylab as pl  # useful to plot stuff!
#import time
import math as mt 	# basic math stuff

## number of evaluations
cyc = 100

## number of warm-up and measurement cycles
Nwup = 3000
Nmeas = 10000

## parameter
sigma = [0.1, 1.0, 10.0]
print('Below are the results of the integral {-inf,+inf}(sin(pi*X * X/2 ))^2 * exp[ - X * X  / (sigma)] with Nmeas=10000')
print('___________________________________________________________________________________________________________________')
for s in range (0,3):
	Sval = sigma[s] # sigma Value

## true result
	Ival0 = 0.5*((Sval*np.pi)**4)*(((Sval*np.pi)**2)+3)/((((Sval*np.pi)**2)+1)**2)
#Ival4 = 0.5*(np.pi**2)*np.exp(-8/Sval)*(-16*Sval-2*Sval**2-(16*Sval/(Sval*np.pi)**2)+((2*Sval**2)*(1-(Sval*np.pi)**2)/((1+(Sval*np.pi)**2)**2)))
#print('Ival4 =',Ival4)
## normalization factor due to \int dX P(X) = 1
	Prenorm = (np.pi*Sval)**2 #noralization of probabaility
	Ival = Prenorm
## Initial guess
	X0 = Sval * np.random.randn((4))
## Displacement value
	h0 = 0.1
## Define here all functions we need
	def g(XX):
		sq2 = (np.sin(0.5*np.pi*(XX.dot(XX))))**2
		return sq2
	def P(XX): #prob. function
		sq = (XX.dot(XX))	
		ValP = mt.exp( - sq / Sval)
		return ValP

## defines if we update or not
	def Update(Y0, Y1):
		Ratio = P(Y1)/P(Y0)
		AXX = min(1., Ratio)
		if AXX >= 1.0:
			return Y1, True
		else:
			rr = np.random.random()
			if rr > AXX:
				return Y0, False
			else:
				return Y1, True

## This defines the Markov chain cycle
## It takes one initial value, tries CC updates, and return a new one

	def MC_Cycle(Z0, CC):
	## CC is the number of updates here.
		Xin = Z0
		for j in range(CC):
			Xs = Xin + h0 * Sval * np.random.randn((4))
		## draw 10 random numbers, and displace all 10 variables
			Jump, Pr = Update(Xin, Xs)
			Xin = Jump
		return Xin

##
## Step 2: Perform integration
##

# vector of values for $f(X)$ 
	AA = []

# sum of the values
	BB = 0

# average values at each step - to show convergence
	IntRes = []

# vector of standard deviations
	sigma_vec = []

## Warm-up procedure
	Xwp = MC_Cycle(X0, Nwup*cyc)

## Measurements
	Xstart = Xwp

	for j in range(Nmeas):
		Xval = MC_Cycle(Xstart, cyc)
	#Meas = 1.0
		Meas = g(Xval)
		Xstart = Xval
	## storing the value of $g(X)$
		AA.append(Meas)    
	## add this to the 'total' function
		BB += Meas
	## new average value
		IntRes.append( (BB/(j+1)))
	## standard deviations calculated and stored
		stand = np.std(AA / np.sqrt(j+1))
		sigma_vec.append(stand)

	AARes = np.array(AA)

	StepsRes = np.array(IntRes)

	sigma_vec2 = np.array(sigma_vec)

	Res = ( np.sum(AARes)/cyc ) / (Ival)

	print('For sigma =', Sval, ';','Exact Value = ', Ival0, ';','Approx value= ', Ival*StepsRes[Nmeas-1], ';','Error = ',np.abs(Ival0-Ival*StepsRes[Nmeas-1]), ';','Relative Error = ',np.abs((Ival0-Ival*StepsRes[Nmeas-1])/Ival0))
print('Coments: ','Relative errors for differente sigmas are of the same order of magnitude')

