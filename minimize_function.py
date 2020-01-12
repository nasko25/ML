import scipy.optimize as optimize
import math

def function(params):
	print(params)
	x, y, z = params
	return math.log((z-2)**2+4*math.sin(z-2)+7) - 1/((x-y-9)**4 + (y-7)**2 + 8)

guess = [1, 1, 1]
result = optimize.minimize(function, guess)
if result.success:
	fitted_params = result.x
	print("\n\nMinimal value for the function is: ", function(fitted_params))
else:
	raise ValueError(result.message)
