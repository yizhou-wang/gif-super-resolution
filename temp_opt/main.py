if __name__ == '__main__':

	'''
	Step 1: Read images.
		'lr_gif': 	read lr GIF in a array (frame X 8 X 8)
	'''

	'''
	Step 2: BI on each frame.
		'bi_gif': 	bicubic interpolation on each frame (frame X 32 X 32)
		'alpha':	coefficients of interpolation function (frame X 32 X 32)
					(each alpha: 4 X 4, 8 X 8 alphas for each frame)
	'''

	'''
	Step 3: Optimization.
		- Compute cost = BI cost + TR cost
		- Gradient descent
		- Next iteration
	'''


