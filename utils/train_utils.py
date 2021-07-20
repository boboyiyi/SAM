import numpy as np


def aggregate_loss_dict(agg_loss_dict):
	mean_vals = {}
	for output in agg_loss_dict:
		for key in output:
			# Python 字典 setdefault() 函数和 get()方法 类似, 如果键不存在于字典中，将会添加键并将值设为默认值。
			mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
	for key in mean_vals:
		if len(mean_vals[key]) > 0:
			mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
		else:
			print(f'{key} has no value')
			mean_vals[key] = 0
	return mean_vals


def compute_cosine_weights(x):
	""" Computes weights to be used in the id loss function with minimum value of 0.5 and maximum value of 1. """
	values = np.abs(x.cpu().detach().numpy())
	assert np.min(values) >= 0. and np.max(values) <= 1., "Input values should be between 0. and 1!"
	weights = 0.25 * (np.cos(np.pi * values)) + 0.75
	return weights

