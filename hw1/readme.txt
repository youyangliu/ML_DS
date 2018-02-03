Team members: Chang Ding, cd2959, Zhaoyue Bi, zb2225, Youyang Liu, yl3767

"hw1_p1.py": The source code of hw1 problem 1. 
	Some main variables:
		- n_list: the list which stores n’s value
			eg. [1000, 2000, 4000, 8000]

		- test_result/train_result: the dict which stores the running result of 10 times, the key is n, the value is a list of error rate of each prediction.
			eg. {1000: array([ 0.1202,  0.1182,  0.1188,  0.1154,  0.1146,  0.1176,  0.1226,
         0.116 ,  0.1165,  0.1115]),
 2000: array([ 0.0841,  0.088 ,  0.0878,  0.088 ,  0.0855,  0.0888,  0.0921,
         0.0827,  0.0925,  0.0919]),
 4000: array([ 0.07  ,  0.0694,  0.0699,  0.0668,  0.0708,  0.0738,  0.0683,
         0.0716,  0.0684,  0.0695]),
 8000: array([ 0.0537,  0.0565,  0.0551,  0.0551,  0.0521,  0.0536,  0.0569,
         0.0527,  0.0553,  0.0586])}

		- test_n_mean/train_n_mean: the list which stores the error mean of each n
			eg. [0.11713999999999999, 0.08814000000000001, 0.069849999999999995, 0.054959999999999995]
		- test_n_std/train_n_std: the list which stores the error standard deviation of each n
			eg. [0.0029370733732748317, 0.0031984996482726088, 0.0018364367672206966, 0.0019043108989868229]

"hw_1p1.ipynb": Show the plot and variables result in hw1_p1.py.
