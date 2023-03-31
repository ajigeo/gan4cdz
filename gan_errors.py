from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio

import numpy as np
#import seaborn as sns
from osgeo import gdal
def read_image(filename):
    image = gdal.Open(filename)
    #image_transform = image.GetGeoTransform()
    #image_projection = image.GetProjectionRef()
    image_array = image.ReadAsArray()
    #image_shape = image_array.shape()
    return image_array
#%%
#x = np.linspace(0,10)
#y = 3*np.random.randn(50) + x
#X = sm.add_constant(x)
#res = sm.OLS(y, X).fit()

from numpy import sum as arraysum
def compute_errors(x,y):
    sum_errs = arraysum((x - y)**2)
    stdev = np.sqrt(1/(len(y)-2) * sum_errs)
    p_interval = 1.96 * stdev #95% 
    
    
    # Confidence interval
    # https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
    mse = mean_squared_error(x,y,squared=False)
    mae = mean_absolute_error(x,y)
    psnr = peak_signal_noise_ratio(x,y)
    mse_interval = 1.96 * np.sqrt((mse * (1 - mse)) / len(y))
    mae_interval = 1.96 * np.sqrt((mae * (1 - mae)) / len(y))
    #psnr_interval = 1.96 * np.sqrt((psnr * (1 - psnr)) / len(y))
    
    return str(round(mse,3)) + "±"+ str(round(mse_interval,3)), str(round(mae,3)) + "±"+ str(round(mae_interval,3)),str(round(psnr,3)),str(round((np.corrcoef(x,y)[0][1]),3)),str(round(p_interval,3))
    #print("The RMSE of the model @ 95% confidence interval is " + str(round(mse,3)) + "±"+ str(round(mse_interval,3)))
    #print("The MAE of the model @ 95% confidence interval is " + str(round(mae,3)) + "±"+ str(round(mae_interval,3)))
    #print("The PSNR of the model @ 95% confidence interval is " + str(round(psnr,3)))
    #print("Correlation is " + str(round((np.corrcoef(x,y)[0][1]),3)))
    #print("Theres a 95% likelihood that the true value is between +/- " +str(round(p_interval,3)))
#%%
import glob
pred_list = glob.glob('E:/data/ISRO-project/intermediate_products/VH_predicted_NDVI/rescaled_NDVI/norm_pred_*.tif')[:-1]
act_list = glob.glob('E:/data/ISRO-project/intermediate_products/VH_predicted_NDVI/rescaled_NDVI/act_*.tif')[:-1]

x = read_image(act_list[11])
x = np.asarray(x).ravel()

vh_y = read_image(pred_list[11])
vh_y = np.asarray(vh_y).ravel()
compute_errors(x,vh_y)

metrics_list = []
for i in range(12):
    x = read_image(act_list[i])
    x = np.asarray(x).ravel()
    
    vh_y = read_image(pred_list[i])
    vh_y = np.asarray(vh_y).ravel()
    metrics = compute_errors(x,vh_y)
    metrics_list.append(metrics)
    print("***************")

'''
def xc(a,b):
    print(np.matmul(a,b) / np.sqrt(np.sum(np.square(a)) * np.sum(np.square(b))))

xc(x,vv_y)
'''
#vh_y = read_image(r'E:\data\ISRO-project\intermediate_products\VH_predicted_NDVI\pred_12.tif')
#vh_y = np.asarray(vh_y).ravel()


# Prediction interval
# https://machinelearningmastery.com/prediction-intervals-for-machine-learning/


    
'''
A confidence interval is different from a tolerance interval that describes the bounds of data sampled from the distribution. 
It is also different from a prediction interval that describes the bounds on a single observation. 
Instead, the confidence interval provides bounds on a population parameter, such as a mean, standard deviation, or similar.
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
 
 
# VH errors
x = np.arange(1,13)
y1_vh = np.array([0.4124,0.84173,0.51484,0.10983,0.40876,0.41284,0.52695,0.40809,0.37002,0.41266,0.49665,0.51216])
y2_vh = np.array([0.62522,0.76512,0.51763,0.19303,0.43448,0.44742,0.65637,0.46985,0.6604,0.72288,0.44539,0.46113])
y3_vh = np.array([0.58891,	0.55369,0.62717,0.3358,0.48019,	0.55051,	0.6394,	0.54458,	0.59896,	0.70515,	0.75597,	0.6656])
y4_vh = np.array([0.60932,	0.66379,0.43503,	0.27346,	0.45331,	0.45779,	0.60929,	0.46077,	0.43335,	0.5885,	0.41007,	0.47045])
y5_vh = np.array([0.56124,	0.59359,0.518,	0.26847,	0.38512,	0.45205,	0.59728,	0.48876,	0.49527,	0.4574,	0.58739,	0.51079])
y6_vh = np.array([0.61454,	0.86809,0.59527,	0.49525,	0.56569,	0.5143,	0.5901,	0.51366,	0.56048,	0.79072,	0.47892,	0.52486])
y7_vh = np.array([0.5689,	0.75806,0.49041,	0.16188,	0.39729,	0.47352,	0.59303,	0.50696,	0.56255,	0.73363,	0.42894,	0.51859])
y8_vh = np.array([0.57831,	0.51219,	0.52674,	0.12989,	0.37571,	0.47253,	0.57446,	0.45843,	0.55072,	0.64931,	0.48344,	0.56328])
y9_vh = np.array([0.55225,	0.65951,	0.51754,	0.21619,	0.4255,	0.43788,	0.60648,	0.44265,	0.60525,	0.42929,	0.73496,	0.51965])
y10_vh = np.array([0.58942,	0.59707,	0.4993,	0.1971,	0.39696,	0.41901,	0.57021,	0.55934,	0.50935,	0.44468,	0.66754,	0.51132])
# defining our error
y_error_vh = np.array([0.384,0.302,0.208,0.347,0.405,0.284,0.518,0.398,0.550,0.368,0.486,0.760])
#%%
# VH errors

y1_vv = np.array([0.65692,	0.73402,	0.6857,	0.65899,	0.65577,	0.6605,	0.62021,	0.72125,	0.47783,	0.65139,	0.77855,	0.70135])
y2_vv = np.array([0.69762,	0.66768,	0.58878,	0.59835,	0.60348,	0.56343,	0.57429,	0.69304,	0.70755,	0.585,	0.51674,	0.5913])
y3_vv = np.array([0.69978,	0.59619,	0.63285,	0.74417,	0.73386,	0.71239,	0.70586,	0.71532,	0.61353,	0.68795,	0.85108,	0.61352])
y4_vv = np.array([0.65868,	0.6048,	0.71262,	0.60371,	0.71964,	0.70503,	0.62094,	0.73199,	0.65343,	0.64868,	0.50163,	0.5908])
y5_vv = np.array([0.70514,	0.64873,	0.64815,	0.63292,	0.61018,	0.60604,	0.70288,	0.65666,	0.61891,	0.48841,	0.66838,	0.61837])
y6_vv = np.array([0.72959,	0.65365,	0.70385,	0.72908,	0.65124,	0.65289,	0.66343,	0.73633,	0.65103,	0.64104,	0.50608,	0.56784])
y7_vv = np.array([0.57323,	0.7461,	0.59292,	0.62223,	0.60231,	0.57734,	0.66071,	0.75028,	0.6711,	0.72777,	0.75792,	0.64571])
y8_vv = np.array([0.7155,	0.62818,	0.56612,	0.65733,	0.60036,	0.6133,	0.63963,	0.6039,	0.693,	0.57926,	0.58323,	0.71554])
y9_vv = np.array([0.671,	0.71379,	0.64881,	0.71357,	0.59384,	0.62424,	0.64405,	0.59428,	0.62293,	0.53812,	0.6752,	0.64174])
y10_vv = np.array([0.72792,	0.61915,	0.6356,	0.72627,	0.64554,	0.584,	0.63748,	0.59899,	0.61026,	0.52238,	0.71007,	0.69491])
# defining our error
y_error_vv = np.array([0.590544817205121,	0.302588879835678,	0.370155694783363,	0.586432848457724,	0.655974279381866,	0.45308466738304,	0.498016916792174,	0.612729455258196,	0.667873079137876,	0.353962672231628,	0.592107856103136,	0.909909866711467])
#%%
# https://www.geeksforgeeks.org/errorbar-graph-in-python-using-matplotlib/
l = 3
b = 4
plt.subplot(l,b,1)
plt.plot(x, y1_vh)
plt.errorbar(x, y1_vh, yerr = y_error_vh, fmt ='o',color='black') 
plt.title('Field 1')

plt.subplot(l,b,2)
plt.plot(x, y2_vh)
plt.errorbar(x, y2_vh, yerr = y_error_vh, fmt ='o',color='black') 
plt.title('Field 2')

plt.subplot(l,b,3)
plt.plot(x, y3_vh)
plt.errorbar(x, y3_vh, yerr = y_error_vh, fmt ='o',color='black') 
plt.title('Field 3')

plt.subplot(l,b,4)
plt.plot(x, y4_vh)
plt.errorbar(x, y4_vh, yerr = y_error_vh, fmt ='o',color='black') 
plt.title('Field 4')

plt.subplot(l,b,5)
plt.plot(x, y5_vh)
plt.errorbar(x, y5_vh, yerr = y_error_vh, fmt ='o',color='black') 
plt.title('Field 5')

plt.subplot(l,b,6)
plt.plot(x, y6_vh)
plt.errorbar(x, y6_vh, yerr = y_error_vh, fmt ='o',color='black') 
plt.title('Field 6')

plt.subplot(l,b,7)
plt.plot(x, y7_vh)
plt.errorbar(x, y7_vh, yerr = y_error_vh, fmt ='o',color='black') 
plt.title('Field 7')

plt.subplot(l,b,8)
plt.plot(x, y8_vh)
plt.errorbar(x, y8_vh, yerr = y_error_vh, fmt ='o',color='black') 
plt.title('Field 8')

plt.subplot(l,b,10)
plt.plot(x, y9_vh)
plt.errorbar(x, y9_vh, yerr = y_error_vh, fmt ='o',color='black') 
plt.title('Field 9')

plt.subplot(l,b,11)
plt.plot(x, y10_vh)
plt.errorbar(x, y10_vh, yerr = y_error_vh, fmt ='o',color='black') 
plt.title('Field 10')

plt.show()
#%%
l = 3
b = 4
plt.subplot(l,b,1)
plt.plot(x, y1_vv)
plt.errorbar(x, y1_vv, yerr = y_error_vv, fmt ='o',color='black') 
plt.title('Field 1')

plt.subplot(l,b,2)
plt.plot(x, y2_vv)
plt.errorbar(x, y2_vv, yerr = y_error_vv, fmt ='o',color='black') 
plt.title('Field 2')

plt.subplot(l,b,3)
plt.plot(x, y3_vv)
plt.errorbar(x, y3_vv, yerr = y_error_vv, fmt ='o',color='black') 
plt.title('Field 3')

plt.subplot(l,b,4)
plt.plot(x, y4_vv)
plt.errorbar(x, y4_vv, yerr = y_error_vv, fmt ='o',color='black') 
plt.title('Field 4')

plt.subplot(l,b,5)
plt.plot(x, y5_vv)
plt.errorbar(x, y5_vv, yerr = y_error_vv, fmt ='o',color='black') 
plt.title('Field 5')

plt.subplot(l,b,6)
plt.plot(x, y6_vv)
plt.errorbar(x, y6_vv, yerr = y_error_vv, fmt ='o',color='black') 
plt.title('Field 6')

plt.subplot(l,b,7)
plt.plot(x, y7_vv)
plt.errorbar(x, y7_vv, yerr = y_error_vv, fmt ='o',color='black') 
plt.title('Field 7')

plt.subplot(l,b,8)
plt.plot(x, y8_vv)
plt.errorbar(x, y8_vv, yerr = y_error_vv, fmt ='o',color='black') 
plt.title('Field 8')

plt.subplot(l,b,10)
plt.plot(x, y9_vv)
plt.errorbar(x, y9_vv, yerr = y_error_vv, fmt ='o',color='black') 
plt.title('Field 9')

plt.subplot(l,b,11)
plt.plot(x, y10_vv)
plt.errorbar(x, y10_vv, yerr = y_error_vv, fmt ='o',color='black') 
plt.title('Field 10')

plt.show()
