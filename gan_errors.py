import sewar
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from matplotlib import pyplot as plt
#import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from osgeo import gdal
def read_image(filename):
    image = gdal.Open(filename)
    #image_transform = image.GetGeoTransform()
    #image_projection = image.GetProjectionRef()
    image_array = image.ReadAsArray()
    #image_shape = image_array.shape()
    return image_array

#x = np.linspace(0,10)
#y = 3*np.random.randn(50) + x
#X = sm.add_constant(x)
#res = sm.OLS(y, X).fit()
x = read_image(r'E:\data\ISRO-project\intermediate_products\VH_predicted_NDVI\act_06.tif')
x = np.asarray(x).ravel()
y = read_image(r'E:\data\ISRO-project\intermediate_products\VH_predicted_NDVI\pred_06.tif')
y = np.asarray(y).ravel()
X = sm.add_constant(x)
res = sm.OLS(y, X).fit()

st, data, ss2 = summary_table(res, alpha=0.05)
fittedvalues = data[:,2]
predict_mean_se  = data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
predict_ci_low, predict_ci_upp = data[:,6:8].T

# https://stackoverflow.com/questions/51917377/different-results-of-seaborn-and-statsmodels

plt.figure(1)
plt.plot(x, y, 'o', label="data")
plt.plot(X[:,1], fittedvalues, 'r-', label='OLS')
plt.plot(X[:,1], predict_ci_low, 'b--')
plt.plot(X[:,1], predict_ci_upp, 'b--')
plt.plot(X[:,1], predict_mean_ci_low, 'g--')
plt.plot(X[:,1], predict_mean_ci_upp, 'g--')
plt.legend()

print("mean is " + str(np.mean(y)))
print("SD is " + str(np.std(y)))
print("95% CI is " + str((1.96*(np.std(y)/np.sqrt(len(y))))))
print("CI is " + str(1.96 * np.sqrt((0.145 * (1 - 0.145)) / 65536)))
print("Correlation is " + str(np.corrcoef(x,y)[0][1]))


mse = mean_squared_error(x,y,squared=False)
mae = mean_absolute_error(x,y)
