import numpy as np
from numpy import load
from matplotlib import pyplot
#from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import mean_absolute_error
import sewar
import seaborn as sb
sb.set_context('paper',2,rc={"lines.linewidth": 2})
import matplotlib.pyplot as plt
#%%
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = 2*((X1-np.min(X1))/(np.max(X1)-np.min(X1)))-1
    X2 = 2*((X2-np.min(X2))/(np.max(X2)-np.min(X2)))-1
    return [X1, X2]
#%%
def plot_images(data,num):
    pyplot.suptitle('NDWI VV image pair of index ' + str(num))
    pyplot.subplot(1,2,1)
    pyplot.imshow(data[0][num])
    pyplot.title('VV')

    pyplot.subplot(1,2,2)
    pyplot.imshow(data[1][num])
    pyplot.title('NDWI')
    pyplot.show() 
#%%
[X1, X2] = load_real_samples('E:/GAN_data/vh_ndvi_testing.npz')
[X3, X4] = load_real_samples('E:/GAN_data/vv_ndvi_testing.npz')
#%%
# load model
import tensorflow as tf
unet_vh_model = tf.keras.models.load_model('E:/GAN_results/mixed/unet_vh/model_726400.h5')
unet_vv_model = tf.keras.models.load_model('E:/GAN_results/mixed/unet_vv/model_726400.h5')
resnet_vh_model = tf.keras.models.load_model('E:/GAN_results/mixed/resnet_vh/model_726400.h5')
resnet_vv_model = tf.keras.models.load_model('E:/GAN_results/mixed/resnet_vv/model_726400.h5')
#model.compile(loss=['binary_crossentropy', 'mae'], optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[1, 100])
#%% Predicting NDVI for VH data
ix = np.arange(0,1140)#randint(0, len(X1), 1)

vh_src = [X1[i] for i in ix]
reshaped_vh_src = [i.reshape(1,256,256) for i in vh_src]
vh_targ = [X2[i] for i in ix]

unet_vh_pred = [unet_vh_model.predict(reshaped_vh_src[i]) for i in ix] 
reshaped_unet_vh_pred = [i.reshape(256,256) for i in unet_vh_pred]   
resnet_vh_pred = [resnet_vh_model.predict(reshaped_vh_src[i]) for i in ix] 
reshaped_resnet_vh_pred = [i.reshape(256,256) for i in resnet_vh_pred] 
#%% VH metrics for UNet and ResNet
unet_vh_rmse = [sewar.rmse(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
unet_vh_mae = [mean_absolute_error(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
unet_vh_psnr = [peak_signal_noise_ratio(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
unet_vh_corr = [np.corrcoef(np.asanyarray(vh_targ[i]).ravel(),np.asanyarray(reshaped_unet_vh_pred[i]).ravel())[0][1] for i in ix]

#unet_vh_ssi = [structural_similarity(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
#unet_vh_mse = [sewar.mse(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
#unet_vh_sam = [sewar.sam(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
#unet_vh_ergas = [sewar.ergas(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
#unet_vh_vif = [sewar.vifp(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
#unet_vh_scc = [sewar.scc(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
#unet_vh_uqi = [sewar.uqi(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]

resnet_vh_rmse = [sewar.rmse(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]
resnet_vh_mae = [mean_absolute_error(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]
resnet_vh_psnr = [peak_signal_noise_ratio(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]
resnet_vh_corr = [np.corrcoef(np.asanyarray(vh_targ[i]).ravel(),np.asanyarray(reshaped_resnet_vh_pred[i]).ravel())[0][1] for i in ix]
#resnet_vh_mse = [sewar.mse(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]    
#resnet_vh_sam = [sewar.sam(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix] 
#resnet_vh_ergas = [sewar.ergas(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]
#resnet_vh_vif = [sewar.vifp(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]
#resnet_vh_scc = [sewar.scc(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]
#resnet_vh_uqi = [sewar.uqi(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]   
#%% Predicting NDVI for VV data
vv_src = [X3[i] for i in ix]
reshaped_vv_src = [i.reshape(1,256,256) for i in vv_src]
vv_targ = [X4[i] for i in ix]

unet_vv_pred = [unet_vv_model.predict(reshaped_vv_src[i]) for i in ix] 
reshaped_unet_vv_pred = [i.reshape(256,256) for i in unet_vv_pred]   
resnet_vv_pred = [resnet_vv_model.predict(reshaped_vv_src[i]) for i in ix] 
reshaped_resnet_vv_pred = [i.reshape(256,256) for i in resnet_vv_pred]    
#%% VV metrics for UNet and ResNet
unet_vv_rmse = [sewar.rmse(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
unet_vv_mae = [mean_absolute_error(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
unet_vv_psnr = [peak_signal_noise_ratio(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
unet_vv_corr = [np.corrcoef(np.asanyarray(vv_targ[i]).ravel(),np.asanyarray(reshaped_unet_vv_pred[i]).ravel())[0][1] for i in ix]
#unet_vv_mse = [sewar.mse(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
#unet_vv_sam = [sewar.sam(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix] 
#unet_vv_ergas = [sewar.ergas(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
#unet_vv_vif = [sewar.vifp(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
#unet_vv_scc = [sewar.scc(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
#unet_vv_uqi = [sewar.uqi(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
  
resnet_vv_rmse = [sewar.rmse(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
resnet_vv_mae = [mean_absolute_error(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
resnet_vv_psnr = [peak_signal_noise_ratio(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
resnet_vv_corr = [np.corrcoef(np.asanyarray(vv_targ[i]).ravel(),np.asanyarray(reshaped_resnet_vv_pred[i]).ravel())[0][1] for i in ix]
#resnet_vv_mse = [sewar.mse(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]    
#resnet_vv_sam = [sewar.sam(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
#resnet_vv_ergas = [sewar.ergas(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
#resnet_vv_vif = [sewar.vifp(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
#resnet_vv_scc = [sewar.scc(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
#resnet_vv_uqi = [sewar.uqi(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]     
#%% Plotting error ditribution
#def plot_error_dist():
'''
fig = plt.figure()
fig.suptitle('Experiment 3: Image similarity metrics')
plt.subplot(2,3,1)
sb.distplot(unet_vh_rmse,label='UNet')
sb.distplot(resnet_vh_rmse,label='ResNet')
plt.title('VH RMSE')
plt.legend()
plt.subplot(2,3,2)
sb.distplot(unet_vh_psnr,label='UNet').set(ylabel=None)
sb.distplot(resnet_vh_psnr,label='ResNet').set(ylabel=None)
plt.title('VH PSNR')
#plt.legend()
plt.subplot(2,3,3)
sb.distplot(unet_vh_ssi,label='UNet').set(ylabel=None)
sb.distplot(resnet_vh_ssi,label='ResNet').set(ylabel=None)
plt.title('VH SSIM')
#plt.legend()
plt.subplot(2,3,4)
sb.distplot(unet_vv_rmse,label='UNet')
sb.distplot(resnet_vv_rmse,label='ResNet')
plt.title('VV RMSE')
plt.legend()
plt.subplot(2,3,5)
sb.distplot(unet_vv_psnr,label='UNet').set(ylabel=None)
sb.distplot(resnet_vv_psnr,label='ResNet').set(ylabel=None)
plt.title('VV PSNR')
#plt.legend()
plt.subplot(2,3,6)
sb.distplot(unet_vv_ssi,label='UNet').set(ylabel=None)
sb.distplot(resnet_vv_ssi,label='ResNet').set(ylabel=None)
plt.title('VV SSIM')
#plt.legend()
plt.savefig('experiment3-metrics.png',dpi=1000)
'''

#%% Plotting the source traget and generated images
def plot_final_vv(w,x,y,z):
    plt.figure()
    #fig.suptitle(title)
    plt.subplot(2,4,1)
    plt.imshow(w,cmap="gray")
    plt.axis('off')
    #plt.title('Source VV image')
    
    plt.subplot(2,4,2)
    plt.imshow(x,cmap="gray")
    plt.axis('off')
    #plt.title('U-Net GAN Generated NDVI')
    
    plt.subplot(2,4,3)
    plt.imshow(y,cmap="gray")
    plt.axis('off')
    #plt.title('ResNet GAN Generated NDVI')
    
    plt.subplot(2,4,4)
    plt.imshow(z,cmap="gray")
    plt.axis('off')
    #plt.title('Target NDVI image')
    
    plt.subplot(2,4,5)
    sb.distplot(w)
    plt.ylabel('Normalized Density')
    plt.xlabel('Backscatter \n (a)')
    #plt.title('Source VV image')
    
    plt.subplot(2,4,6)
    sb.distplot(x)
    plt.ylabel('')
    plt.xlabel('NDVI \n (b)')
    #plt.title('U-Net GAN Generated NDVI')
    
    plt.subplot(2,4,7)
    sb.distplot(y)
    plt.ylabel('')
    plt.xlabel('NDVI \n (c)')
    #plt.title('ResNet GAN Generated NDVI')
    
    plt.subplot(2,4,8)
    sb.distplot(z)
    plt.ylabel('')
    plt.xlabel('NDVI \n (d)')
    #plt.title('Target NDVI image')
    #plt.savefig('experiment3-vv.png',dpi=1000)
    #plt.show()
#%% Plotting the source traget and generated images
def plot_final_vh(w,x,y,z):
    plt.figure()
    #fig.suptitle(title)
    plt.subplot(2,4,1)
    plt.imshow(w,cmap="gray")
    plt.axis('off')
    #plt.title('Source VH image')
    
    plt.subplot(2,4,2)
    plt.imshow(x,cmap="gray")
    plt.axis('off')
    #plt.title('U-Net GAN Generated NDVI')
    
    plt.subplot(2,4,3)
    plt.imshow(y,cmap="gray")
    plt.axis('off')
    #plt.title('ResNet GAN Generated NDVI')
    
    plt.subplot(2,4,4)
    plt.imshow(z,cmap="gray")
    plt.axis('off')
    #plt.title('Target NDVI image')
    
    plt.subplot(2,4,5)
    sb.distplot(w)
    plt.ylabel('Normalized Density')
    plt.xlabel('Backscatter \n (a)')
    #plt.title('Source VH image')
    
    plt.subplot(2,4,6)
    sb.distplot(x,norm_hist=False)
    plt.ylabel('')
    plt.xlabel('NDVI \n (b)')
    #plt.title('U-Net GAN Generated NDVI')
    
    plt.subplot(2,4,7)
    sb.distplot(y)
    plt.ylabel('')
    plt.xlabel('NDVI \n (c)')
    #plt.title('ResNet GAN Generated NDVI')
    
    plt.subplot(2,4,8)
    sb.distplot(z)
    plt.ylabel('')
    plt.xlabel('NDVI \n (d)')
    #plt.title('Target NDVI image')
    #plt.savefig('experiment3-vh.png',dpi=1000)
    #plt.show()
#%%
# plot all three images
iy=0 #100 #340 160
plot_final_vh(vh_src[iy], reshaped_unet_vh_pred[iy], reshaped_resnet_vh_pred[iy], vh_targ[iy])
plot_final_vv(vv_src[iy], reshaped_unet_vv_pred[iy], reshaped_resnet_vv_pred[iy], vv_targ[iy])
#%%
import scipy.stats as st
unet_vh = [unet_vh_rmse,unet_vh_mae,unet_vh_psnr,unet_vh_corr]
unet_vv = [unet_vv_rmse,unet_vv_mae,unet_vv_psnr,unet_vv_corr]
resnet_vh = [resnet_vh_rmse,resnet_vh_mae,resnet_vh_psnr,resnet_vh_corr]
resnet_vv = [resnet_vv_rmse,resnet_vv_mae,resnet_vv_psnr,resnet_vv_corr]
def errors(x):
    for i in x:
        print(np.mean(i))
        print(st.norm.interval(alpha=0.95,loc=np.mean(i),scale=st.sem(i)))
        print((1.96 *(np.std(i)/np.sqrt(1140))))
        #print(np.median(i))
        print(np.std(i))
        print(15 * "#")
    #print([(np.mean(i),st.norm.interval(alpha=0.95,loc=np.mean(i),scale=st.sem(i)),np.median(i)) for i in x])
# https://www.geeksforgeeks.org/confidence-interval/
# https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
#%%
errors(unet_vh)
errors(unet_vv)
errors(resnet_vh)
errors(resnet_vv)
#%%
#unet_vh_metrics = errors(unet_vh)
#unet_vv_metrics = errors(unet_vv)
#resnet_vh_metrics = errors(resnet_vh)
#resnet_vv_metrics = errors(resnet_vv)
