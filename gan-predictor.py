import numpy as np
from numpy import load
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import sewar
import seaborn as sb
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
[X1, X2] = load_real_samples('C:/Users/admin/vh_ndvi_entire_256.npz')
[X3, X4] = load_real_samples('C:/Users/admin/vv_ndvi_entire_256.npz')
#%%
# load model
import tensorflow as tf
unet_vh_model = tf.keras.models.load_model('E:/arun/GAN/results/unet_200_ndvi_vh/model_224200.h5')
unet_vv_model = tf.keras.models.load_model('E:/arun/GAN/results/unet_200_ndvi_vv/model_224200.h5')
resnet_vh_model = tf.keras.models.load_model('E:/arun/GAN/results/resnet_200_ndvi_vh/model_224200.h5')
resnet_vv_model = tf.keras.models.load_model('E:/arun/GAN/results/resnet_200_ndvi_vv/model_224200.h5')
#model.compile(loss=['binary_crossentropy', 'mae'], optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[1, 100])
#%% Calculating the metrics for VH backscatter
ix = np.arange(0,1062)#randint(0, len(X1), 1)

vh_src = [X1[i] for i in ix]
reshaped_vh_src = [i.reshape(1,256,256) for i in vh_src]
vh_targ = [X2[i] for i in ix]

unet_vh_pred = [unet_vh_model.predict(reshaped_vh_src[i]) for i in ix] 
reshaped_unet_vh_pred = [i.reshape(256,256) for i in unet_vh_pred]   
resnet_vh_pred = [resnet_vh_model.predict(reshaped_vh_src[i]) for i in ix] 
reshaped_resnet_vh_pred = [i.reshape(256,256) for i in resnet_vh_pred] 
#%%
unet_vh_rmse = [mean_squared_error(vh_targ[i], reshaped_unet_vh_pred[i], squared=False) for i in ix]
unet_vh_mse = [mean_squared_error(vh_targ[i], reshaped_unet_vh_pred[i], squared=True) for i in ix]
unet_vh_psnr = [peak_signal_noise_ratio(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
unet_vh_ssi = [structural_similarity(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix] 
unet_vh_sam = [sewar.sam(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
unet_vh_ergas = [sewar.ergas(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
unet_vh_vif = [sewar.vifp(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
unet_vh_scc = [sewar.scc(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]
unet_vh_uqi = [sewar.uqi(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]

resnet_vh_rmse = [mean_squared_error(vh_targ[i], reshaped_resnet_vh_pred[i], squared=False) for i in ix]
resnet_vh_mse = [mean_squared_error(vh_targ[i], reshaped_resnet_vh_pred[i], squared=True) for i in ix]    
resnet_vh_psnr = [peak_signal_noise_ratio(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]
resnet_vh_ssi = [structural_similarity(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix] 
resnet_vh_sam = [sewar.sam(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix] 
resnet_vh_ergas = [sewar.ergas(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]
resnet_vh_vif = [sewar.vifp(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]
resnet_vh_scc = [sewar.scc(vh_targ[i], reshaped_resnet_vh_pred[i]) for i in ix]
resnet_vh_uqi = [sewar.uqi(vh_targ[i], reshaped_unet_vh_pred[i]) for i in ix]   
#%% Calculating the metrics for VV backscatter
vv_src = [X3[i] for i in ix]
reshaped_vv_src = [i.reshape(1,256,256) for i in vv_src]
vv_targ = [X4[i] for i in ix]

unet_vv_pred = [unet_vv_model.predict(reshaped_vv_src[i]) for i in ix] 
reshaped_unet_vv_pred = [i.reshape(256,256) for i in unet_vv_pred]   
resnet_vv_pred = [resnet_vv_model.predict(reshaped_vv_src[i]) for i in ix] 
reshaped_resnet_vv_pred = [i.reshape(256,256) for i in resnet_vv_pred]    
#%%
unet_vv_rmse = [mean_squared_error(vv_targ[i], reshaped_unet_vv_pred[i], squared=False) for i in ix]
unet_vv_mse = [mean_squared_error(vv_targ[i], reshaped_unet_vv_pred[i], squared=True) for i in ix]
unet_vv_psnr = [peak_signal_noise_ratio(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
unet_vv_ssi = [structural_similarity(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
unet_vv_sam = [sewar.sam(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix] 
unet_vv_ergas = [sewar.ergas(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
unet_vv_vif = [sewar.vifp(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
unet_vv_scc = [sewar.scc(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
unet_vv_uqi = [sewar.uqi(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]
  
resnet_vv_rmse = [mean_squared_error(vv_targ[i], reshaped_resnet_vv_pred[i], squared=False) for i in ix]
resnet_vv_mse = [mean_squared_error(vv_targ[i], reshaped_resnet_vv_pred[i], squared=True) for i in ix]    
resnet_vv_psnr = [peak_signal_noise_ratio(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
resnet_vv_ssi = [structural_similarity(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
resnet_vv_sam = [sewar.sam(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
resnet_vv_ergas = [sewar.ergas(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
resnet_vv_vif = [sewar.vifp(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
resnet_vv_scc = [sewar.scc(vv_targ[i], reshaped_resnet_vv_pred[i]) for i in ix]
resnet_vv_uqi = [sewar.uqi(vv_targ[i], reshaped_unet_vv_pred[i]) for i in ix]     
#%% Plotting the source traget and generated images
def plot_final_vv(w,x,y,z,title):
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(2,4,1)
    plt.imshow(w,cmap="gray")
    plt.title('Source VV image')
    
    plt.subplot(2,4,2)
    plt.imshow(x,cmap="gray")
    plt.title('U-Net GAN Generated NDVI image')
    
    plt.subplot(2,4,3)
    plt.imshow(y,cmap="gray")
    plt.title('Resnet GAN Generated NDVI image')
    
    plt.subplot(2,4,4)
    plt.imshow(z,cmap="gray")
    plt.title('Target NDVI image')
    
    plt.subplot(2,4,5)
    sb.distplot(w)
    plt.title('Source VV image')
    
    plt.subplot(2,4,6)
    sb.distplot(x)
    plt.title('U-Net GAN Generated NDVI')
    
    plt.subplot(2,4,7)
    sb.distplot(y)
    plt.title('Resnet GAN Generated NDVI')
    
    plt.subplot(2,4,8)
    sb.distplot(z)
    plt.title('Target NDVI image')

    plt.show()
#%% Plotting the source traget and generated images
def plot_final_vh(w,x,y,z,title):
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(2,4,1)
    plt.imshow(w,cmap="gray")
    plt.title('Source VH image')
    
    plt.subplot(2,4,2)
    plt.imshow(x,cmap="gray")
    plt.title('U-Net GAN Generated NDVI image')
    
    plt.subplot(2,4,3)
    plt.imshow(y,cmap="gray")
    plt.title('Resnet GAN Generated NDVI image')
    
    plt.subplot(2,4,4)
    plt.imshow(z,cmap="gray")
    plt.title('Target NDVI image')
    
    plt.subplot(2,4,5)
    sb.distplot(w)
    plt.title('Source VH image')
    
    plt.subplot(2,4,6)
    sb.distplot(x)
    plt.title('U-Net GAN Generated NDVI')
    
    plt.subplot(2,4,7)
    sb.distplot(y)
    plt.title('Resnet GAN Generated NDVI')
    
    plt.subplot(2,4,8)
    sb.distplot(z)
    plt.title('Target NDVI image')

    plt.show()
#%%
# plot all three images
iy=588 #340 160
plot_final_vh(vh_src[iy], reshaped_unet_vh_pred[iy], reshaped_resnet_vh_pred[iy], vh_targ[iy],'NDVI generated by VH polarization')
plot_final_vv(vv_src[iy], reshaped_unet_vv_pred[iy], reshaped_resnet_vv_pred[iy], vv_targ[iy],'NDVI generated by VV polarization')
#%%
def plot_errors(a,b,c,d,w,x,y,z,title):
    fig = plt.figure()
    fig.suptitle(title)
    plt.title('VH Data')
    plt.subplot(2,4,1)
    sb.distplot(a)
    plt.title('RMSE of Unet Generator')
    
    plt.subplot(2,4,2)
    sb.histplot(b)
    plt.title('MSE of Unet Generator')
    
    plt.subplot(2,4,3)
    sb.histplot(c)
    plt.title('PSNR of Unet Generator')
    
    plt.subplot(2,4,4)
    sb.histplot(d)
    plt.title('SSI of Unet Generator')
    
    plt.subplot(2,4,5)
    sb.histplot(w)
    plt.title('RMSE of Resnet Generator')
    
    plt.subplot(2,4,6)
    sb.histplot(x)
    plt.title('MSE of Unet Generator')
    
    plt.subplot(2,4,7)
    sb.histplot(y)
    plt.title('PSNR of Unet Generator')
    
    plt.subplot(2,4,8)
    sb.histplot(z)
    plt.title('SSI of Unet Generator')
    plt.show()
#%%
plot_errors(unet_vh_rmse,unet_vh_mse,unet_vh_psnr,unet_vh_ssi,resnet_vh_rmse,resnet_vh_mse,resnet_vh_psnr,resnet_vh_ssi,'Metrics for VH Data')
plot_errors(unet_vv_rmse,unet_vv_mse,unet_vv_psnr,unet_vv_ssi,resnet_vv_rmse,resnet_vv_mse,resnet_vv_psnr,resnet_vv_ssi,'Metrics for VV Data')
#%%
unet_vh = [unet_vh_mse,unet_vh_rmse,unet_vh_psnr,unet_vh_ssi,unet_vh_sam,unet_vh_ergas,unet_vh_vif,unet_vh_scc,unet_vh_uqi]
unet_vv = [unet_vv_mse,unet_vv_rmse,unet_vv_psnr,unet_vv_ssi,unet_vv_sam,unet_vv_ergas,unet_vv_vif,unet_vv_scc,unet_vv_uqi]
resnet_vh = [resnet_vh_mse,resnet_vh_rmse,resnet_vh_psnr,resnet_vh_ssi,resnet_vh_sam,resnet_vh_ergas,resnet_vh_vif,resnet_vh_scc,resnet_vh_uqi]
resnet_vv = [resnet_vv_mse,resnet_vv_rmse,resnet_vv_psnr,resnet_vv_ssi,resnet_vv_sam,resnet_vv_ergas,resnet_vv_vif,resnet_vv_scc,resnet_vv_uqi]
def errors(x):
    return [np.mean(i) for i in x]
#%%
unet_vh_metrics = errors(unet_vh)
unet_vv_metrics = errors(unet_vv)
resnet_vh_metrics = errors(resnet_vh)
resnet_vv_metrics = errors(resnet_vv)
