import os
import glob
from osgeo import gdal
#from numpy import savez_compressed
#from numpy import load
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
resnet_vh_model2 = tf.keras.models.load_model('C:/Users/ISR1/model_728800.h5')
resnet_vh_model1 = tf.keras.models.load_model('C:/Users/ISR1/model_1089600.h5')
#resnet_vv_model = tf.keras.models.load_model('E:/GAN_results/mixed/resnet_vv/model_726400.h5')
#%%
# Reading the image and rescaling to [-1,+1] range 
def read_image(filename):
    image = gdal.Open(filename)
    image_transform = image.GetGeoTransform()
    image_projection = image.GetProjectionRef()
    image_array = image.ReadAsArray()
    image_array = 2*((image_array-np.min(image_array))/(np.max(image_array)-np.min(image_array)))-1
    #image_shape = image_array.shape()
    return [image_array,image_projection,image_transform]
#%%

#os.chdir('E:/data/ISRO-project/intermediate_products/VV/')
#imgs = [glob.glob(i) for i in location]
#sar_files = glob.glob('E:/data/ISRO-project/intermediate_products/large_fields/VH/New folder/*.tif')
sar_files = glob.glob('E:/data/ISRO-project/intermediate_products/VH/*.tif')
sar_data = [read_image(sar_files[i]) for i in range(len(sar_files))]
sar_images = [sar_data[i][0] for i in range(len(sar_files))]
reshaped_sar_images = [sar_images[i].reshape(1,256,256) for i in range(len(sar_files))]


#%%
#os.chdir('E:/data/ISRO-project/intermediate_products/NDVI/')
ndvi_files = glob.glob('E:/data/ISRO-project/intermediate_products/NDVI/*.tif')
ndvi_data = [read_image(ndvi_files[i]) for i in range(len(ndvi_files))]
ndvi_images = [ndvi_data[i][0] for i in range(len(ndvi_files))]
#reshaped_ndvi_images = [ndvi_images[i].reshape(1,256,256) for i in range(len(ndvi_files))]
#%%
#unet_vh_pred = vh1[0].reshape(1,256,256) 
resnet_vh_pred1 = [resnet_vh_model1.predict(reshaped_sar_images[i]) for i in range(len(sar_files))]
final_pred1 = [resnet_vh_pred1[i].reshape(256,256) for i in range(len(sar_files))]
resnet_vh_pred2 = [resnet_vh_model2.predict(reshaped_sar_images[i]) for i in range(len(sar_files))]
final_pred2 = [resnet_vh_pred2[i].reshape(256,256) for i in range(len(sar_files))]
#resnet_vv_pred = [resnet_vv_model.predict(reshaped_sar_images[i]) for i in range(len(sar_files))]
#final_pred = [resnet_vv_pred[i].reshape(256,256) for i in range(len(sar_files))]
#%%

def plot_final(x,y,z,a,title):
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(1,4,1)
    plt.imshow(x)
    plt.title('Source SAR image')
    
    plt.subplot(1,4,2)
    plt.imshow(y)
    plt.title('Old GAN Generated NDVI image')
    
    plt.subplot(1,4,3)
    plt.imshow(z)
    plt.title('New GAN Generated NDVI image')
    
    plt.subplot(1,4,4)
    plt.imshow(a)
    plt.title('Actual NDVI image')
    
    plt.show()
#%%
i=2
plot_final(sar_images[i], final_pred1[i],final_pred2[i], ndvi_images[i],'Comparison')
#plt.savefig('13.png',dpi=1000)
#%%
def plot_difference(i):
    plt.subplot(1,2,1)
    plt.imshow(np.abs(ndvi_images[i] - final_pred1[i]),cmap='jet')
    plt.title('Old GAN difference image')
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.imshow(np.abs(ndvi_images[i] - final_pred2[i]),cmap='jet')
    plt.title('New GAN difference image')
    plt.colorbar()
    plt.show() 
#%%
plot_difference(3)
#%%
os.chdir('E:/data/ISRO-project/intermediate_products/VH_predicted_NDVI/new_NDVI')
from osgeo import osr
def predicted_images(input_matrix,filename):
    rows,cols = ndvi_images[0].shape
    output_raster = gdal.GetDriverByName('GTiff').Create(filename+'.tif',cols, rows, 1 ,gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(ndvi_data[0][2])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32644)
    output_raster.SetProjection( srs.ExportToWkt() )
    output_raster.GetRasterBand(1).WriteArray(input_matrix)
    #(output_raster-np.min(output_raster))/(np.max(output_raster)-np.min(output_raster))
    output_raster.FlushCache()
#%% Rescaling actual and predicted NDVI to [0,1] scale
norm_pred_ndvi = [2*((final_pred1[i]-np.min(final_pred1[i]))/(np.max(final_pred1[i])-np.min(final_pred1[i])))-1 for i in range(len(final_pred1))]
norm_act_ndvi = [2*((ndvi_images[i]-np.min(ndvi_images[i]))/(np.max(ndvi_images[i])-np.min(ndvi_images[i])))-1 for i in range(len(ndvi_images))]
#%%
for i in range(len(norm_act_ndvi)):
    predicted_images(norm_act_ndvi[i], 'act_%02d' % (i+1))
    predicted_images(norm_pred_ndvi[i], 'norm_pred_%02d' % (i+1))
    predicted_images(final_pred1[i], 'pred_%02d' % (i+1))
            
#%%
os.chdir('E:/data/ISRO-project/intermediate_products/VH_predicted_NDVI/rescaled_NDVI')
for i in range(0,12):
    predicted_images(ndvi_images[i], 'act_%02d' % (i+1))
    predicted_images(final_pred1[i], 'pred_%02d' % (i+1))

#%% 
import seaborn as sb
sb.set_context('paper',2,rc={"lines.linewidth": 2})
#%% Plotting VH images
fig = plt.figure()
#fig.suptitle('VV backscatter in 2021')
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
for i,j in zip(range(0,12),months):
    plt.subplot(3,4,i+1)
    plt.imshow(sar_images[i],cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(j)
#%% Plotting actual NDVI
fig = plt.figure()
#fig.suptitle('Actual NDVI in 2021')
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
for i,j in zip(range(0,12),months):
    plt.subplot(3,4,i+1)
    plt.imshow(norm_act_ndvi[i],cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(j)
#%% Plotting VH predicted NDVI
fig = plt.figure()
#fig.suptitle('NDVI predicted using VV polarization in 2021')
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
for i,j in zip(range(0,12),months):
    plt.subplot(3,4,i+1)
    plt.imshow(norm_pred_ndvi[i],cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(j)
#%% Plotting difference  
fig = plt.figure()
#fig.suptitle('NDVI predicted using VV polarization in 2021')
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
for i,j in zip(range(0,12),months):
    plt.subplot(3,4,i+1)
    plt.imshow(np.abs(norm_act_ndvi[i] - norm_pred_ndvi[i]),cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.title(j)
#%%
plt.savefig('pred_ndvi_500.png',dpi=500)
