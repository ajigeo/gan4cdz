import glob
from osgeo import gdal
from numpy import savez_compressed
from numpy import load
from matplotlib import pyplot
import numpy as np
#%%
image_list = glob.glob(r"E:\arun\GAN\GAN_data_256\ndvi\*.tif")
file_numbers = [file[35:-4] for file in image_list]

odd_list = []
even_list = []
for i in file_numbers:
    if int(i)%2 == 0:
        even_list.append(i)
    else:
        odd_list.append(i)
        
ndvi_odd_list_names = [r"E:\arun\GAN\GAN_data_256\ndvi\ndvi."+i+".tif" for i in odd_list]
ndvi_even_list_names = [r"E:\arun\GAN\GAN_data_256\ndvi\ndvi."+i+".tif" for i in even_list]

vv_odd_list_names = [r"E:\arun\GAN\GAN_data_256\vv\vv."+i+".tif" for i in odd_list]
vv_even_list_names = [r"E:\arun\GAN\GAN_data_256\vv\vv."+i+".tif" for i in even_list]

#vh_odd_list_names = [r"E:\arun\GAN\GAN_data_256\vh\vh."+i+".tif" for i in odd_list]
#vh_even_list_names = [r"E:\arun\GAN\GAN_data_256\vh\vh."+i+".tif" for i in even_list]
#%%

def read_image(filename):
    image = gdal.Open(filename)
    image_transform = image.GetGeoTransform()
    image_projection = image.GetProjectionRef()
    image_array = image.ReadAsArray()
    #image_shape = image_array.shape()
    return [image_array,image_projection,image_transform]

    
def prepare_patches(input_list):
    final_list = []
    for i in input_list:
        pixels = read_image(i)
        image_pixels = pixels[0]
        final_list.append(image_pixels)
    return final_list
#%%
even_ndvi_list = prepare_patches(ndvi_even_list_names)
even_vv_list = prepare_patches(vv_even_list_names)
#even_vh_list = prepare_patches(vh_even_list_names)
savez_compressed('vv_ndvi_train.npz', even_vv_list, even_ndvi_list)
#savez_compressed('vh_ndvi_train_256.npz', even_vh_list, even_ndvi_list)
#%%
odd_ndvi_list = prepare_patches(ndvi_odd_list_names)
odd_vv_list = prepare_patches(vv_odd_list_names)
#odd_vh_list = prepare_patches(vh_odd_list_names)
savez_compressed('vv_ndvi_test.npz', odd_vv_list, odd_ndvi_list)
#savez_compressed('vh_ndvi_test_256.npz', odd_vh_list, odd_ndvi_list)

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
    pyplot.suptitle('NDVI VH image pair of index ' + str(num))
    pyplot.subplot(1,2,1)
    pyplot.imshow(data[0][num],cmap="gray")
    pyplot.title('VH')

    pyplot.subplot(1,2,2)
    pyplot.imshow(data[1][num],cmap="gray")
    pyplot.title('NDVI')
    pyplot.show() 
    
dataset = load_real_samples('C:/Users/admin/vv_ndvi_train.npz')
