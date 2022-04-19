import glob
from osgeo import gdal
from numpy import savez_compressed,savez
from numpy import load
from matplotlib import pyplot
import numpy as np
#%%
vh_locations = [r"E:\GAN\20210825_data\vh1\*.tif",r"E:\GAN\20210203_data\vh2\*.tif",r"E:\GAN\20211108_data\vh3\*.tif",r"E:\GAN\20211117_data\vh4\*.tif"]
vv_locations = [r"E:\GAN\20210825_data\vv1\*.tif",r"E:\GAN\20210203_data\vv2\*.tif",r"E:\GAN\20211108_data\vv3\*.tif",r"E:\GAN\20211117_data\vv4\*.tif"]
ndvi_locations = [r"E:\GAN\20210825_data\ndvi1\*.tif",r"E:\GAN\20210203_data\ndvi2\*.tif",r"E:\GAN\20211108_data\ndvi3\*.tif",r"E:\GAN\20211117_data\ndvi4\*.tif"]
lai_locations = [r"E:\GAN\20210825_data\lai1\*.tif",r"E:\GAN\20210203_data\lai2\*.tif",r"E:\GAN\20211108_data\lai3\*.tif",r"E:\GAN\20211117_data\lai4\*.tif"]
fcover_locations = [r"E:\GAN\20210825_data\fcover1\*.tif",r"E:\GAN\20210203_data\fcover2\*.tif",r"E:\GAN\20211108_data\fcover3\*.tif",r"E:\GAN\20211117_data\fcover4\*.tif"]
fapar_locations = [r"E:\GAN\20210825_data\fapar1\*.tif",r"E:\GAN\20210203_data\fapar2\*.tif",r"E:\GAN\20211108_data\fapar3\*.tif",r"E:\GAN\20211117_data\fapar4\*.tif"]
#%%
def read_data(x):
    subfolders = [glob.glob(i) for i in x]
    rootfolder = []
    for i in subfolders:
        for j in i:
            rootfolder.append(j)
    return rootfolder
#%%
vh_files = read_data(vh_locations)    
vv_files = read_data(vv_locations)    
ndvi_files = read_data(ndvi_locations)    
lai_files = read_data(lai_locations)    
fcover_files = read_data(fcover_locations)    
fapar_files = read_data(fapar_locations)    
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
vh_images = prepare_patches(vh_files)
vv_images = prepare_patches(vv_files)
ndvi_images = prepare_patches(ndvi_files)
lai_images = prepare_patches(lai_files)
fcover_images = prepare_patches(fcover_files)
fapar_images = prepare_patches(fapar_files)
#%%
savez_compressed('E:/vh_lai_training.npz', vh_images, lai_images)
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

dataset = load_real_samples('E:/vh_lai_training.npz')
#%%
def plot_images(data,num):
    pyplot.suptitle('NDWI VV image pair of index ' + str(num))
    pyplot.subplot(1,2,1)
    pyplot.imshow(data[0][num],cmap="gray")
    pyplot.title('VH')

    pyplot.subplot(1,2,2)
    pyplot.imshow(data[1][num],cmap="gray")
    pyplot.title('LAI')
    pyplot.show() 