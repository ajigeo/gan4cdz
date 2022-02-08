import glob

ndwi_list = glob.glob(r"G:\vv-ndwi-data\train\ndwi\*.tif")
file_numbers = [file[32:-4] for file in ndwi_list]

odd_list = []
even_list = []
for i in file_numbers:
    if int(i)%2 == 0:
        even_list.append(i)
    else:
        odd_list.append(i)
        
final_odd_list = ['ndwi.'+i+'.tif' for i in odd_list]
final_even_list = ['ndwi.'+i+'.tif' for i in even_list]
