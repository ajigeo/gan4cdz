# Generating NDVI from SAR data

## Requirments
```
python=3.9.7
tensorflow=2.5.0
keras=2.4.3
matplotlib=3.4.3
seaborn=0.11.2
gdal=3.3.2
numpy=1.21.2
```

## Steps to train the model

Open the pix2pix_combined.py and run it. Make sure to chnage the locations of input data and the output location to save the trained model. The saved model can be loaded using

```python
tf.keras.models.load_model
```
and used to predict the test data.

## Using pretrained model for predictions

Open the image-prediction.py file. Now load the pretrained model weights from the previous step. Change the locations of the VH and VV test images to predict. Also change the location of the final predictions.
Now running the code will genearte the output plots for the twelve months and save the generated NDVI in the assigned folder.

## Validating the predicted images

The errors in the prediction can be calculated and plotted using the gan_errors.py file. 
