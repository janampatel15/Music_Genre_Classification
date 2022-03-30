# # Generating Mel Spectrogram
# In this file, we generate Mel Spectrogram images to use them with our CNN models.
#
# We will be generating over 1000 images

from scipy.io import wavfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy import signal
import IPython.display as ipd
import librosa
import librosa.display
import os
import wave
from pandas_profiling import ProfileReport
from PIL import Image

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set()
from numpy import asarray

# ### Problem
# We noticed that images provided to us had thick white border, which isn't good when we pass it through our CNN model, since white carries value of **255,255,255** in RGB, it'll mess with our model. We thought it would be best to generate new image without those thick borders.
#
# You can see the difference between image provided in the dataset from Kaggle, [here](#Kaggle-Image), and image generate by us over [here](#Newly-Generated-Image).

# #### Kaggle Image

image = Image.open('../data/images_original/blues/blues00000.png')

# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)
print(asarray(image).shape)
image

# #### Newly Generated Image

image_gen = Image.open('../data/images_gen/blues/blues00002.png')

# summarize some details about the image
print(image_gen.format)
print(image_gen.size)
print(image_gen.mode)
print(asarray(image_gen).shape)
image_gen

# this code goes through all the genres, and their respective audio files, create mel spectrogram,
#
# and saves it in the images_gen folder with each genre having its oown folder
#
# careful if running since there are 1000 audio files, 1000 images will be genrated, which can be time consuming
#
# since going through classical and blues genre took us about 1hr and 8min, and that only generatted 200 images.

genres = [
    'classical', 'blues', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop',
    'reggae', 'rock'
]

for i in tqdm(genres):
    print(i)

    isExist = os.path.exists(f'../data/images_gen/{i}')
    if not isExist:
        os.makedirs(f'../data/images_gen/{i}')

    for j in tqdm(sorted(os.listdir(f'../data/genres_original/{i}'))):
        filename = f'../data/genres_original/{i}/{j}'
        try:
            data, sr = librosa.load(filename)
            #             print(filename)
            mel = librosa.feature.melspectrogram(y=data, sr=sr)
            S_dB = librosa.power_to_db(mel, ref=np.max)
            img = librosa.display.specshow(S_dB, sr=sr, fmax=8000)

            plt.savefig(
                f'../data/images_gen/{i}/{i+filename.split(".")[-2]}.png',
                bbox_inches='tight',
                pad_inches=0)

        except:
            pass
