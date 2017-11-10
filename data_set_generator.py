import requests
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import scipy.misc


def process_image(url, masks_vector):
    intensity_vector = []
    try:
        im = np.asarray(Image.open(requests.get(url, stream=True).raw))
    except:
        pass
        return [0],[0]
    im = scipy.misc.imresize(im, (128, 128))[:, :, 0]
    for j in masks_vector:
        intensity_vector.append(np.sum(j * im))
    res = np.reshape(np.matmul(intensity_vector, matrix_vector), (128, 128))
    return im, res


matrix_vector = []
masks_vector = []
np.random.seed(1)
for _ in range(0, 5000):
    random_matrix = (np.random.rand(128, 128) < 0.5) * 1
    masks_vector.append(random_matrix)
    matrix_vector.append(random_matrix.flatten())

#imagenet
N = 100
fh = open("fall11_urls.txt")
for i in range(0, N):
    url = fh.readline().split()[1]
    print(url)
    im, result = process_image(url, masks_vector)
    if len(im)==1 and len(result) ==0:
        pass
    print(im)
    print(result)

#flickr
"""
params = (
    ('method', 'flickr.photos.search'),
    ('license', '9'),
    ('text', 'dog'),
    ('api_key', '8d1a17fd31157d62322d4599c7247bf8'),
    ('extras', 'license,url_s'),
    ('format', 'json'),
    ('nojsoncallback','1'),
    ('page','1')
)

resp = requests.get('https://api.flickr.com/services/rest/', params=params)
j_resp = json.loads(resp.text)

for i in j_resp['photos']['photo']:
    print(i['url_s'])
    im,result = process_image(i['url_s'],masks_vector)
    print(im)
    print(result)
"""