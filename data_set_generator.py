import requests
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import scipy.misc


def generate_batch(batch_size,fh,masks_vector,matrix_vector):
    batch_im = np.zeros([batch_size,64,64,1])
    batch_res = np.zeros([batch_size,64,64,1])
    for i in range(0,batch_size):
        intensity_vector = []
        im=[]
        res=[]
        count=0
        while len(im)==0 and len(res)==0 :
            url = fh.readline().split()[1]
            try:
                im = np.asarray(Image.open(requests.get(url, stream=True).raw))
            except:
                print("ERROR")
                continue
            im = scipy.misc.imresize(im, (64, 64))[:, :, 0]
            for j in masks_vector:
                intensity_vector.append(np.sum(j * im))
            im = im.astype('float32')

            res = np.reshape(np.matmul(intensity_vector, matrix_vector), (64, 64))
            count+=1
        batch_im[i,:,:,0]=im
        batch_res[i,:,:,0]=res
    return batch_im, batch_res


matrix_vector = []
masks_vector = []
np.random.seed(1)
for _ in range(0, 5000):
    random_matrix = (np.random.rand(64, 64) < 0.5) * float32(1)
    masks_vector.append(random_matrix)
    matrix_vector.append(random_matrix.flatten())

#imagenet
fh = open("fall11_urls.txt") 
im,res=generate_batch(30,fh,masks_vector,matrix_vector)
print(im.shape)
print(res.shape)
print(type(im[0,:,:,0]))
print(res[0,:,:,0])
"""
params = (
    ('method', 'flickr.photos.search'),
    ('license', '9'),
    ('text', 'dog'),
    ('api_key', ''),
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
