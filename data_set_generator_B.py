import requests
from PIL import Image
import numpy as np
import scipy.misc


global matrix_vector
global masks_vector
ii=0
def generate_batch(batch_size, fh):
    global ii
    batch_im = np.zeros([batch_size, 64, 64, 1], dtype="float32")
    batch_intensity = np.zeros([batch_size, 64*64, 1], dtype="float32")
    try:
        masks_vector
    except:
        masks_vector = []
        np.random.seed(1)
        for _ in range(0, 64*64):
            random_matrix = (np.random.rand(64, 64) < 0.5) * np.float32(1)
            masks_vector.append(random_matrix)
    for i in range(0, batch_size):
        intensity_vector = []
        im = []
        res = []
        count = 0
        while len(im) == 0 and len(res) == 0:
            try:
                url = fh.readline().split()[1]
                ii=ii+1
                im = np.asarray(
                    Image.open(
                        requests.get(url, stream=True, timeout=(0.5,0.5)).raw))
                im = scipy.misc.imresize(im, (64, 64))[:, :, 0]
                
            except:
                im = []
                res = []
                continue
            for j in masks_vector:
                intensity_vector.append(np.sum(j * im))
            im = im.astype('float32')
            count += 1
        batch_im[i, :, :, 0] = im
        batch_im = batch_im.astype('float32')
        batch_intensity[i, :, 0] = intensity_vector
        batch_intensity = batch_intensity.astype('float32')
    return batch_im,batch_intensity


#imagenet
fh = open("fall11_urls.txt")
for i in range(0,1):
    try:
        url = fh.readline()
    except:
        continue
for i in range(1,4000):
    im,intensity=generate_batch(256,fh)
    print(ii)
    np.save("data_set/im_"+str(i), im)
    np.save("data_set/intensities_"+str(i), intensity)
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
