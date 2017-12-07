import requests
from PIL import Image
import numpy as np
import scipy.misc


global matrix_vector
global masks_vector

def generate_batch(batch_size, fh):
    batch_im = np.zeros([batch_size, 64, 64, 1], dtype="float32")
    batch_res = np.zeros([batch_size, 64, 64, 1], dtype="float32")
    batch_intensity = np.zeros([batch_size, 410, 1], dtype="float32")
    try:
        matrix_vector
        masks_vector
    except:
        matrix_vector = []
        masks_vector = []
        np.random.seed(1)
        for _ in range(0, 410):
            random_matrix = (np.random.rand(64, 64) < 0.5) * np.float32(1)
            masks_vector.append(random_matrix)
            matrix_vector.append(random_matrix.flatten())
    for i in range(0, batch_size):
        intensity_vector = []
        im = []
        res = []
        count = 0
        while len(im) == 0 and len(res) == 0:
            try:
                url = fh.readline().split()[1]
                im = np.asarray(
                    Image.open(
                        requests.get(url, stream=True, timeout=(0.5,0.5)).raw))
                im = scipy.misc.imresize(im, (64, 64))[:, :, 0]
                print(i)
            except:
                im = []
                res = []
                continue
            for j in masks_vector:
                intensity_vector.append(np.sum(j * im))
            im = im.astype('float32')
            Tmatrix_vector=np.linalg.pinv(np.matrix(matrix_vector))
            res = np.reshape(
                np.matmul(Tmatrix_vector,intensity_vector), (64, 64))
            count += 1
        batch_im[i, :, :, 0] = im
        batch_im = batch_im.astype('float32')
        batch_res[i, :, :, 0] = res
        batch_res = batch_res.astype('float32')
        batch_intensity[i, :, 0] = intensity_vector
        batch_intensity = batch_intensity.astype('float32')
    return batch_im, batch_res ,batch_intensity


#imagenet
fh = open("fall11_urls.txt")

for i in range(0,150):
    im,res,intensity=generate_batch(256,fh)
    np.save("images_2/im_"+str(i), im)
    np.save("images_2/res_410_"+str(i), res)
    #np.save("images_2/intensity_1000_"+str(i), intensity)
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
