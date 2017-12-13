import requests
from PIL import Image
import numpy as np
import scipy.misc
import qrcode


global matrix_vector
global masks_vector

ii=0
subsampling = 1000

def generate_batch(batch_size):
    global ii 
    batch_im = np.zeros([batch_size, 63, 63, 1], dtype="float32")
    batch_res = np.zeros([batch_size, 63, 63, 1], dtype="float32")
    batch_intensity = np.zeros([batch_size, subsampling, 1], dtype="float32")
    try:
        matrix_vector
        masks_vector
        Tmatrix_vector
    except:
        matrix_vector = []
        masks_vector = []
        np.random.seed(1)
        for _ in range(0, subsampling):
            random_matrix = (np.random.rand(63, 63) < 0.5) * np.float32(1)
            masks_vector.append(random_matrix)
            matrix_vector.append(random_matrix.flatten())
        Tmatrix_vector=np.linalg.pinv(np.matrix(matrix_vector))
    for i in range(0, batch_size):
        intensity_vector = []
        im = []
        res = []
        count = 0
        while len(im) == 0 and len(res) == 0:
            try:
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                    box_size=3,
                    border=0,
                )
                qr.add_data(ii)
                ii=ii+1
                img = qr.make_image()
                im = np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])
                im = np.asarray(im)*1
                qr.clear()
                print(i)
            except:
                im = []
                res = []
                continue
            for j in masks_vector:
                intensity_vector.append(np.sum(j * im))
            im = im.astype('float32')
            res = np.reshape(
                np.matmul(Tmatrix_vector,intensity_vector), (63, 63))
            count += 1
        batch_im[i, :, :, 0] = im
        batch_im = batch_im.astype('float32')
        batch_res[i, :, :, 0] = res
        batch_res = batch_res.astype('float32')
        batch_intensity[i, :, 0] = intensity_vector
        batch_intensity = batch_intensity.astype('float32')
    return batch_im, batch_res ,batch_intensity

ii = 0

for i in range(0,4000):
    im,res,intensity=generate_batch(256)
    np.save("images_qr/im_63_"+str(i), im)
    np.save("images_qr/res_"+str(subsampling)+"_"+str(i), res)
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
