import qrcode
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=3,
    border=0,
)
#99999999999999999
#100000000000000000
qr.add_data(99999999999999999)
img = qr.make_image()
print(np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0]))
plt.imshow(img)
plt.show()