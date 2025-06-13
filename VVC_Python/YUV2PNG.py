from PIL import Image
import numpy as np
 
def YUV2PNG(image_name):
    # image_name = "frame2.raw" #Change to user input
    width = int(256) #Assumed to be static
    height = int(256) #Assumed to be static
    y_end = width*height

    yuv = np.fromfile(image_name, dtype='uint8')

    y = yuv[0:y_end].reshape(height,width)
    u = yuv[y_end::2].reshape(height//2, width//2)
    v = yuv[y_end+1::2].reshape(height//2, width//2)

    u = u.repeat(2, axis=0).repeat(2, axis=1)
    v = v.repeat(2, axis=0).repeat(2, axis=1)

    y = y.reshape((y.shape[0], y.shape[1], 1))
    u = u.reshape((u.shape[0], u.shape[1], 1))
    v = v.reshape((v.shape[0], v.shape[1], 1))

    yuv_array = np.concatenate((y, u, v), axis=2)

    # Overflow: yuv_array.dtype = 'uint8', so subtracting 128 overflows.
    #yuv_array[:, :, 0] = yuv_array[:, :, 0].clip(16, 235).astype(yuv_array.dtype) - 16
    #yuv_array[:, :, 1:] = yuv_array[:, :, 1:].clip(16, 240).astype(yuv_array.dtype) - 128

    # Convert from uint8 to float32 before subtraction
    yuv_array = yuv_array.astype(np.float32)
    yuv_array[:, :, 0] = yuv_array[:, :, 0].clip(16, 235) - 16
    yuv_array[:, :, 1:] = yuv_array[:, :, 1:].clip(16, 240) - 128


    convert = np.array([#[1.164,  0.000,  1.793],[1.164, -0.213, -0.533],[1.164,  2.112,  0.000]
                        [1.164,  0.000,  2.018], [1.164, -0.813, -0.391],[1.164,  1.596,  0.000]
                    ])
    rgb = np.matmul(yuv_array, convert.T).clip(0,255).astype('uint8')


    rgb_image = Image.fromarray(rgb)

    rgb_image.save('output.png')