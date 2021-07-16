## Traning
### 准备数据
首先修改configs/paths_config.py，修改如下字段：
```
dataset_paths = {
    'ffhq': '/path/to/ffhq/images256x256'
    'celeba_test': '/path/to/CelebAMask-HQ/test_img',
}
```

CelebAMask-HQ的repo：https://github.com/switchablenorms/CelebAMask-HQ，下载方法和FFHQ中GDrive的下载方法一致。

FFHQ下载方法参看我fork的ffhq dataset的repo。

FFHQ images1024x1024 resize到images256x256

```
from PIL import Image
import os

in_folder = 'images1024x1024'
out_folder = 'images256x256'

os.makedirs(out_folder, exist_ok=True)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

for p, d, fs in os.walk(in_folder):
    for f in fs:
        print(f)
        if is_image_file(f):
            in_path = os.path.join(in_folder, f)
            out_path = os.path.join(out_folder, f)
            im = Image.open(in_path)
            new_size = (256, 256)
            im = im.resize(new_size)
            im.save(out_path)
```

configs/paths_config.py，下载pretrained model，注意先将各个共享文件的快捷链接加到自己的云盘。
```
$ mkdir pretrained_models && cd pretrained_models
$ rclone copy -P fanbo:/psp_ffhq_encode.pt ./
$ rclone copy -P fanbo:/stylegan2-ffhq-config-f.pt ./
$ rclone copy -P fanbo:model_ir_se50.pth ./
$ rclone copy -P fanbo:dex_age_classifier.pth ./
$ cd ..
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
$ bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

