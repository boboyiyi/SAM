## Traning
### 配置环境

```
$ conda env update -n sam --file environment/sam_env.yaml
$ conda activate sam
```

### 准备数据
首先修改configs/paths_config.py，修改如下字段：
```
dataset_paths = {
    'ffhq': '/path/to/ffhq/images256x256'
    'celeba_test': '/path/to/CelebAMask-HQ/test_img',
}
```

CelebAMask-HQ的repo：https://github.com/switchablenorms/CelebAMask-HQ，下载方法和FFHQ中GDrive的下载方法一致。

FFHQ下载方法参看我fork的ffhq dataset的repo，只用下载zips里的image1024x1024即可，unzip后执行resize.py将images1024x1024 resize到images256x256。

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

