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

并将pretrained_psp_encoder修改为pretrained_psp，因为option里用的是pretrained_psp。

CelebAMask-HQ的[repo](https://github.com/switchablenorms/CelebAMask-HQ)，下载方法和FFHQ中GDrive的下载方法一致。

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

### 算法概述

直接看coach_aging.py中的train函数：

首先预测一个batch的FFHQ图像中人脸的年龄，得到input_ages shape: [B,]

设置有1/3的概率不做年龄变化，直接将预测的年龄当成第4个channel加入到图像中，而2/3的概率从[0, 101]之间随机年龄作为待转年龄，同样加入到第4 channel。

这样最终网络的输入x_input shape: [B, C, 256, 256]，C = 4

将x_input输入到网络，执行y_hat, latent = self.perform_forward_pass(x_input)，这里即是论文中Fig. 2所示的算法流程，下面拆解来看。

x_input即为图中将x和α_t结合起来的输入x_age，E_age即为图中的绿色和蓝色部分，其实结构和pSp Encoder完全一致。

执行perform_forward_pass(x_input)内部实际上调用了psp.py中定义的网络结构，我们看forward函数。

首先用绿框和蓝框锁表示的self.encoder编码x_input，得到codes shape[B, 18, 512]，正是StyleGAN2所需的输入latent space W+，Fig. 2中的E_age(x_age)。

得到codes之后，Fig. 2中还有上半部分用pretrained psp来对x [B, 3， 256， 256]进行编码，预训的pSp模型从图像上提取latent space encoded_latents（这里要加上latent_avg，预测的是做过归一化的），预训得到的latent即为图中的紫色部分，而上述pSp提取的是包含了年龄信息的latent表示，可以看成预训pSp模型提取的latent表示的age残差，所以将两者相加再过一个预训的StyleGAN2，即得到图中的黄色部分。

```
images, result_latent = self.decoder([codes],
											 input_is_latent=input_is_latent,
											 randomize_noise=randomize_noise,
											 return_latents=return_latents)
```

这部分代码就是预训的StyleGAN2，用刚才得到的黄色部分的latent生成图像，这里images的shape是[B, 3， 1024， 1024]，需要resize到256x256，代码中采用均值池化的方法完成。

回到y_hat, latent = self.perform_forward_pass(x_input)；这里得到的y_hat即为图中的y_out，latent其实和黄色部分的latent是完全一样的。

至此，整个训练前向的过程完毕，需要进行loss的计算。

```
loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,
														  target_ages=target_ages,
														  input_ages=input_ages,
														  no_aging=no_aging,
														  data_type="real")
```

这是论文最复杂的部分，上述代码里的data_type有两个取值，"real"和"cycle"，real就是上述的正常流程，cycle指的是预测出来的y_out和x的预测年龄α_s（“真实”年龄）组成新的输入去得到原图的预测（故名为cycle），两个模式的loss计算有区别，这里挑代码中不容易理解的点记录。

1. ID loss，年龄转换过程中身份应该是保持的。

年龄差别越大，ID保持的权重应该越低，比如5岁和55岁的同一人，其实长相差异已经很大了，所以训练阶段需要根据年龄差异求权重系数，论文中的Eq. (10)，代码即为use_weighted_id_loss这个条件下的语句。

```
loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, label=data_type, weights=weights)
```

使用ArcFace从x，y和y_hat中提取特征，特征维度512，二范式为1，所以我们就可以利用cosine distance（因为二范式是1，所以直接求512维特征的点积即为cos距离）判断图像之间的相似性，diff_target即为目标和预测的相似度，也就是id的loss，在aging任务中diff_target和diff_input是一致的，diff_view是1，代码中的sim_improvement不知道何用

2. L2 loss

这块直接算像素级的L2 loss，直接看代码。

3. LPIPS loss

这这个loss定义在Eq. (8)，具体参考文章详见论文，这个loss定义了perceptual similarity（感知相似度）。

4. L2和LPIPS loss在图像中心权重更大

这个部分在Eq. (8)的表述之下那句话有说。代码中由lpips_lambda_crop和l2_lambda_crop控制，这部分相当于早了一个中心的ROI，然后再算了一遍L2和LPIPS加到最终loss中，相当于中心权重更大…这块比较tricky。

5. Regularization loss

论文Eq. (8)下描述的loss，这个loss鼓励生成的latent和avg latent更接近，详见w_norm_lambda这里的定义。

6. Aging loss

预测y和y_out的年龄，算mse loss。

cycle loss即为将预测的y_out当成x，x预测的age当成α_s，即用待转年龄预测的人脸和原图原始年龄当成输入预测原始人脸，形成cycle。loss计算和正向一致，单独加了一个权重。

至此，loss部分全部算完。