dataset_paths = {
    'celeba_test': '/mnt/cephfs/fanbo/database/CelebAMask-HQ/CelebA-HQ-img',
    'ffhq': '/mnt/cephfs/fanbo/database/ffhq-dataset/zips/images256x256',
}

model_paths = {
    'pretrained_psp': 'pretrained_models/psp_ffhq_encode.pt',
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'age_predictor': 'pretrained_models/dex_age_classifier.pth'
}
