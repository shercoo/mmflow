import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import copy

import mmcv

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow

import os
import os.path as osp

def run_inference(model,img1,img2,out_dir,model_name,out_suffix):

    flow = inference_model(model, img2, img1)

    img = cv2.imread(img1)
    dst = cv2.imread(img2)

    h = img.shape[0]
    w = img.shape[1]
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.float32(np.dstack([y_coords, x_coords]))

    flow=np.dstack([flow[:,:,1],flow[:,:,0]])

    pixel_map = coords + flow

    new_frame = cv2.remap(img, pixel_map[:, :, 1], pixel_map[:, :, 0], cv2.INTER_CUBIC, cv2.BORDER_REPLICATE)

    out_name='_'.join([model_name,osp.basename(img1).split('.')[0][:-2],out_suffix])

    cv2.imwrite(osp.join(out_dir, out_name+'.png'), new_frame)

    #print(out_name,f'PSNR: {PSNR(dst,new_frame)} SSIM: {SSIM(dst,new_frame,multichannel=True)}')

    mmcv.mkdir_or_exist(out_dir)
    visualize_flow(flow, osp.join(out_dir, out_name+'_flow.png'))
    # write_flow(flow, osp.join(out_dir, 'flow.flo'))

    return PSNR(dst,new_frame),SSIM(dst,new_frame,multichannel=True)



xlabel=[]
for img_name in os.listdir('demo/ImagePairs'):
    if osp.splitext(img_name)[0].endswith('01'):
        xlabel.append(osp.splitext(img_name)[0].replace('01',''))
x=[i+1 for i in range(len(xlabel))]


results=[{},{},{},{}]

for cfg in os.listdir('checkpoints'):
    #print(cfg)
    if osp.splitext(cfg)[-1]=='.py':
        config_file=osp.join('checkpoints',cfg)
        # print(config_file)
        checkpoint_file=osp.join('checkpoints',osp.splitext(cfg)[0]+'.pth')
        # print(checkpoint_file)
        model_name=cfg.split('_')[0]

        model = init_model(config_file, checkpoint_file, device='cuda:0')

        out_dir = f'results/{model_name}'
        os.makedirs(out_dir,exist_ok=True)


        PSNR1to2=[]
        PSNR2to1=[]
        SSIM1to2=[]
        SSIM2to1 = []
        for img_name in os.listdir('demo/ImagePairs'):
            imgN = osp.splitext(img_name)[0].replace('01', '')
            if osp.splitext(img_name)[0].endswith('01'):
                img1 = osp.join('demo/ImagePairs/', img_name)
                img2 = osp.join('demo/ImagePairs/', img_name.replace('01','02'))

                p1,s1=run_inference(model,img1,img2,out_dir,model_name,'_1to2')
                p2,s2=run_inference(model,img2,img1,out_dir,model_name,'_2to1')
                PSNR1to2.append(p1)
                PSNR2to1.append(p2)
                SSIM1to2.append(s1)
                SSIM2to1.append(s2)
                print("%s & %.4f & %.4f & %.4f & %.4f \\\\ \\hline \n" % (osp.splitext(img_name)[0].replace('01',''),p1,s1,p2,s2))
        results[0][model_name]=copy.deepcopy(PSNR1to2)
        results[1][model_name]=copy.deepcopy(SSIM1to2)
        results[2][model_name]=copy.deepcopy(PSNR2to1)
        results[3][model_name]=copy.deepcopy(SSIM2to1)


titles=['PSNR1to2','SSIM1to2','PSNR2to1','SSIM2to1']
for i in range(4):
    plt.cla()
    plt.xticks(x, xlabel)
    plt.title(titles[i])  # 折线图标题
    plt.xlabel('image names')  # x轴标题
    for model_name,res in results[i].items():
        plt.plot(x, res, marker='o',label=model_name, markersize=3)  # 绘制折线图，添加数据点，设置点的大小

    plt.legend()
    plt.savefig(f'results/{titles[i]}.png',dpi=300)  # 显示折线图
