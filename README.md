环境安装指南请参考[mmflow](https://github.com/open-mmlab/mmflow).

本次实践用到的模型以及下载地址如下表：

<table>
    <thead>
        <tr>
            <td rowspan=2>Models</td>
            <td rowspan=2>Training Datasets</td>
            <td rowspan=2>Log</td>
            <td rowspan=2>Config</td>
            <td rowspan=2>Download</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>FlowNetC+ft</th>
            <th>Flying Chairs + Sintel</th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet/flownetc_8x1_sfine_sintel_384x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet/flownetc_8x1_sfine_sintel_384x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet/flownetc_8x1_sfine_sintel_384x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>FlowNet2CSS-sd</th>
            <th>Flying Chairs + FlyingThing3d subset + ChairsSDHom</th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>PWC-Net+</th>
            <th>FlyingChairs + FlyingThing3d subset + Sintel + KITTI2015 + HD1K</th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.py'>config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.pth'>model</a></th>
        </tr>
        <tr>
            <th>RAFT</th>
            <th>FlyingChairs + FlyingThing3d + Sintel + KITTI2015 + HD1K</th>
            <th><a href='https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_mixed_368x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_mixed_368x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_mixed_368x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA(+p)</th>
            <th>FlyingChairs + FlyingThing3d + Sintel + KITTI2015 + HD1K</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_mixed_368x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_mixed_368x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_mixed_368x768.pth'>Model</a></th>
        </tr>        
    </tbody>
</table>

将上表中各模型的Config与Model下载保存至 `checkpoints` 子目录中，在根目录下运行 `demo/warp_optical_flow.py` ，运行结果生成在 `results` 子目录中。

