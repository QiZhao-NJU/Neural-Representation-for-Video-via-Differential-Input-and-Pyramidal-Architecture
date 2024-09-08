# DNeRV: Modeling Inherent Dynamics via Difference Neural Representation for Videos (CVPR2023)
# PNeRV: Enhancing Spatial Consistency via Pyramidal Neural Representation for Video (CVPR2024)

This is the official implementation of the paper "[DNeRV: Modeling Inherent Dynamics via Difference Neural Representation for Videos](https://arxiv.org/pdf/2304.06544.pdf) (CVPR2023)" and "[PNeRV: Enhancing Spatial Consistency via Pyramidal Neural Representation for Videos](https://arxiv.org/pdf/2404.08921.pdf) (CVPR2024)".

Our work is modified base on "[E-NeRV: Expedite Neural Video Representation with Disentangled Spatial-Temporal Context](https://arxiv.org/pdf/2207.08132.pdf) (ECCV 2022)", https://github.com/kyleleey/E-NeRV. Thanks to Zizhang Li for his excellent implementation. The code structure of the DNeRV is the same as E-NeRV.

Also, DNeRV is inspire by "[HNeRV: A Hybrid Neural Representation for Videos](https://arxiv.org/pdf/2304.02633.pdf) (CVPR2023)". More than content stream proposed by HNeRV, we add the difference stream with a difference encoder and a attention module, Collaborative Content Unit (CCU).

use command: _bash scripts/run.sh cfgs/DNeRV.yaml xxx 29500_ to run the code.
