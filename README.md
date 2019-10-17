# CenterNet-darknet
A Darknet implementation of CenterNet, which support training and inference.
# Usage
1. clone CenterNet-darknet repo
```bash
git clone https://github.com/CaoWGG/CenterNet-darknet.git
```
<br><br>
2. convert dla34v0(no deform conv) model to darknet.
```bash
cd torch2darknet/
python3 torch2darknet.py
cp dla34_ctdet.cfg ../darknet/train_config/
cp dla34_ctdet.weights ../darknet/weights/pretrain
```
Convtranspose2d is not supported. convert deconv layer to upsample layer.
<br><br>
3. prepare data
```bash
cd darknet/scripts/
bash get_coco_dataset.sh
```
and then update `darknet/train_config/coco.data`
<br><br>
4. train
```bash
cd darknet/
make -j4
./darknet detector train train_config/coco.data train_config/dla34_ctdet.cfg weights/pretrain/dla34_ctdet.weights -gpus 0,1
```
<br><br>
5. test
```bash
./darknet detect train_config/coco.data train_config/dla34_ctdet.cfg.test weights/backup/dla34_ctdet.backup 5k.txt -i 0
```
batch=1 and subdivisions=1 in `dla34_ctdet.cfg.test`
<br><br>
### Notice
(xy and wh) loss is l1loss, hm loss is focalloss.
<br><br>
GPU is supported only.
