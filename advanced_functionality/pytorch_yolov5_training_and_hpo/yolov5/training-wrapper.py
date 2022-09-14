#!/usr/bin/env python3

import os
import argparse
import shutil
import yaml ## pyyaml has to be installed for this to work
import boto3
import botocore

DATACFGFILE = '/yolov5/datacfg.yaml'
SMHYPFILE = '/yolov5/sagemaker-hyps.yaml'

def buildDataConfig(trainingdata,validationdata):
    dictfile = {"train": trainingdata + "/images", "val": validationdata + "/images", "nc": 1, "names": ["licence"]}
    with open(DATACFGFILE,'w') as fil:
        docs = yaml.dump(dictfile,fil)
    return DATACFGFILE

def createHypsFile(opts):
    ## Create a new file that we can use
    if os.path.exists(SMHYPFILE):
        os.remove(SMHYPFILE)
    hypfile = open(SMHYPFILE,'w') # New file
    args = yaml.dump(opts,hypfile)    
    hypfile.close()
    return " --hyp " + SMHYPFILE

def buildCMD(arguments):
    initialstr = "/opt/conda/bin/python3.8 /yolov5/train.py --project /opt/ml/model --cache "
    if arguments.batchsize:
        initialstr += " --batch " + str(arguments.batchsize)
    if arguments.freeze:
        initialstr += " --freeze " + str(arguments.freeze)
    if arguments.epochs:
        initialstr += " --epochs " + str(arguments.epochs)
    if arguments.patience:
       initialstr += " --patience " + str(arguments.patience) 
    if arguments.name:
       initialstr += " --name " + str(arguments.name) 
    ## Our default hyperparameter values are sourced from yolov5/data/hyps/hyp.scratch-low.yaml
    ## We then override what we want changed.
    defhyps = open('/yolov5/data/hyps/hyp.scratch-low.yaml')
    params = yaml.full_load(defhyps)
    defhyps.close()
    opts = dict()
    for item,value in params.items():
        opts[item] = value

    if arguments.lr0:
        opts['lr0'] = arguments.lr0
    if arguments.lrf:
        opts['lrf'] = arguments.lrf
    if arguments.momentum:
        opts['momentum'] = arguments.momentum
    if arguments.weight_decay:
        opts['weight_decay'] = arguments.weight_decay
    if arguments.warmup_epochs:
        opts['warmup_epochs'] = arguments.warmup_epochs
    if arguments.warmup_momentum:
        opts['warmup_momentum'] = arguments.warmup_momentum
    if arguments.warmup_bias_lr:
        opts['warmup_bias_lr'] = arguments.warmup_bias_lr
    if arguments.box:
        opts['box'] = arguments.box
    if arguments.cls:
        opts['cls'] = arguments.cls
    if arguments.cls_pw:
        opts['cls_pw'] = arguments.cls_pw
    if arguments.obj:
        opts['obj'] = arguments.obj
    if arguments.obj_pw:
        opts['obj_pw'] = arguments.obj_pw
    if arguments.iou_t:
        opts['iou_t'] = arguments.iou_t
    if arguments.anchor_t:
        opts['anchor_t'] = arguments.anchor_t
    if arguments.fl_gamma:
        opts['fl_gamma'] = arguments.fl_gamma
    if arguments.hsv_h:
        opts['hsv_h'] = arguments.hsv_h
    if arguments.hsv_s:
        opts['hsv_s'] = arguments.hsv_s
    if arguments.hsv_v:
        opts['hsv_v'] = arguments.hsv_v
    if arguments.degrees:
        opts['degrees'] = arguments.degrees
    if arguments.translate:
        opts['translate'] = arguments.translate
    if arguments.scale:
        opts['scale'] = arguments.scale
    if arguments.shear:
        opts['shear'] = arguments.shear
    if arguments.perspective:
        opts['perspective'] = arguments.perspective
    if arguments.flipud:
        opts['flipud'] = arguments.flipud
    if arguments.fliplr:
        opts['fliplr'] = arguments.fliplr
    if arguments.mosaic:
        opts['mosaic'] = arguments.mosaic
    if arguments.mixup:
        opts['mixup'] = arguments.mixup
    if arguments.copy_paste:
        opts['copy_paste'] = arguments.copy_paste

    initialstr += createHypsFile(opts)
    return initialstr

    

def getDataFromS3(s3source,dest,dt):
    '''Requires, S3 source, local destination directory to copy to, and
       and whether train or val'''
    bucketname = s3source.split('/')[2]
    prfx = (s3source.split(bucketname)[1]).strip('/')
    fls = []
    client = boto3.client('s3')
    kwargs = {'Bucket': bucketname,'Prefix': prfx + '/' + dt}
    while True:
        resp = client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            fls.append(obj['Key'])
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break
    for fl in fls:
        fname = fl.split('/')[-1]
        client.download_file(bucketname,fl,dest + dt + '/' + fname)

def separateimages(prefx,chan):
    '''Here prefx is the full path to the parent of train or val'''
    pngs = [f for f in os.listdir(prefx + '/' + chan) if f.endswith('png')]
    if not os.path.exists(prefx + '/' + chan + '/images'):
        os.makedirs(prefx + '/' + chan + '/images')
    for png in pngs:
        shutil.copyfile(prefx + '/' + chan + '/' + png,prefx + '/' + chan + '/images' + '/' + png)
    txts = [f for f in os.listdir(prefx + '/' + chan) if f.endswith('txt')] 
    if not os.path.exists(prefx + '/' + chan + '/labels'):
        os.makedirs(prefx + '/' + chan + '/labels')
    for t in txts:
        shutil.copyfile(prefx + '/' + chan + '/' + t,prefx + '/' + chan + '/labels' + '/' + t)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # sagemaker-containers passes hyperparameters as arguments
    parser.add_argument("--img-size", type=int, default=640,required=False)
    parser.add_argument("--batchsize", type=int,default=16,required=True)
    parser.add_argument("--epochs", type=int,default=500,required=True)
    parser.add_argument("--weights", type=str,required=False)
    parser.add_argument("--freeze", type=int,required=False)
    parser.add_argument("--patience",type=int,required=False)
    parser.add_argument("--hyp",type=str,required=False)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument('--name', type=str,required=False)

    # YOLOv7 HPO arguments, apart from batch and epochs etc. considered above
    # We will use the data/hyps/hyp.scratch.p5.yaml as the source of all default values.
    parser.add_argument("--lr0",type=float,required=False)
    parser.add_argument("--lrf",type=float,required=False)
    parser.add_argument("--momentum",type=float,required=False)
    parser.add_argument("--weight_decay",type=float,required=False)
    parser.add_argument("--warmup_epochs",type=int,required=False)
    parser.add_argument("--warmup_momentum",type=float,required=False)
    parser.add_argument("--warmup_bias_lr",type=float,required=False)
    parser.add_argument("--box",type=float,required=False)
    parser.add_argument("--cls",type=float,required=False)
    parser.add_argument("--cls_pw",type=float,required=False)
    parser.add_argument("--obj",type=float,required=False)
    parser.add_argument("--obj_pw",type=float,required=False)
    parser.add_argument("--iou_t",type=float,required=False)
    parser.add_argument("--anchor_t",type=float,required=False)
    parser.add_argument("--fl_gamma",type=float,required=False)
    parser.add_argument("--hsv_h",type=float,required=False)
    parser.add_argument("--hsv_s",type=float,required=False)
    parser.add_argument("--hsv_v",type=float,required=False)
    parser.add_argument("--degrees",type=float,required=False)
    parser.add_argument("--translate",type=float,required=False)
    parser.add_argument("--scale",type=float,required=False)
    parser.add_argument("--shear",type=float,required=False)
    parser.add_argument("--perspective",type=float,required=False)
    parser.add_argument("--flipud",type=float,required=False)
    parser.add_argument("--fliplr",type=float,required=False)
    parser.add_argument("--mosaic",type=float,required=False)
    parser.add_argument("--mixup",type=float,required=False)
    parser.add_argument("--copy_paste",type=float,required=False)

    args = parser.parse_args()
    ## This covers two channels - train, and val
    separateimages('/opt/ml/input/data','train')
    separateimages('/opt/ml/input/data','val')

    cmd = buildCMD(args)
    ## Standard Amazon SageMaker paths, you could add a test path as well
    cmd += " --data " + buildDataConfig('/opt/ml/input/data/train','/opt/ml/input/data/val')

    ## get weights from S3, if we need it, else we just leave it to the YOLOv7 train.py
    ## to get the weights from the official github repo.
    if args.weights:
        s3 = boto3.resource('s3')
        bucketname = (args.weights).split('/')[2]
        key = (args.weights).split(bucketname)[-1]
        cmd += " --weights " + "/yolov5/" + key.split('/')[-1]
        try:
            s3.Bucket(bucketname).download_file(key[1:], '/yolov5/' + key.split('/')[-1])
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
    print("**Data Configuration**:")
    with open(DATACFGFILE) as f:
        print(f.read())
    print("**Hyperparameters Configuration**:")
    with open(SMHYPFILE) as f:
        print(f.read())
    print("**Executing** : {}".format(cmd))
    status = os.system(cmd) ## We will replace this with subprocess.Popen once we see that this works
    if status != 0:
        print("Training failure!")
    else:    
        print("Training succeeded!")
