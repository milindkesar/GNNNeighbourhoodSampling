import json
import os
def construct_agg(dir=None):
    try:
        os.makedirs(dir+'agg')
    except:
        pass
    sub_dirs=[x[0] for x in os.walk(dir)]
    print(sub_dirs)
    agg_training_info_l=[]
    agg_test_l=[]
    bigdata=[]
    best_test_f1={'epoch':-1,'test_f1':-1}
    for sub_dir in sub_dirs[1:]:
        if 'agg' in sub_dir:
            continue
        data=[]
        with open(sub_dir+'/test.txt') as f:
            for line in f:
                data.append(json.loads(line))
        bigdata.append(data)
    for i in range(len(bigdata[0])):
        agg_training_info = {'train_loss': [], 'val loss': []}
        agg_test = {'epoch':bigdata[0][i]['epoch'],'test_f1': []}
        for j in range(len(bigdata)):
            agg_test['test_f1'].append(bigdata[j][i]['test_f1'])
        agg_test_l.append(agg_test)
    for item in agg_test_l:
        avg_test_f1=sum(item['test_f1'])/len(item['test_f1'])
        item['test_f1'] = avg_test_f1
        if avg_test_f1 > best_test_f1['test_f1']:
            best_test_f1['test_f1']=avg_test_f1
            best_test_f1['epoch']=item['epoch']
    with open(dir+'/agg'+'/test.txt','a+') as out:
        for item in agg_test_l:
            out.write(json.dumps(item)+'\n')
    with open(dir+'/agg'+'/best.txt','a+') as out:
        out.write(json.dumps(best_test_f1))
construct_agg('/home/thummala/graphsage-pytorch/res/wikics/')
