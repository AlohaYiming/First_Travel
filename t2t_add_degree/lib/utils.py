import os
import numpy as np
import time
import random
from random import choice
from lib import models_siamese,graph
from scipy.sparse import csr_matrix

from sklearn.cross_validation import train_test_split
scores = []
def build_tx_graph():
    start = time.clock()
    file=open('./data/txt_graph.txt','r+')
    line=file.readline().strip(' \n')
    graph=np.zeros((10055,10055),dtype=np.int)
    y=0
    while line:
        new_col=np.array(line.split(' '))
        for i in new_col:
            graph[y][int(i)]=1
        y=y+1
        line=file.readline().strip('\n')
    end=time.clock()
    print('build txt graph cost time:',(end-start)*1000,'ms')
    return graph

def prepair_data():
    # n_x0,n_y0=txt_arr('./data/txt_list.npy')
    # np.save('./data/n_load_nx0.npy',n_x0)
    # np.save('./data/n_load_ny0.npy',n_y0)
    #n_x1,n_y1=img_arr('./data/img_list.npy')
    #n_x1=np.expand_dims(n_x1,axis=2)
    #np.save('./data/load_nx1.npy',n_x1)
    #np.save('./data/load_ny1.npy',n_y1)

    # n_x0=np.load('./data/txt_w2c.npy').astype(float)
    #print('111',n_x0.shape)
    n_x0=np.load('./data/load_nx0.npy').astype(float)
    # cr_n_x0 = np.expand_dims(coar_n_x0(n_x),axis=2)
    # np.save('./data/load_coar_nx0.npy',cr_n_x0)
    # n_x0=np.load('./data/load_coar_nx0.npy')
    n_y0=np.load('./data/load_ny0.npy').astype(int) - 1
    n_x1=np.load('./data/load_nx1.npy').astype(float)
    n_y1=np.load('./data/load_ny1.npy').astype(int) - 1


    x_x0=n_x0[693:,:,:]
    x_x1=n_x1[693:,:,:]
    #x_x1=np.load('./data/train_imgout.npy').astype(float)
    x_y0=n_y0[693:]
    x_y1=n_y1[693:]
    c_x0=n_x0[:693,:,:]
    c_x1=n_x1[:693,:,:]
    #c_x1=np.load('./data/test_imgout.npy').astype(float)
    c_y0=n_y0[:693]
    c_y1=n_y1[:693]
    return x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,indptr=array.indptr, shape=array.shape)
# print(np.max(perm))
def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
def coar_n_x0(n_x0):
    perm = np.load('./data/coar_perm.npy')
    n_x0 = np.squeeze(n_x0)
    all = []
    for i in range(n_x0.shape[0]):
        new_arr = np.zeros([int(perm.shape[0]/2),2],dtype='float32')
        p=0
        while p < int(perm.shape[0]/2):
            if perm[p]<10055 :
                new_arr[p][0] = n_x0[i][perm[p]]
            else :
                new_arr[p][0] = n_x0[i][0]
            if perm[p+1]<10055:
                new_arr[p][1] = n_x0[i][perm[p+1]]
            else:
                new_arr[p][1] = n_x0[i][0]
            p += 2
        arr = np.sum(new_arr,axis=1)
        all.append(arr)
    return np.array(all)

def txt_arr(filename):
    start=time.clock()
    arr=np.load(filename)
    y=[]
    mat=[]
    for i in range(0,arr.shape[0]):
        y.append(arr[i][1])
        brr=np.loadtxt('./mat/'+arr[i][0]+'.mat')
        mat.append(brr)
    print(np.array(mat).shape,np.array(y).shape)
    # n,m=np.array(mat).shape[0],np.array(mat).shape[1]
    end=time.clock()
    print('read txt matrix cost time:',end-start)
    return np.array(mat),np.array(y)
def img_arr(filename):
    start=time.clock()
    arr=np.load(filename)
    y=[]
    mat=[]
    for i in range(0,arr.shape[0]):
        y.append(arr[i][1])
        brr=np.loadtxt('./mat/'+arr[i][0]+'.mat')
        print(i,':',brr.shape)
        mat.append(brr)
        # print(np.array(mat).shape)
    # print(np.array(mat).shape,np.array(y).shape)
    end=time.clock()
    print('read img matrix cost time:',end-start)
    return np.array(mat),np.array(y)
def get_list(filename):
    file=open(filename,'r+')
    label=[[],[],[],[],[],[],[],[],[],[]]
    row=0
    line=file.readline()
    while line:
        cla=int(line.strip('\n').split('\t')[2])
        label[cla-1].append(row)
        row+=1
        line=file.readline()
    return label
def make_index(n1,n2,dest):
    n2=(n2*11)/10
    if dest==0:
        n1=n1-2000
        filename='./data/trainset_txt_img_cat.list'
        r1=[i for i in range(2173)]
        r2=[i for i in range(2173)]
    if dest==1:
        n1=n1-1000
        filename='./data/testset_txt_img_cat.list'
        r1=[i for i in range(693)]
        r2=[i for i in range(693)]
    list=get_list(filename)
    ind=[]
    for i in range(10):
        # print(choice(list[i]))
        r1+=[choice(list[i]) for _ in range(int(n1/10))]
        r2+=[choice(list[i]) for _ in range(int(n1/10))]
    for p in range(10):
        for q in range(10):
            if p!=q:
                r1+=[choice(list[p]) for _ in range(int(n2/100))]
                r2+=[choice(list[q]) for _ in range(int(n2/100))]
    ind.append(r1)
    ind.append(r2)
    arr=np.array(ind)
    return arr    
def index_make(x0,x1,N1,N2,type):
    start=time.clock()
    y0=x0.tolist()
    y1=x1.tolist()
    list,id0,fir,sec=[],[],[],[]
    n1,n2=0,0
    state = False
    while state != True:
        if type==0:
            r0=random.randint(693,len(y0)-1)
            r1=random.randint(693,len(y1)-1)
        else:
            r0=random.randint(0,693)
            r1=random.randint(0,693)
        if y0[r0] == y1[r1] and n1<N1:
            id0.append([r0,r1])
            n1+=1
        if y0[r0] != y1[r1] and n2<N2:
            id0.append([r0,r1])
            n2+=1
        if n1>N1-1 and n2>N2-1:
            state=True
    for i in id0:
        fir.append(i[0])
        sec.append(i[1])
    list.append(fir)
    list.append(sec)
    return np.array(list)
    end=time.clock()
    print('make train index cost time:',end-start)
    return arr
def index_shuffle(index):
    N,M=index.shape
    list_all=[]
    for i in range(0,int(M)):
        list1=[]
        list1.append(index[0][i])
        list1.append(index[1][i])
        list_all.append(list1)
    random.shuffle(list_all)
    list_re,fir,sec=[],[],[]
    for y in list_all:
        fir.append(y[0])
        sec.append(y[1])
    list_re.append(fir)
    list_re.append(sec)
    return np.array(list_re)
def out_tsne(c_data, x_data, c_y, x_y, type, model):
    # search code
    name=['txt2img','img2txt']
    n1, n2 = c_y.shape[0], x_y.shape[0]
    b_size = 256
    map = 0
    n_map = 0
    start_to = time.time()
    list_auc = []
    index = [i for i in range(0, n2)]
    step = int(n1 / b_size)
    print('step',step)
    re_step = int(n1 % b_size)
    print('re_step',re_step)
    for i in range(0, step):
        start0 = time.time()
        begin = i * b_size
        end = begin + b_size
        res = model.search(c_data[begin:end, :, :], x_data[begin:end, :, :],type)
        if i == 0: 
            fea0 = res[0]
            fea1 = res[1]
        else:
            fea0 = np.concatenate((fea0,res[0]),axis=0)
            fea1 = np.concatenate((fea1,res[1]),axis=0)
    if re_step != 0:
        res = model.search(c_data[-256:, :, :], x_data[-256:, :, :],type)
        fea0 = np.concatenate((fea0,res[0][:re_step]),axis=0)
        fea1 = np.concatenate((fea1,res[1][:re_step]),axis=0)
    print('fea',fea0.shape)
    np.save('./tsne/'+name[type]+'-1.npy',fea0)
    np.save('./tsne/'+name[type]+'-2.npy',fea1)

        

def out_map(c_data, x_data, c_y, x_y, type, model):
    # search code
    name=['txt2txt','txt2txt']
    n1, n2 = c_y.shape[0], x_y.shape[0]
    b_size = 256
    map = 0
    n_map = 0
    start_to = time.time()
    list_auc = []
    index = [i for i in range(0, n2)]
    step = int(n2 / b_size)
    re_step = int(n2 % b_size)
    for i in range(0, n1):
        cor_num = 0
        p_num = 0
        start0 = time.time()
        li_res = []
        for p in range(0, step):
            begin = p * b_size
            end = begin + b_size
            res = model.search(c_data[[i] * b_size, :, :], x_data[begin:end, :, :], type)[0].tolist()
            scores.append(res)
            li_res += res
        if re_step != 0:
            res = \
            model.search(c_data[[i] * b_size, :, :], x_data[(step - 1) * b_size + re_step:step * b_size + re_step, :, :], type)[0]
            scores.append(res)
            res = res[-re_step:].tolist()
            li_res += res
        b = sorted(range(len(li_res)), reverse=True, key=li_res.__getitem__)
        # np.save('./result_save/'+name[type]+'-'+str(i)+'.npy',np.array(b))
        newlist = [int(i) for i in x_y[b]]
        # print(newlist)
        newarr = np.array(newlist)
        label = int(c_y[i])
        th_map = 0
        for n in range(newarr.shape[0]):
            if newarr[n] == label:
                cor_num += 1
                p_num += float(cor_num/(n+1))
                th_map = p_num / cor_num
        end0 = time.time()
        n_map += 1
        map += th_map
        np.save('all_score.npy',scores)
        print('The:', i, 'cost time:', int(end0 - start0), 's,total cost time:', int(end0 - start_to),
              's')
        if i != 0: print(name[type],'-total mean:', map / n_map)

def out_img(data1, data2, c_y, x_y, model):
    # search code
    name=['txt2txt','txt2txt']
    n1, n2 = c_y.shape[0], x_y.shape[0]
    b_size = 256
    map = 0
    n_map = 0
    start_to = time.time()
    list_auc = []
    index = [i for i in range(0, n2)]
    step = int(n2 / b_size)
    re_step = int(n2 % b_size)
    li_res = []
    for p in range(0, step):
        begin = p * b_size
        end = begin + b_size
        _,res = model.search(data1[0:256, :, :], data2[begin:end, :, :], 0)
        li_res.append(res)
    if re_step != 0:
        _,res = \
        model.search(data1[0:256, :, :], data2[(step - 1) * b_size + re_step:step * b_size + re_step, :, :], 0)
        res = res[-re_step:]
        li_res.append(res)
    out_arr = li_res[0]
    for i in range(1,len(li_res)):
        out_arr = np.concatenate([out_arr,li_res[i]],0)
    print(out_arr.shape)
    np.save('./data/train_imgout.npy',out_arr)
