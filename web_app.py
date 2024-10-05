from flask import Flask, render_template, Response,jsonify,request
#from keras.models import load_model
import numpy as np

import os 
import shutil
import numpy as np
import wfdb
import hrvanalysis
import matplotlib.pyplot as plt
from wfdb import get_record_list,plot_wfdb
from wfdb.io import rdheader,rdann,rdrecord
from wfdb.processing import normalize_bound
import  ecgdetectors
import scipy
from scipy import signal
import pandas as pd
from scipy.signal import hilbert,find_peaks
from PyEMD import EMD,Visualisation,CEEMDAN
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import os
import glob
from scipy.stats import skew
from hrvanalysis import get_frequency_domain_features,get_csi_cvi_features,get_geometrical_features,get_poincare_plot_features
from scipy.stats import skew
from scipy.stats import zscore
import heartpy
from scipy.interpolate import splev, splrep
import pickle
import numpy as np
import os
from keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.layers import Dense,Flatten,MaxPooling2D,Conv2D
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
    
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)



@app.route('/')
def index():
    test_signal=['c03','c05','x02','x05','x08','x09','x10','x11','x12','x13','x14','x15','x16','x19','x21','x23','x27','x30','x32']
    print("length",len(test_signal))
    k=-1
    for i in test_signal:
        k+=1
    print('hello',k)    
    lenth=len(test_signal)
    print(len)
    return render_template('home.html',test=test_signal,len=k)



def get_signal_from_dir(file_path):
    #ann=rdann(record_name=file_path,extension='apn',pn_dir=None)
    #print(ann)
    signals, fields = wfdb.rdsamp(file_path,pn_dir=None)
    fs=fields['fs']
    T=1/fs
    t=np.arange(0,fields['sig_len'])*T
    #t=np.arange(0,fields['sig_len'])*T
    signals=200.*signals
    signals=(signals-np.min(signals))/(np.max(signals)-np.min(signals))
    signals=pd.DataFrame(signals)[0].values
    signals=heartpy.remove_baseline_wander(signals,fs,cutoff=0.05)
    #signals= heartpy.filtering.filter_signal(signals, cutoff = 50, sample_rate = 100.0, filtertype='notch')
    return signals,t

def segment_signal(signals,ann,t):
    segmented_series=[]
    time_series=[]
    for i in ann:
        #print(i)
        temp=signals[i:i+6000]
        T_im=t[0:6000]
        segmented_series.append(temp)
        time_series.append(T_im)
    return segmented_series,time_series

def peaks(signals,t):
    #emd=EMD()
    #IMFs=emd(signals,t)
    #imfs,res=emd.get_imfs_and_residue()
    peaks,_=find_peaks(signals,height=np.max(signals)/2)
    rr=np.diff(t[peaks])
    amp=signals[peaks]
    R_amp=amp
    rr_tm=t[peaks]
    rr_tm=rr_tm[0:len(rr)]
    R_amp=R_amp[0:len(rr)]
    return peaks,rr,R_amp,rr_tm

time_range=60
ir=3
def hrv_signal(rr,amp,rr_tm):
    #rr=remove_outliers(rr,low_rri=np.mean(rr)-2*np.std(rr),high_rri=np.mean(rr)+2*np.std(rr))
    #rr=interpolate_nan_values(rr,interpolation_method='linear')
    tm = np.arange(0, (time_range), step=(1) / float(ir))
    interpolated_rr=splev(tm,splrep(rr_tm,rr,k=3),ext=0)
    interpolated_ramp=splev(tm,splrep(rr_tm,amp,k=3),ext=0)
    if interpolated_rr[0]<0:
        interpolated_rr[0]=-1*interpolated_rr[0]
    else:
        interpolated_rr[0]=interpolated_rr[0]
    #net_input=np.concatenate((interpolated_ramp+(np.mean(interpolated_rr)),interpolated_rr+(np.mean(interpolated_ramp))),axis=0)
    #net_input=np.concatenate((interpolated_rr+(np.mean(interpolated_ramp)),interpolated_ramp+(np.mean(interpolated_rr))),axis=0)
    #print(np.mean(interpolated_rr)-np.mean(interpolated_ramp))
    #print(np.mean(interpolated_ramp))
    interpolated_rr=interpolated_rr-np.min(interpolated_rr)/(np.max(interpolated_rr)-np.min(interpolated_rr))
    #interpolated_ramp=interpolated_ramp-np.mean(interpolated_ramp)/(np.std(interpolated_ramp))
    return interpolated_rr, interpolated_ramp
R_amp=None
R_Rt=None
Rt_amp=None
segmented_series=None

@app.route('/uploads', methods=['POST'])
def upload():
    print('#'*50)
    data = request.form
    data_input = data['test_input']
    print(data_input,data_input,data_input)

    global R_amp
    global R_Rt
    global Rt_amp
    global segmented_series

    #r=os.path.join('D:\sleep_apnea\apnea-ecg-database-1.0.0','a01')
    train_name=[data_input]
    R_amp=pd.DataFrame()
    r_r=pd.DataFrame()
    ann_d=pd.DataFrame()
    #df=pd.DataFrame(columns=['input_feature','Target'])
    #df=pd.DataFrame()
    #net_input=[]
    for i in train_name:
        file=os.path.join(r'../M1/apnea-ecg-database-1.0.0',i)
        print(file)
        [signals,time]=get_signal_from_dir(file)
        ann_data=np.load(r'../M3/ann_index.npy')
        print(ann_data)
        [segmented_series,time_series]=segment_signal(signals,ann_data,time)

        #deep_input=[]
        net_input=[]
        for j in range(0,len(segmented_series)):
            
            loc,rr,amp,rr_tm=peaks( segmented_series[j],time_series[j])
            
            interpolated_rr, interpolated_ramp=hrv_signal(rr,amp,rr_tm)
           
            R_amp.loc[:,j]=pd.DataFrame(pd.Series(interpolated_ramp))
            r_r.loc[:,j]=pd.DataFrame(pd.Series(interpolated_rr))
            #ann_d.loc[:,j]=pd.DataFrame(pd.Series(S))
        R_Rt=r_r.transpose()
        Rt_amp=R_amp.transpose()

        print(R_Rt)
        print(R_Rt.shape[0])
        #print(R_Rt.shape(0),R_Rt.shape(0),R_Rt.shape(0),R_Rt.shape(0),R_Rt.shape(0))    
        
    # return jsonify({ 'status':'success','message': ' received successfully'})
    return render_template('home.html',size=R_Rt.shape[0],test=data_input)       



@app.route('/predict', methods=['POST'])
def predict():
    global R_amp
    global R_Rt
    global Rt_amp
    global segmented_series

    i=int(request.form['number'])
    plt.figure(figsize=(10, 6))
    plt.plot(segmented_series[i])  # Assuming this is how you generate your plot

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
  


    plot_url = base64.b64encode(img.getvalue()).decode()

    
    
    print(i,i,i,i,i,i)
    x=[Rt_amp.iloc[i,:],R_Rt.iloc[i,:]]
    x=np.array(x,dtype='float32')
    x=np.expand_dims(x,0)
    x=np.expand_dims(x,2)
    #x=np.array(x,dtype='float32')
    #x=np.array(x, dtype="float32").transpose((0,3,1,2))
    x = np.transpose(x, (0, 3, 2, 1))
    from tensorflow.keras.models import load_model,save_model

    model=load_model(r'../M3/M3/model_VGG16.h5')
    y_score = model.predict(x)
    y_predict= np.argmax(y_score, axis=-1)
    if y_predict==1:
        print('normal')
        result="Normal"
    else:
        print('Sleep apnea')
        result="Sleep apnea"



    return render_template('home.html',result=result,plot_url=plot_url)



if __name__ == '__main__':
    
    app.run(debug=True)