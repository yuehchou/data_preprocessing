import os
import io
import sys
import yaml
import boto3
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import OneHotEncoder

def load_cfg(config_fpath):
    if not os.path.splitext(config_fpath)[-1] in ('.yaml', '.yml'):
        print("[Error] config file should be yaml file!")
        sys.exit()
    else:
        print("Load the config file {}\n".format(config_fpath))
        with open(config_fpath, 'r') as ymlfile:
            cfg = yaml.load(
                ymlfile,
                Loader=yaml.FullLoader
            )
    return cfg

def pd_read_s3_parquet(key, bucket, s3_client=None, **args):
    if s3_client is None:
        s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()), **args)
    
def pd_read_s3_multiple_parquets(filepath, bucket, s3=None, 
                                 s3_client=None, verbose=False, **args):
    if not filepath.endswith('/'):
        filepath = filepath + '/'  # Add '/' to the end
    if s3_client is None:
        s3_client = boto3.client('s3')
    if s3 is None:
        s3 = boto3.resource('s3')
    s3_keys = [item.key for item in s3.Bucket(bucket).objects.filter(Prefix=filepath)
               if item.key.endswith('.parquet')]
    if not s3_keys:
        print('No parquet found in', bucket, filepath)
    elif verbose:
        print('Load parquets:')
        for p in s3_keys: 
            print(p)
    dfs = [pd_read_s3_parquet(key, bucket=bucket, s3_client=s3_client, **args) 
           for key in s3_keys]
    return pd.concat(dfs, ignore_index=True)

def check_timestamp(string, check_format="%Y-%m-%d %H:%M:%S.%f"):
    try:
        # If this line doesn't throw an Error, it's indeed a timestamp in proper format.
        dt.datetime.strptime(string, check_format)
        return True
    except (ValueError, TypeError):
        return False

def datestr2int(datestring, date_format="%Y-%m-%d %H:%M:%S.%f"):
    d = dt.datetime.strptime(datestring, date_format)
    if d.minute >= 30:
        d = d + dt.timedelta(minutes=30)
    else:
        pass
    s = dt.datetime.strftime(d, '%Y%m%d%H')
    return int(s)
    
def onehot_enc_array(str_array):
    enc = OneHotEncoder()
    enc.fit(str_array.values.astype(str).reshape(-1,1))
    return enc.transform(str_array.values.astype(str).reshape(-1,1))