
# Using H2O's Restful API in Python to Import Data, Train Models, and Export Models to MOJOs

Buidling a model scoring application in production using H2O is greatly benefited from its ability to export models to [MOJOs](http://www.highdimensional.space/2017/12/19/scoring-h2o-mojo-models-with-spark-dataframe-and-dataset/) which allows scoring large datasets in Spark without native H2O dependency. However, the model training process is still hard to productionize given the nature of how data scientists work vs engineering requirements to build a reliable system. H2O's Restful API provides a solution to standardize the model training performance by allowing the model training application to run from any environment without the need of installing H2O. The following script will demo how to make simple API calls to the H2O cluster (which could be run on a separate server) to import a sample csv file, parse the file, create a H2OFrame, train a gbm model, and export the model to a MOJO file.

### Prerequisites
#### Install H2O if you have not
In command line run:
```
pip install requests
pip install tabulate
pip install "colorama>=0.3.8"
pip install future
pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
```

**0a. Import libraries**



```python
import h2o
import requests
from requests.compat import urljoin, quote_plus
from time import sleep
h2o.init()
```

    Checking whether there is an H2O instance running at http://localhost:54321..... not found.
    Attempting to start a local H2O server...
      Java Version: java version "1.8.0_144"; Java(TM) SE Runtime Environment (build 1.8.0_144-b01); Java HotSpot(TM) 64-Bit Server VM (build 25.144-b01, mixed mode)
      Starting server from /usr/local/lib/python3.7/site-packages/h2o/backend/bin/h2o.jar
      Ice root: /var/folders/8y/bj6bvwyx3qxb_y7t17ztm78j3sf4sg/T/tmp59ehn4bx
      JVM stdout: /var/folders/8y/bj6bvwyx3qxb_y7t17ztm78j3sf4sg/T/tmp59ehn4bx/h2o_liuji_started_from_python.out
      JVM stderr: /var/folders/8y/bj6bvwyx3qxb_y7t17ztm78j3sf4sg/T/tmp59ehn4bx/h2o_liuji_started_from_python.err
      Server is running at http://127.0.0.1:54321
    Connecting to H2O server at http://127.0.0.1:54321... successful.



<div style="overflow:auto"><table style="width:50%"><tr><td>H2O cluster uptime:</td>
<td>01 secs</td></tr>
<tr><td>H2O cluster timezone:</td>
<td>America/New_York</td></tr>
<tr><td>H2O data parsing timezone:</td>
<td>UTC</td></tr>
<tr><td>H2O cluster version:</td>
<td>3.20.0.6</td></tr>
<tr><td>H2O cluster version age:</td>
<td>6 days </td></tr>
<tr><td>H2O cluster name:</td>
<td>H2O_from_python_liuji_cmxjh8</td></tr>
<tr><td>H2O cluster total nodes:</td>
<td>1</td></tr>
<tr><td>H2O cluster free memory:</td>
<td>3.556 Gb</td></tr>
<tr><td>H2O cluster total cores:</td>
<td>8</td></tr>
<tr><td>H2O cluster allowed cores:</td>
<td>8</td></tr>
<tr><td>H2O cluster status:</td>
<td>accepting new members, healthy</td></tr>
<tr><td>H2O connection url:</td>
<td>http://127.0.0.1:54321</td></tr>
<tr><td>H2O connection proxy:</td>
<td>None</td></tr>
<tr><td>H2O internal security:</td>
<td>False</td></tr>
<tr><td>H2O API Extensions:</td>
<td>XGBoost, Algos, AutoML, Core V3, Core V4</td></tr>
<tr><td>Python version:</td>
<td>3.7.0 final</td></tr></table></div>


**0b. Define variables**


```python
base_url = 'http://127.0.0.1:54321/3/'
import_url = urljoin(base_url, "ImportFiles")
parse_setup_url = urljoin(base_url, "ParseSetup")
parse_url = urljoin(base_url, "Parse")
jobs_url = urljoin(base_url, 'Jobs')
gbm_url = urljoin(base_url, 'ModelBuilders/gbm')
xgboost_url = urljoin(base_url, 'ModelBuilders/xgboost')
```

We need a function for job polling since when the model training starts we need to wait until it finishes to proceed to the next steps. The funciton will check the job status given the job id and will return the results once it's finished.


```python
# Define the helper function
def poll(job_key):
    while True:
        try:
            r = requests.get(jobs_url + '/' + quote_plus(job_key))
        except:
            print("Catched error")
            raise
        response = r.json()
        jobs = response['jobs']
        if len(jobs) != 1:
            raise RuntimeError('Could not find the job')
        else:    
            status = response['jobs'][0]['status']
            if status != 'RUNNING':
                print(status)
                break
            else:
                sleep(1)
                print("RUNNING")
                print('progress: {0}'.format(response['jobs'][0]['progress']))
    return response['jobs'][0]
```

**0c. Download the sample training file**

Dataset description: https://www.kaggle.com/blastchar/telco-customer-churn

The version of the data in this script includes a binary response variable created from the originial Churn variable.


```python
filepath = 'https://raw.githubusercontent.com/jeffreyliu3230/h2o-api-demo/master/telco_customer_churn.csv'
```

**2. Make a sample request for importing data using H2O's import API**


```python
import_params = {'path': filepath}
```


```python
r = requests.get(import_url, params=import_params)
```


```python
r.content
```




    b'{"__meta":{"schema_version":3,"schema_name":"ImportFilesV3","schema_type":"ImportFiles"},"_exclude_fields":"","path":"https://raw.githubusercontent.com/jeffreyliu3230/h2o-api-demo/master/telco_customer_churn.csv","pattern":null,"files":["https://raw.githubusercontent.com/jeffreyliu3230/h2o-api-demo/master/telco_customer_churn.csv"],"destination_frames":["https://raw.githubusercontent.com/jeffreyliu3230/h2o-api-demo/master/telco_customer_churn.csv"],"fails":[],"dels":[]}'




```python
# Retrieve content
import_result = r.json()
```


```python
destination_frames = import_result['destination_frames']
print(destination_frames)
```

    ['https://raw.githubusercontent.com/jeffreyliu3230/h2o-api-demo/master/telco_customer_churn.csv']


**3.Parse setup**


```python
# Specify the data parameters for setting up the data parsing step
data_params = {'source_frames': destination_frames}
```


```python
r = requests.post(parse_setup_url, data=data_params)
```


```python
parse_setup_result = r.json()
```


```python
parse_setup_result
```




    {'__meta': {'schema_version': 3,
      'schema_name': 'ParseSetupV3',
      'schema_type': 'ParseSetup'},
     '_exclude_fields': '',
     'source_frames': [{'__meta': {'schema_version': 3,
        'schema_name': 'FrameKeyV3',
        'schema_type': 'Key<Frame>'},
       'name': 'https://raw.githubusercontent.com/jeffreyliu3230/h2o-api-demo/master/telco_customer_churn.csv',
       'type': 'Key<Frame>',
       'URL': '/3/Frames/https://raw.githubusercontent.com/jeffreyliu3230/h2o-api-demo/master/telco_customer_churn.csv'}],
     'parse_type': 'CSV',
     'separator': 44,
     'single_quotes': False,
     'check_header': 1,
     'column_names': ['customerID',
      'gender',
      'SeniorCitizen',
      'Partner',
      'Dependents',
      'tenure',
      'PhoneService',
      'MultipleLines',
      'InternetService',
      'OnlineSecurity',
      'OnlineBackup',
      'DeviceProtection',
      'TechSupport',
      'StreamingTV',
      'StreamingMovies',
      'Contract',
      'PaperlessBilling',
      'PaymentMethod',
      'MonthlyCharges',
      'TotalCharges',
      'Churn',
      'ChurnResponse'],
     'column_types': ['String',
      'Enum',
      'Numeric',
      'Enum',
      'Enum',
      'Numeric',
      'Enum',
      'Enum',
      'Enum',
      'Enum',
      'Enum',
      'Enum',
      'Enum',
      'Enum',
      'Enum',
      'Enum',
      'Enum',
      'Enum',
      'Numeric',
      'Numeric',
      'Enum',
      'Enum'],
     'na_strings': None,
     'column_name_filter': None,
     'column_offset': 0,
     'column_count': 0,
     'destination_frame': 'telco_customer_churn1.hex',
     'header_lines': 0,
     'number_columns': 22,
     'data': [['7590-VHVEG',
       'Female',
       '0',
       'Yes',
       'No',
       '1',
       'No',
       'No phone service',
       'DSL',
       'No',
       'Yes',
       'No',
       'No',
       'No',
       'No',
       'Month-to-month',
       'Yes',
       'Electronic check',
       '29.85',
       '29.85',
       'No',
       '0'],
      ['5575-GNVDE',
       'Male',
       '0',
       'No',
       'No',
       '34',
       'Yes',
       'No',
       'DSL',
       'Yes',
       'No',
       'Yes',
       'No',
       'No',
       'No',
       'One year',
       'No',
       'Mailed check',
       '56.95',
       '1889.5',
       'No',
       '0'],
      ['3668-QPYBK',
       'Male',
       '0',
       'No',
       'No',
       '2',
       'Yes',
       'No',
       'DSL',
       'Yes',
       'Yes',
       'No',
       'No',
       'No',
       'No',
       'Month-to-month',
       'Yes',
       'Mailed check',
       '53.85',
       '108.15',
       'Yes',
       '1'],
      ['7795-CFOCW',
       'Male',
       '0',
       'No',
       'No',
       '45',
       'No',
       'No phone service',
       'DSL',
       'Yes',
       'No',
       'Yes',
       'Yes',
       'No',
       'No',
       'One year',
       'No',
       'Bank transfer (automatic)',
       '42.3',
       '1840.75',
       'No',
       '0'],
      ['9237-HQITU',
       'Female',
       '0',
       'No',
       'No',
       '2',
       'Yes',
       'No',
       'Fiber optic',
       'No',
       'No',
       'No',
       'No',
       'No',
       'No',
       'Month-to-month',
       'Yes',
       'Electronic check',
       '70.7',
       '151.65',
       'Yes',
       '1'],
      ['9305-CDSKC',
       'Female',
       '0',
       'No',
       'No',
       '8',
       'Yes',
       'Yes',
       'Fiber optic',
       'No',
       'No',
       'Yes',
       'No',
       'Yes',
       'Yes',
       'Month-to-month',
       'Yes',
       'Electronic check',
       '99.65',
       '820.5',
       'Yes',
       '1'],
      ['1452-KIOVK',
       'Male',
       '0',
       'No',
       'Yes',
       '22',
       'Yes',
       'Yes',
       'Fiber optic',
       'No',
       'Yes',
       'No',
       'No',
       'Yes',
       'No',
       'Month-to-month',
       'Yes',
       'Credit card (automatic)',
       '89.1',
       '1949.4',
       'No',
       '0'],
      ['6713-OKOMC',
       'Female',
       '0',
       'No',
       'No',
       '10',
       'No',
       'No phone service',
       'DSL',
       'Yes',
       'No',
       'No',
       'No',
       'No',
       'No',
       'Month-to-month',
       'No',
       'Mailed check',
       '29.75',
       '301.9',
       'No',
       '0'],
      ['7892-POOKP',
       'Female',
       '0',
       'Yes',
       'No',
       '28',
       'Yes',
       'Yes',
       'Fiber optic',
       'No',
       'No',
       'Yes',
       'Yes',
       'Yes',
       'Yes',
       'Month-to-month',
       'Yes',
       'Electronic check',
       '104.8',
       '3046.05',
       'Yes',
       '1']],
     'warnings': None,
     'chunk_size': 30768,
     'total_filtered_column_count': 22,
     'decrypt_tool': None}



Set the response variable to factor so that H2O will train a classification model instead of a regression model



```python
parse_setup_result['column_types'][-1] = 'Enum'
```

**4. Parse**


```python
parse_params = {'destination_frame': 'telco_customer_churn.hex',
                'source_frames': [parse_setup_result['source_frames'][0]['name']],
                'parse_type': parse_setup_result['parse_type'], 
                'separator': parse_setup_result['separator'],
                'number_columns': parse_setup_result['number_columns'],
                'single_quotes': parse_setup_result['single_quotes'],
                'column_names': parse_setup_result['column_names'],
                'column_types': parse_setup_result['column_types'],
                'check_header': parse_setup_result['check_header'],
                'delete_on_done': 'false',
                'chunk_size': parse_setup_result['chunk_size']}
```


```python
r = requests.post(parse_url, data=parse_params)
```


```python
parse_result = r.json()
```


```python
parse_result = poll(parse_result['job']['key']['name'])
```

    DONE



```python
parse_result
```




    {'__meta': {'schema_version': 3, 'schema_name': 'JobV3', 'schema_type': 'Job'},
     'key': {'__meta': {'schema_version': 3,
       'schema_name': 'JobKeyV3',
       'schema_type': 'Key<Job>'},
      'name': '$03017f00000132d4ffffffff$_a4d397e3ff1f81e0a0795bbe8eae4fab',
      'type': 'Key<Job>',
      'URL': '/3/Jobs/$03017f00000132d4ffffffff$_a4d397e3ff1f81e0a0795bbe8eae4fab'},
     'description': 'Parse',
     'status': 'DONE',
     'progress': 1.0,
     'progress_msg': 'Done.',
     'start_time': 1535769130566,
     'msec': 467,
     'dest': {'__meta': {'schema_version': 3,
       'schema_name': 'FrameKeyV3',
       'schema_type': 'Key<Frame>'},
      'name': 'telco_customer_churn.hex',
      'type': 'Key<Frame>',
      'URL': '/3/Frames/telco_customer_churn.hex'},
     'warnings': None,
     'exception': None,
     'stacktrace': None,
     'ready_for_view': True}



**5. Training**


```python
# Define columns used for training
train_columns = ['gender',
                 'SeniorCitizen',
                 'Partner',
                 'Dependents',
                 'tenure',
                 'PhoneService',
                 'MultipleLines',
                 'InternetService',
                 'OnlineSecurity',
                 'OnlineBackup',
                 'DeviceProtection',
                 'TechSupport',
                 'StreamingTV',
                 'StreamingMovies',
                 'Contract',
                 'PaperlessBilling',
                 'PaymentMethod',
                 'MonthlyCharges',
                 'TotalCharges',
                 'ChurnResponse']
ignored_columns = [x for x in parse_params['column_names'] if not x in train_columns]
```


```python
# Define model training parameters
gbm1_params = {'model_id': 'gbm_test',
               'response_column': 'ChurnResponse',
               'ignored_columns': ignored_columns,
               'training_frame': parse_result['dest']['name'],
               'distribution': "AUTO",
               'ntrees': 20,
               'max_depth': 8,
               'min_rows': 2,
               'learn_rate': 0.4,
               'nfolds': 5,
               "fold_assignment": "Stratified",
               'keep_cross_validation_predictions': 'true',
               'seed': 2018}
```


```python
r = requests.post(gbm_url, data=gbm1_params)
```


```python
train_result = r.json()
```


```python
train_result = poll(train_result['job']['key']['name'])
```

    RUNNING
    progress: 0.7083333
    DONE



```python
train_result
```




    {'__meta': {'schema_version': 3, 'schema_name': 'JobV3', 'schema_type': 'Job'},
     'key': {'__meta': {'schema_version': 3,
       'schema_name': 'JobKeyV3',
       'schema_type': 'Key<Job>'},
      'name': '$03017f00000132d4ffffffff$_92b96d9908738e6394a46a5620b64b1e',
      'type': 'Key<Job>',
      'URL': '/3/Jobs/$03017f00000132d4ffffffff$_92b96d9908738e6394a46a5620b64b1e'},
     'description': 'GBM',
     'status': 'DONE',
     'progress': 1.0,
     'progress_msg': 'Done.',
     'start_time': 1535769504759,
     'msec': 2044,
     'dest': {'__meta': {'schema_version': 3,
       'schema_name': 'ModelKeyV3',
       'schema_type': 'Key<Model>'},
      'name': 'gbm_test',
      'type': 'Key<Model>',
      'URL': '/3/Models/gbm_test'},
     'warnings': None,
     'exception': None,
     'stacktrace': None,
     'ready_for_view': True}




```python
# view model
r = requests.get(base_url + 'Models/gbm_test')
r.text[0:300]
```




    '{"__meta":{"schema_version":3,"schema_name":"ModelsV3","schema_type":"Models"},"_exclude_fields":"","models":[{"__meta":{"schema_version":3,"schema_name":"GBMModelV3","schema_type":"GBMModel"},"model_id":{"__meta":{"schema_version":3,"schema_name":"ModelKeyV3","schema_type":"Key<Model>"},"name":"gbm'



**6. Save the model to MOJO**


```python
r = requests.get(base_url + 'Models/gbm_test/mojo', stream=True)
```


```python
type(r.content)
```




    bytes




```python
with open('gbm_test.zip', 'wb') as f:
    for chunk in r.iter_content(8192):
        f.write(chunk)
```

The MOJO file could then be loaded in a spark application for scoring (http://www.highdimensional.space/2017/12/19/scoring-h2o-mojo-models-with-spark-dataframe-and-dataset/)
