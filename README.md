# Epidemiological Prediction using Deep Learning

Epidemic prediction over the world has been an important problem for public health. Periodically detecting the new changes and trends in epidemiology is very essential and this has increased the attention of data mining and machine learning communities. Timely detection, tracking and forecasting of key information of epidemics is very crucial and could help to tackle many health problems.
AI ways to deal with displaying epidemiologic information are getting progressively more predominant in the writing. These techniques can possibly improve our comprehension of wellbeing and openings for mediation, a long way past our past capacities.

### Approach

We intend to use Recurrent Neural Networks (RNNs) to capture the long-term correlation in the data and Convolutional Neural Networks (CNNs) to fuse information from data of different sources and a residual structure is also applied to prevent overfitting issues in the training process.

### Instructions for execution

```bash <shell script> <data> <adjacency matrix> <type of data> <gpu> <normalization>```

```shell script``` is the path for the required shell script, ```data``` is the dataset being considered, ```adjacency matrix``` is the matrix designed for the respective dataset, ```type of data``` is basically a log name for the data, ```gpu``` is for mentioning gpu number and ```normalization``` is for choosing the type of normalization.

Example: ```bash ./sh/grid_CNNRNN_Res.sh ./data/us_hhs/data.txt ./data/us_hhs/ind_mat.txt hhs 0 1```

GPU is required for the execution. Hence, [Google Colab](https://colab.research.google.com/) can be used where GPU can be levearaged for free.

#### Steps for execution in Google Colab:
* Upload the entire repository folder onto drive (same account for which you plan to use colab)
* Create a new notebook on Google Colab
* "Change Runtime" to GPU
* Use the "Mount Drive" option available in the Google Colab gui and follow the prompted steps
* Use the below script to change the directory
  * ```import os```
  * ```os.chdir('/content/drive/MyDrive/CNNRNNres')```
* Next line of code for execution
  * ```!bash ./sh/grid_CNNRNN_Res.sh ./data/us_hhs/data.txt ./data/us_hhs/ind_mat.txt hhs 0 1```
  
#### RMSE is the metric used for calculating the accuracy of the model.
