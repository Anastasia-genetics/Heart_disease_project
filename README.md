# Heart disease project

This project shows development of Machine Learning and Neural Network models for heart diseseases prognosis.
Nowadays computer technologies are developing very fast in different areas of humen life. With development of computational computer resources it became more easy to work with big data. Modern medicine is also develoiping very fast and includes many new and well-known methods of analysis. Accessibility of medicine improves quality of analysis and prognosis for patients. Now personilized medical approach is becomeing more common as well as preventive medicine. It is very convenient to have easy to use and reliable tools for comprehensive patient's health analysis. This tool can help doctors to pay more attention to individual profile of patients and their susceptibility to diseases. Modern machine learning models can help to develop such tools (Chang et al., 2022).

# Aim
The research aims to prognose patient's heart disease using results from database from other patients with a help of python tools of Machine Learning and Deep Learning.

Machine Learning (ML) is important in predicting the existence or absence of heart arrhythmia, locomotor disorders, heart diseases, and other conditions. It was expected well to provide significant insights to physicians, allowing them to adjust their diagnosis and care on a patient-by-patient basis (Chang et al., 2022). In this project I used linear regression model, random forest classifier, catboost algorithm and simple tensorflow network for binary classification.

# Research workflow
1. I have uploaded csv tables with data from 599999 patients with informatio rows and 14 parameters.
   A link for the dataset is: https://www.kaggle.com/competitions/tech-weekend-data-science-hackathon/data
   
2. Data analysis.
   Dataset includes following patient's information:
   
   -- 1. ID (unique identifier)
   
   -- 2. age
   
   -- 3. sex
   
   -- 4. chest pain type (4 values)
   
   -- 5. resting blood pressure
   
   -- 6. serum cholestoral in mg/dl
   
   -- 7. fasting blood sugar > 120 mg/dl
   
   -- 8. resting electrocardiographic results (values 0,1,2)
   
   -- 9. maximum heart rate achieved
   
   -- 10. exercise induced angina
   
   -- 11. oldpeak = ST depression induced by exercise relative to rest
   
   -- 12. the slope of the peak exercise ST segment
   
   -- 13. number of major vessels (0-3) colored by flourosopy
   
   -- 14. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
   
Dataframe doesn't have any duplicates, missing data (NA) or artifacts.
<img width="578" height="673" alt="image" src="https://github.com/user-attachments/assets/09c4cdcb-fa91-4137-b5e9-c4f4113db9a9" />

4. Data processing.
   According to df.describe() method there is no much artifacts in data. Mean and Average are more or less at the same level for quantitive data, that means that it's normal distribution. But there can be some outliyers for quantitive data. I have used box with whisker to visualize outliyers. Example data:
   
   <img width="571" height="416" alt="image" src="https://github.com/user-attachments/assets/b767a588-3d5f-485e-aa67-d9f311c006e8" />

   After I used Interquartile range method (IQR) to clean data from outliyers.

   After I proceeded with correlation analysis that is necessary for each model. In case there are any high correlations we should delete one of the features that correlates with the other one.
   For quantitive data there were no high correlations found
   <img width="977" height="887" alt="image" src="https://github.com/user-attachments/assets/f2b2ea02-4557-4226-88ef-8949e8a7ebcc" />
   For all data including categorial statistical algorithm has found quite high correlations (slope - oldpeack, class - thal >0.5 correlation level). 
   <img width="1010" height="919" alt="image" src="https://github.com/user-attachments/assets/a4b0f2b4-1fd8-467d-8755-a2bb025f26f4" />

   To be on safe side and make balanced ML models I have decided to use PCA (Principal Component Analysis) to get rid of correlations effect.
   Also for better and more reliable model work I have used StandaredScaler standartization of data.
   All categorial data are already in proper integer format so I haven't made any encoding of data.

5. Models developmet.
   The aim is to make binary classification of patients depending on their analysis results. For discrimination I used data in column "class" (as y).
   # Linear regression model
   Model showed following metrics after learning and testing:
   
   Metrics for train data:
Accuracy: 0.6852369079930148
Precision: 0.669494454394068
Recall: 0.5234126959757798
F1-score: 0.5875090902197851
Metrics for test data:
Accuracy: 0.6877703934166682
Precision: 0.6734360667767382
Recall: 0.5260073365048432
F1-score: 0.5906611530354471

There are no signs of overfitting because results for train and test data are equal. But accuracy and recall of model are not very good. In general this model is not very reliable.
After PCA we have only 3 features left and those features are not the same as in initial data because they include other features that were processed during PCA application.
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/8a59dfbe-a443-4c9f-8d73-be965d1a988c" />

# Random Forest Classifier
Model showed following metrics after learning and testing:

Metrics for train data:
Accuracy: 0.6928007111447748
Precision: 0.6540789138962899
Recall: 0.600009825656016
F1-score: 0.6258787958375105
Metrics for test data:
Accuracy: 0.6941524451235602
Precision: 0.6561805142338166
Recall: 0.6004613973749069
F1-score: 0.627085671530116

There are no signs of overfitting because results for train and test data are equal. Accuracy and recall of model are better then for linear regression. But still this model is not very reliable.
I am not satisfied with results of the model and try optimization with a help of Optuna library. Score for optimmization will be 'Recall' because it's oblicatory to have as low amount of False Negatives for medical prognosis as possible. I have tried small amount of trials with Optuna due to linitations of my local computer.
After Optuna optimization best parameters have been found on grid. Metrics for Random Forest Classifier model with new parameters became:
Metrics for train data:
Accuracy: 0.7189452755160007
Precision: 0.6918447227534772
Recall: 0.6197962404583669
F1-score: 0.6538416688261207
Metrics for test data:
Accuracy: 0.6936983375982622
Precision: 0.6594205223641147
Recall: 0.5889694503353012
F1-score: 0.6222070844686648

Optuna algorithm of optimization haven't change much our results. Results for test data started to be even worse.

# Catboost model
CatBoost algorithm is well-known high-performance gradient boosting library for decision trees classifier. It is specially dedicated to work with categorial data without additional data transformation. I decided to try this algorithm on our initial data before PCA application. 
Metrics for train and test data with initial data are much better than data after PCA algorithm.

Metrics for train data:
Accuracy: 0.9111253129668203
Precision: 0.9035966722962407
Recall: 0.8871216354804439
F1-score: 0.8952833667483708
Metrics for test data:
Accuracy: 0.903502150874157
Precision: 0.8931911736897992
Recall: 0.8798933914139967
F1-score: 0.8864924171881879

Let's have a look on features importance chart:

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/ea7ad20e-5be7-44ae-8b82-d57d677a35a8" />


If we compare this feature importance data with correlation matrix we'll see that most influence on model have more correlated features vs class from correlation matrix that is logical. Even though in general for ML models we need to avoid features with high correlations for this particular case such model with more correlated features works much better and more reliable in all metrics. 
# Top-5 prognostic features for heart disease are: 'maximum heart rate achieved', 'thal', 'number_of_major_vessels', 'chest', 'age'



# Neural Network based on TensorFlow library
According to literature even simple neaural networks can solve classification tasks with better scores than classical ML models. I have tried to develop 3-layers simple network with Adam optimization on initial data like I used in CatBoost model.
Here is a structure of model used:
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 256)            │         3,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1)              │           129 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 
 Total params: 36,609 (143.00 KB)
 
 Trainable params: 36,609 (143.00 KB)
 
 Non-trainable params: 0 (0.00 B)

During several trials I have revealed that learning_rate=0.002 in Adam optimization is better for 10 epochs of analysis (model faster can find minimum loss and highest recall). I used 10 epochs due to limitations of local computer.

Metrics for train data:
Accuracy:  0.9010
Precision: 0.8933
Recall:    0.8733
F1-score:  0.8831
Metrics for test data:
Accuracy:  0.9005
Precision: 0.8919
Recall:    0.8735
F1-score:  0.8826

Results of our network showed almost the same metrics as CatBoost model on test data.
We also had second dataset in study. I have made a classes prediction for this dataset with a help of neural network. I have saved final model in file heart_desease_classifier.keras.


# Summary

Machine Learning and Deep Learning are powerfull tools for disease prognosis. During this stude I have revealed that data transformation should bve made very careful as it has a direct influence on model quality. If it is possible it is good to try several variants with and withoud additional transformations.
In our study CatBoost model and Tensorflow 3-layers neural network worked the best. Main features that have an influence on model and final prognosis are: 'maximum heart rate achieved', 'thal', 'number_of_major_vessels', 'chest', 'age'.

