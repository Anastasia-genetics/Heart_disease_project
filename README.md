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

3. Data processing.
   According to df.describe() method there is no much artifacts in data. But there can be some outliyers for quantitive data. I have used box with whisker to visualize outliyers. Example data:
   
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

4. Models developmet.
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

#Random Forest Classifier
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

   


