https://www.cdc.gov/nchs/nhis/nhis_2016_data_release.htm

Downloaded Sample Adult file (CSV data)

survey summary
ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2016/srvydesc.pdf

To replicate the findings in this study follow the instructions below:  
    1) Clone the github repository and run all .ipynb files in Jupyter Notebook.   
    2) First run the cleaning of the dataset, [01_cleaning.ipynb](dslc_documentation/01_cleaning.ipynb).  
    3) If you would like to run the EDA analysis next run, [02_eda.ipynb](dslc_documentation/02_eda.ipynb).  
    4) To run the model and see the evaluation results next run, [03_prediction.ipynb](dslc_documentation/03_prediction.ipynb).  
    5) Once you have run the model in, [03_prediction.ipynb](dslc_documentation/03_prediction.ipynb), you can calculate a risk factor by loading in your data 
    and running the following line of code:  
    
    lgb_clf.predict_proba(*Insert_your_Data*)[:, 1]

For data documentation and some of the studies referenced in our paper see [Data Folder](data).
