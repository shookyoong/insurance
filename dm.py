import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import imblearn
from sklearn.cluster import KMeans 
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import silhouette_visualizer
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import  roc_auc_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
#cd desktop \trimester 1 2021\dm\dm project
#streamlit run project.py




df = pd.read_csv('Insurance_Data.csv')


df.columns = df.columns.str.upper()

st.sidebar.header('Section')
section = st.sidebar.radio("Choose a section:", 
                              ("EDA & Data Preprocessing", "Feature Selection", "Classification", "Clustering","Association Rule Mining")
                              )

if section == "EDA & Data Preprocessing":
    st.title('Insurance Product Recommendation')
    st.header("EDA  & Data Preprocessing")
    st.subheader("Original Dataframe")
    st.write(df)

    dftemp = df.copy()
    select_row = st.number_input('Select row: ', min_value=0,
                           max_value=len(dftemp), value=0, key=0)
    dftemp = dftemp.iloc[select_row]
    st.write(dftemp.astype('object'))

    st.subheader("Target Data")
    select_feature = st.radio('Select a feature:', ('Purchase Plan 1','Purchase Plan 2'))
    
    if select_feature == 'Purchase Plan 1':
        fig1 = plt.figure(figsize=(2,2))
        df["PURCHASEDPLAN1"].value_counts().sort_values(ascending=True).plot(kind='barh')
        st.pyplot(fig1)
                
    elif select_feature == 'Purchase Plan 2':
        fig2 = plt.figure(figsize=(2,2))
        df["PURCHASEDPLAN2"].value_counts().sort_values(ascending=True).plot(kind='barh')
        st.pyplot(fig2)
            
            
    table1 = pd.crosstab(df['CUSTOMER_NEEDS_1'], df['PURCHASEDPLAN1'])
    table2 = pd.crosstab(df['CUSTOMER_NEEDS_2'], df['PURCHASEDPLAN2'])

    st.subheader("Customer Needs VS Purchased Plans")
    select_feature2 = st.radio('Select a feature:', ('Customer needs 1 VS Purchased Plan 1','Customer needs 2 VS Purchased Plan 2'))

    if select_feature2 == 'Customer needs 1 VS Purchased Plan 1':
    
        table1.plot(kind='barh', title='Counts')
        st.pyplot()

    elif select_feature2 == 'Customer needs 2 VS Purchased Plan 2':    
    
        table2.plot(kind='barh', title='Counts')
        st.pyplot()


    
    df['SALARY(MONTH)'] = df['ANNUALSALARY']/12
    
    st.subheader("Salary(month) VS Purchased Plans")
    select_feature5 = st.radio('Select a feature:', ('Salary(month) VS Purchased Plan 1','Salary(month) VS Purchased Plan 2'))

    if select_feature5 == 'Salary(month) VS Purchased Plan 1':
    
        sns.boxplot(x='PURCHASEDPLAN1', y='SALARY(MONTH)', data = df)
        st.pyplot()

    elif select_feature5 == 'Salary(month) VS Purchased Plan 2':    
    
        sns.boxplot(x='PURCHASEDPLAN2', y='SALARY(MONTH)', data = df)
        st.pyplot()

    
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[10,20,30,40,50])
    st.subheader("Salary(month) vs Age Group")
    select_feature6 = st.radio('Select a feature:', ('Salary(month) VS AgeGroup (Purchase Plan 1)','Salary(month) VS AgeGroup (Purchase Plan 2)'))

    if select_feature6 == 'Salary(month) VS AgeGroup (Purchase Plan 1)':
    
        sns.boxplot(data=df, x="AGE_GROUP", y="SALARY(MONTH)", hue="PURCHASEDPLAN1")
        st.pyplot()

    elif select_feature6 == 'Salary(month) VS AgeGroup (Purchase Plan 2)':    
    
        sns.boxplot(data=df, x="AGE_GROUP", y="SALARY(MONTH)", hue="PURCHASEDPLAN2")
        st.pyplot()

    
    df2 = df.copy()

    df2["AGE"] = df2['AGE'].fillna(round(df2["AGE"].mean(),0))
    df2['NOOFDEPENDENT'] = df2['NOOFDEPENDENT'].fillna(round(df2["NOOFDEPENDENT"].mean(),0))
    df2["FAMILYEXPENSES(MONTH)"] = df2["FAMILYEXPENSES(MONTH)"].fillna(df2.groupby("RESIDENTIALTYPE")['FAMILYEXPENSES(MONTH)'].transform('mean'))
    df2["ANNUALSALARY"] = df2["ANNUALSALARY"].fillna(df2.groupby("RESIDENTIALTYPE")['ANNUALSALARY'].transform('mean'))
    df2["SALARY(MONTH)"] = df2["SALARY(MONTH)"].fillna(df2.groupby("RESIDENTIALTYPE")['SALARY(MONTH)'].transform('mean'))

    categorical_columns_mask = df2.dtypes==object
    categorical_columns = df2.columns[categorical_columns_mask].tolist()

    for col_name in categorical_columns:
        df2[col_name].fillna('NotSpecified', inplace=True)
    
    dfle = df2.copy()
    le = LabelEncoder()
    d = defaultdict(LabelEncoder)
    dfle[categorical_columns] = dfle[categorical_columns].apply(lambda x: d[x.name].fit_transform(x))

    min_max_scaler = MinMaxScaler()
    dfle[["FAMILYEXPENSES(MONTH)", "ANNUALSALARY", "SALARY(MONTH)"]] = min_max_scaler.fit_transform(dfle[["FAMILYEXPENSES(MONTH)", "ANNUALSALARY","SALARY(MONTH)"]])

    dfle = dfle.drop(["AGE_GROUP","SALARY(MONTH)"],axis=1)
    
    st.subheader("Dataframe after preprocessing")
    st.write(dfle)

    
    st.subheader("Handling Imbalanced Data")

    X = dfle.drop("PURCHASEDPLAN1", 1)
    y = dfle["PURCHASEDPLAN1"]
    features = X.columns
    os = SMOTE(sampling_strategy="not majority", k_neighbors=5,random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X,y.values.ravel(), test_size = 0.2, random_state=7)
    X,y = os.fit_resample(X_train,y_train)
    X = pd.DataFrame(data=X,columns=features)
    y = pd.DataFrame(data=y,columns=['PURCHASEDPLAN1'])
    st.subheader("Purchased Plan 1") 
    select_feature3 = st.radio('Select a feature:', ('Data before SMOTE (Purchased Plan 1)','Data after SMOTE (Purchased Plan 1)'))
      
    if select_feature3 == 'Data before SMOTE (Purchased Plan 1)':
        fig5 = plt.figure(figsize=(2,2))
        df["PURCHASEDPLAN1"].value_counts().plot(kind='bar')
        st.pyplot(fig5)
                
    elif select_feature3 == 'Data after SMOTE (Purchased Plan 1)':
        fig6 = plt.figure(figsize=(2,2))
        y["PURCHASEDPLAN1"].value_counts().plot(kind='bar')
        st.pyplot(fig6)
            
            
    X2 = dfle.drop("PURCHASEDPLAN2", 1)
    y2 = dfle["PURCHASEDPLAN2"]
    features = X2.columns
    os = SMOTE(sampling_strategy="not majority", k_neighbors=5, random_state=10)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2.values.ravel(), test_size = 0.1, random_state=7)
    X2,y2 = os.fit_resample(X_train2,y_train2)
    X2 = pd.DataFrame(data=X2,columns=features)
    y2 = pd.DataFrame(data=y2,columns=['PURCHASEDPLAN2'])
    st.subheader("Purchased Plan 2") 
    select_feature4 = st.radio('Select a feature:', ('Data before SMOTE (Purchased Plan 2)','Data after SMOTE (Purchased Plan 2)'))
    
    if select_feature4 == 'Data before SMOTE (Purchased Plan 2)':
        fig7 = plt.figure(figsize=(2,2))
        df["PURCHASEDPLAN2"].value_counts().plot(kind='bar')
        st.pyplot(fig7)
                
    elif select_feature4 == 'Data after SMOTE (Purchased Plan 2)':
        fig8 = plt.figure(figsize=(2,2))
        y2["PURCHASEDPLAN2"].value_counts().plot(kind='bar')
        st.pyplot(fig8)
    
    y = pd.Series(y['PURCHASEDPLAN1'].values)
    y2 = pd.Series(y2['PURCHASEDPLAN2'].values)
    ydf = y.to_frame()
    ydf.rename(columns={0: "PURCHASEDPLAN1"},inplace=True)
    XmergedDf = X.merge(ydf, left_index=True, right_index=True)
    XmergedDf[categorical_columns] = XmergedDf[categorical_columns].apply(lambda x: d[x.name].inverse_transform(x))
    ydf2 = y2.to_frame()
    ydf2.rename(columns={0: "PURCHASEDPLAN2"},inplace=True)
    XmergedDf2 = X2.merge(ydf2, left_index=True, right_index=True)
    XmergedDf2[categorical_columns] = XmergedDf2[categorical_columns].apply(lambda x: d[x.name].inverse_transform(x))
    
    categorical_columns_smote = XmergedDf[['GENDER', 'MARITALSTATUS', 
                       'LIFESTYLE', 'MOVINGTONEWCOMPANY']]
    numeric_columns = XmergedDf[["AGE","NOOFDEPENDENT","FAMILYEXPENSES(MONTH)","ANNUALSALARY"]]
    fig,axes = plt.subplots(2,2,figsize=(10,15))
    for idx,cat_col in enumerate(categorical_columns_smote):
        row,col = idx//2,idx%2
        sns.countplot(x=cat_col,data=XmergedDf,ax=axes[row,col])
    
    
    table3 = pd.crosstab(XmergedDf['CUSTOMER_NEEDS_1'], XmergedDf['PURCHASEDPLAN1'])
    table4 = pd.crosstab(XmergedDf['CUSTOMER_NEEDS_2'], XmergedDf['PURCHASEDPLAN2'])

    st.subheader("Customer Needs VS Purchased Plans after SMOTE")
    select_feature8 = st.radio('Select a feature:', ('Customer needs 1 vs Purchased Plan 1','Customer needs 2 vs Purchased Plan 2'))

    if select_feature8 == 'Customer needs 1 vs Purchased Plan 1':
    
        table3.plot(kind='barh', title='Counts')
        st.pyplot()

    elif select_feature8 == 'Customer needs 2 vs Purchased Plan 2':    
    
        table4.plot(kind='barh', title='Counts')
        st.pyplot()
        
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
elif section == "Feature Selection":
    
    st.header("Feature Selection")
    df = pd.read_csv('Insurance_Data.csv')

    df.columns = df.columns.str.upper()
    
    df['SALARY(MONTH)'] = df['ANNUALSALARY']/12
    
    df2 = df.copy()

    df2["AGE"] = df2['AGE'].fillna(round(df2["AGE"].mean(),0))
    df2['NOOFDEPENDENT'] = df2['NOOFDEPENDENT'].fillna(round(df2["NOOFDEPENDENT"].mean(),0))
    df2["FAMILYEXPENSES(MONTH)"] = df2["FAMILYEXPENSES(MONTH)"].fillna(df2.groupby("RESIDENTIALTYPE")['FAMILYEXPENSES(MONTH)'].transform('mean'))
    df2["ANNUALSALARY"] = df2["ANNUALSALARY"].fillna(df2.groupby("RESIDENTIALTYPE")['ANNUALSALARY'].transform('mean'))
    df2["SALARY(MONTH)"] = df2["SALARY(MONTH)"].fillna(df2.groupby("RESIDENTIALTYPE")['SALARY(MONTH)'].transform('mean'))

    categorical_columns_mask = df2.dtypes==object
    categorical_columns = df2.columns[categorical_columns_mask].tolist()

    for col_name in categorical_columns:
        df2[col_name].fillna('NotSpecified', inplace=True)
    
    dfle = df2.copy()
    le = LabelEncoder()
    d = defaultdict(LabelEncoder)
    dfle[categorical_columns] = dfle[categorical_columns].apply(lambda x: d[x.name].fit_transform(x))

    min_max_scaler = MinMaxScaler()
    dfle[["FAMILYEXPENSES(MONTH)", "ANNUALSALARY", "SALARY(MONTH)"]] = min_max_scaler.fit_transform(dfle[["FAMILYEXPENSES(MONTH)", "ANNUALSALARY","SALARY(MONTH)"]])

    dfle = dfle.drop(["SALARY(MONTH)"],axis=1)
    
    X = dfle.drop("PURCHASEDPLAN1", 1)
    y = dfle["PURCHASEDPLAN1"]
    features = X.columns
    os = SMOTE(sampling_strategy="not majority", k_neighbors=5,random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X,y.values.ravel(), test_size = 0.2, random_state=7)
    X,y = os.fit_resample(X_train,y_train)
    X = pd.DataFrame(data=X,columns=features)
    y = pd.DataFrame(data=y,columns=['PURCHASEDPLAN1'])
    
    X2 = dfle.drop("PURCHASEDPLAN2", 1)
    y2 = dfle["PURCHASEDPLAN2"]
    features = X2.columns
    os = SMOTE(sampling_strategy="not majority", k_neighbors=5, random_state=10)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2.values.ravel(), test_size = 0.1, random_state=7)
    X2,y2 = os.fit_resample(X_train2,y_train2)
    X2 = pd.DataFrame(data=X2,columns=features)
    y2 = pd.DataFrame(data=y2,columns=['PURCHASEDPLAN2'])
    
    def ranking(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x,2), ranks)
        return dict(zip(names, ranks))
    
    def boruta1(xa, ya):
        X = xa.copy()
        y = ya.copy()
        rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 5)
        feat_selector = BorutaPy(rf, n_estimators=100, random_state = 6)
        feat_selector.fit(X.values, y.values.ravel())
        colnames = X.columns
        boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
        boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
        boruta_score = boruta_score.sort_values("Score", ascending = False)
                
        b = st.radio('Display Top or Bottom',('Top','Bottom'))
        if b == 'Top':
            boruta_display = boruta_score.head(5)
        elif b == 'Bottom':
            boruta_display = boruta_score.tail(5)
        st.dataframe(boruta_display.astype('object'))
        st.subheader('**Top 23 Features**')
        sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind = "bar", 
                        aspect=1.5,palette='coolwarm')
        st.pyplot()
        
    def rf1(xa, ya):
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10, n_estimators=100, random_state=10)
        rf.fit(X, y)
        rfe = RFECV(rf, min_features_to_select=1, cv=3)
        rfe.fit(X,y)
        colnames2 = X.columns
        rfe_score = ranking(list(map(float, rfe.ranking_)), colnames2, order=-1)
        rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features','Score'])
        rfe_score = rfe_score.sort_values("Score",ascending = False)
        
        b = st.radio('Display Top or Bottom',('Top','Bottom'))
        if b == 'Top':
            rfe_display = rfe_score.head(5)
        elif b == 'Bottom':
            rfe_display = rfe_score.tail(5)
        st.dataframe(rfe_display.astype('object'))
        st.subheader('**Top 23 Features**')
        sns.catplot(x="Score", y="Features", data = rfe_score[0:30], kind = "bar", 
                        aspect=1.5,palette='coolwarm')
        st.pyplot()
        
    def boruta2(xa, ya):
        X = xa.copy()
        y = ya.copy()
        rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 5)
        feat_selector = BorutaPy(rf, n_estimators=100, random_state = 6)
        feat_selector.fit(X.values, y.values.ravel())
        colnames = X.columns
        boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
        boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
        boruta_score = boruta_score.sort_values("Score", ascending = False)
                
        b = st.radio('Display Top or Bottom',('Top','Bottom'))
        if b == 'Top':
            boruta_display = boruta_score.head(5)
        elif b == 'Bottom':
            boruta_display = boruta_score.tail(5)
        st.dataframe(boruta_display.astype('object'))
        st.subheader('**Top 23 Features**')
        sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind = "bar", 
                        aspect=1.5,palette='coolwarm')
        st.pyplot()
        
    def rf2(xa, ya):
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10, n_estimators=100, random_state=10)
        rf.fit(X, y)
        rfe = RFECV(rf, min_features_to_select=1, cv=3)
        rfe.fit(X,y)
        colnames2 = X.columns
        rfe_score = ranking(list(map(float, rfe.ranking_)), colnames2, order=-1)
        rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features','Score'])
        rfe_score = rfe_score.sort_values("Score",ascending = False)
        
        b = st.radio('Display Top or Bottom',('Top','Bottom'))
        if b == 'Top':
            rfe_display = rfe_score.head(5)
        elif b == 'Bottom':
            rfe_display = rfe_score.tail(5)
        st.dataframe(rfe_display.astype('object'))
        st.subheader('**Top 23 Features**')
        sns.catplot(x="Score", y="Features", data = rfe_score[0:30], kind = "bar", 
                        aspect=1.5,palette='coolwarm')
        st.pyplot()
    
    fs = st.selectbox("Choose a feature selection technique",("Boruta (Purchased Plan 1)","RF (Purchased Plan 1)","Boruta (Purchased Plan 2)","RF (Purchased Plan 2)"))
    
    if fs == "Boruta (Purchased Plan 1)":
        boruta1(X,y)
    
    elif fs == "RF (Purchased Plan 1)":
        rf1(X,y)
        
    elif fs == "Boruta (Purchased Plan 2)":
        boruta2(X2,y2)
        
    elif fs == "RF (Purchased Plan 2)":
        rf2(X2,y2)
    
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

elif section == "Classification":
    
    st.header("Classification")
    
    df = pd.read_csv('Insurance_Data.csv')

    df.columns = df.columns.str.upper()
    
    df['SALARY(MONTH)'] = df['ANNUALSALARY']/12
    
    df2 = df.copy()

    df2["AGE"] = df2['AGE'].fillna(round(df2["AGE"].mean(),0))
    df2['NOOFDEPENDENT'] = df2['NOOFDEPENDENT'].fillna(round(df2["NOOFDEPENDENT"].mean(),0))
    df2["FAMILYEXPENSES(MONTH)"] = df2["FAMILYEXPENSES(MONTH)"].fillna(df2.groupby("RESIDENTIALTYPE")['FAMILYEXPENSES(MONTH)'].transform('mean'))
    df2["ANNUALSALARY"] = df2["ANNUALSALARY"].fillna(df2.groupby("RESIDENTIALTYPE")['ANNUALSALARY'].transform('mean'))
    df2["SALARY(MONTH)"] = df2["SALARY(MONTH)"].fillna(df2.groupby("RESIDENTIALTYPE")['SALARY(MONTH)'].transform('mean'))

    categorical_columns_mask = df2.dtypes==object
    categorical_columns = df2.columns[categorical_columns_mask].tolist()

    for col_name in categorical_columns:
        df2[col_name].fillna('NotSpecified', inplace=True)
    
    dfle = df2.copy()
    le = LabelEncoder()
    d = defaultdict(LabelEncoder)
    dfle[categorical_columns] = dfle[categorical_columns].apply(lambda x: d[x.name].fit_transform(x))

    min_max_scaler = MinMaxScaler()
    dfle[["FAMILYEXPENSES(MONTH)", "ANNUALSALARY", "SALARY(MONTH)"]] = min_max_scaler.fit_transform(dfle[["FAMILYEXPENSES(MONTH)", "ANNUALSALARY","SALARY(MONTH)"]])

    dfle = dfle.drop(["SALARY(MONTH)"],axis=1)
    
    X = dfle.drop("PURCHASEDPLAN1", 1)
    y = dfle["PURCHASEDPLAN1"]
    features = X.columns
    os = SMOTE(sampling_strategy="not majority", k_neighbors=5,random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X,y.values.ravel(), test_size = 0.2, random_state=7)
    X,y = os.fit_resample(X_train,y_train)
    X = pd.DataFrame(data=X,columns=features)
    y = pd.DataFrame(data=y,columns=['PURCHASEDPLAN1'])
    
    X2 = dfle.drop("PURCHASEDPLAN2", 1)
    y2 = dfle["PURCHASEDPLAN2"]
    features = X2.columns
    os = SMOTE(sampling_strategy="not majority", k_neighbors=5, random_state=10)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2.values.ravel(), test_size = 0.1, random_state=7)
    X2,y2 = os.fit_resample(X_train2,y_train2)
    X2 = pd.DataFrame(data=X2,columns=features)
    y2 = pd.DataFrame(data=y2,columns=['PURCHASEDPLAN2'])
    
    X.drop(columns=['TRANSPORT', 'GENDER','MOVINGTONEWCOMPANY','HIGHESTEDUCATION'], inplace=True)
    X2.drop(columns=['TRANSPORT', 'MOVINGTONEWCOMPANY','MEDICALCOMPLICATION','MALAYSIAPR',"GENDER"], inplace=True)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2.values.ravel(), test_size = 0.2, random_state=7)
    
    
    stf = st.radio("Choose a target feature:", 
                              ("Purchased Plan 1", "Purchased Plan 2")
                              )
    if stf == "Purchased Plan 1":
        select_technique = st.selectbox("Choose a machine learning technique:", 
                              ("Random Forest Classifier", "Naive Bayes Classifier", "KNN Classifier", "Support Vector Classifier")
                              )
    
        if select_technique == "Random Forest Classifier":
            
            rf = RandomForestClassifier(random_state=16)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            prob_RF = rf.predict_proba(X_test)
            prob_RF = prob_RF
            auc_RF= roc_auc_score(y_test, prob_RF, average='weighted', multi_class='ovr')
            st.write("Accuracy on training set : {:.3f}".format(rf.score(X_train, y_train)))
            st.write("Accuracy on test set     : {:.3f}".format(rf.score(X_test, y_test)))
            st.write('AUC: %.2f' % auc_RF)
            confusion_majority=confusion_matrix(y_test, y_pred)
            st.write('Majority classifier Confusion Matrix\n', confusion_majority)
            target_names = ['Class 0', 'Class 1', 'Class 2']
            st.markdown(classification_report(y_test, y_pred, target_names=target_names))
            
        elif select_technique == "Naive Bayes Classifier":
            
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)
            prob_NB = nb.predict_proba(X_test)
            prob_NB = prob_NB
            auc_NB= roc_auc_score(y_test, prob_NB, average='weighted', multi_class='ovr')
            st.write("Accuracy on training set : {:.3f}".format(nb.score(X_train, y_train)))
            st.write("Accuracy on test set     : {:.3f}".format(nb.score(X_test, y_test)))
            st.write('AUC: %.2f' % auc_NB)
            confusion_majority=confusion_matrix(y_test, y_pred)
            st.write('Majority classifier Confusion Matrix\n', confusion_majority)
            target_names = ['Class 0', 'Class 1', 'Class 2']
            st.write(classification_report(y_test, y_pred, target_names=target_names))
            
        elif select_technique == "KNN Classifier":
            
            knn = KNeighborsClassifier(n_neighbors=8)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            prob_KNN = knn.predict_proba(X_test)
            auc_KNN = roc_auc_score(y_test, prob_KNN, multi_class="ovr")
            st.write("Accuracy on training set : {:.3f}".format(knn.score(X_train, y_train)))
            st.write("Accuracy on test set     : {:.3f}".format(knn.score(X_test, y_test)))
            st.write('AUC: %.2f' % auc_KNN)
            confusion_majority=confusion_matrix(y_test, y_pred)
            st.write('Majority classifier Confusion Matrix\n', confusion_majority)
            target_names = ['Class 0', 'Class 1', 'Class 2']
            st.write(classification_report(y_test, y_pred, target_names=target_names))

        elif select_technique == 'Support Vector Classifier':
            
            model = SVC(kernel = 'rbf', gamma = 'auto', probability=True)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            prob_SVM = model.predict_proba(X_test)
            auc_SVM = roc_auc_score(y_test, prob_SVM, multi_class="ovr")
            st.write("Accuracy on training set : {:.3f}".format(model.score(X_train, y_train)))
            st.write("Accuracy on test set     : {:.3f}".format(model.score(X_test, y_test)))
            st.write('AUC: %.2f' % auc_SVM)
            confusion_majority=confusion_matrix(y_test, y_pred)
            st.write('Majority classifier Confusion Matrix\n', confusion_majority)
            target_names = ['Class 0', 'Class 1', 'Class 2']
            st.write(classification_report(y_test, y_pred, target_names=target_names))
            
    elif stf == "Purchased Plan 2":
            
        select_technique2 = st.selectbox("Choose a machine learning technique:", 
                              ("Random Forest Classifier", "Naive Bayes Classifier", "KNN Classifier", "Support Vector Classifier")
                              )
            
        if select_technique2 == "Random Forest Classifier":
            
            rf = RandomForestClassifier(random_state=16)
            rf.fit(X_train2, y_train2)
            y_pred2 = rf.predict(X_test2)
            prob_RF2 = rf.predict_proba(X_test2)
            auc_RF2 = roc_auc_score(y_test2, prob_RF2, average='weighted', multi_class='ovr')
            st.write("Accuracy on training set : {:.3f}".format(rf.score(X_train2, y_train2)))
            st.write("Accuracy on test set     : {:.3f}".format(rf.score(X_test2, y_test2)))
            st.write('AUC: %.2f' % auc_RF2)
            confusion_majority2=confusion_matrix(y_test2, y_pred2)
            st.write('Majority classifier Confusion Matrix\n', confusion_majority2)
            target_names = ['Class 0', 'Class 1', 'Class 2']
            st.write(classification_report(y_test2, y_pred2, target_names=target_names))
                
        elif select_technique2 == "Naive Bayes Classifier":
            
            nb = GaussianNB()
            nb.fit(X_train2, y_train2)
            y_pred2 = nb.predict(X_test2)
            prob_NB2 = nb.predict_proba(X_test2)
            auc_NB2 = roc_auc_score(y_test2, prob_NB2, average='weighted', multi_class='ovr')
            st.write("Accuracy on training set : {:.3f}".format(nb.score(X_train2, y_train2)))
            st.write("Accuracy on test set     : {:.3f}".format(nb.score(X_test2, y_test2)))
            st.write('AUC: %.2f' % auc_NB2)
            confusion_majority2 = confusion_matrix(y_test2, y_pred2)
            st.write('Majority classifier Confusion Matrix\n', confusion_majority2)
            target_names = ['Class 0', 'Class 1', 'Class 2']
            st.write(classification_report(y_test2, y_pred2, target_names=target_names))
                
        elif select_technique2 == "KNN Classifier":
            
            knn = KNeighborsClassifier(n_neighbors=8)
            knn.fit(X_train2, y_train2)
            y_pred2 = knn.predict(X_test2)
            prob_KNN2 = knn.predict_proba(X_test2)
            auc_KNN2 = roc_auc_score(y_test2, prob_KNN2, multi_class="ovr")
            st.write("Accuracy on training set : {:.3f}".format(knn.score(X_train2, y_train2)))
            st.write("Accuracy on test set     : {:.3f}".format(knn.score(X_test2, y_test2)))
            st.write('AUC: %.2f' % auc_KNN2)
            confusion_majority2=confusion_matrix(y_test2, y_pred2)
            st.write('Majority classifier Confusion Matrix\n', confusion_majority2)
            target_names = ['Class 0', 'Class 1', 'Class 2']
            st.write(classification_report(y_test2, y_pred2, target_names=target_names))
        
        elif select_technique2 == 'Support Vector Classifier':
            
            model = SVC(kernel = 'rbf', gamma = 'auto', probability=True)
            model.fit(X_train2,y_train2)
            y_pred2 = model.predict(X_test2)
            prob_SVM2 = model.predict_proba(X_test2)
            auc_SVM2 = roc_auc_score(y_test2, prob_SVM2, multi_class="ovr")
            st.write("Accuracy on training set : {:.3f}".format(model.score(X_train2, y_train2)))
            st.write("Accuracy on test set     : {:.3f}".format(model.score(X_test2, y_test2)))
            st.write('AUC: %.2f' % auc_SVM2)
            confusion_majority2=confusion_matrix(y_test2, y_pred2)
            st.write('Majority classifier Confusion Matrix\n', confusion_majority2)
            target_names = ['Class 0', 'Class 1', 'Class 2']
            st.write(classification_report(y_test2, y_pred2, target_names=target_names))
        
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

elif section == "Clustering":
    
    st.header("Clustering")
    df = pd.read_csv('Insurance_Data.csv')

    df.columns = df.columns.str.upper()
    
    df['SALARY(MONTH)'] = df['ANNUALSALARY']/12
    
    df3 = df.dropna()
    
    y = df3['PURCHASEDPLAN1']
    y2 = df3['PURCHASEDPLAN2']
    X = df3.drop(['PURCHASEDPLAN1', 'PURCHASEDPLAN2'], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    
    st.subheader("Data before clustering")
    select_feature7 = st.radio('Select a feature:', ('Purchase Plan 1 and Purchased Plan 2','Customer Needs 1 and Customer needs 2'))
    
    if select_feature7 == 'Purchase Plan 1 and Purchased Plan 2':
        
        sns.scatterplot(x="SALARY(MONTH)", y="FAMILYEXPENSES(MONTH)", hue="PURCHASEDPLAN1", style="PURCHASEDPLAN2", data=df3)
        st.pyplot()
                
    elif select_feature7 == 'Customer Needs 1 and Customer needs 2':
        
        sns.scatterplot(x="SALARY(MONTH)", y="FAMILYEXPENSES(MONTH)", hue="CUSTOMER_NEEDS_1", style="CUSTOMER_NEEDS_2", data=df3)
        st.pyplot()
          
    km = KMeans(n_clusters=3, random_state=2)
    km.fit(X)
    df3_new = df3.copy()
    df3_new = df3_new.drop(["PURCHASEDPLAN1"],axis=1)
    df3_new['PURCHASEDPLAN1']=km.labels_
    
    st.subheader("Comparison between Data before and after clustering")
    select_feature9 = st.radio('Select a feature:', ('Salary(month) and FamilyExpenses(month) before clustering','Salary(month) and FamilyExpenses(month) after clustering'))
    
    if select_feature9 == 'Salary(month) and FamilyExpenses(month) before clustering':
        
        sns.scatterplot(x="SALARY(MONTH)", y="FAMILYEXPENSES(MONTH)", hue="PURCHASEDPLAN1", data=df3)
        st.pyplot()
                
    elif select_feature9 == 'Salary(month) and FamilyExpenses(month) after clustering':
        
        sns.scatterplot(x="SALARY(MONTH)", y="FAMILYEXPENSES(MONTH)", hue="PURCHASEDPLAN1", data=df3_new)
        st.pyplot()
        
        
    select_feature10 = st.radio('Select a feature:', ('Salary(month) and Age before clustering','Salary(month) and Age after clustering'))
    
    if select_feature10 == 'Salary(month) and Age before clustering':
        
        sns.scatterplot(x="AGE", y="SALARY(MONTH)", hue="PURCHASEDPLAN1", data=df3)
        st.pyplot()
                
    elif select_feature10 == 'Salary(month) and Age after clustering':
        
        sns.scatterplot(x="AGE", y="SALARY(MONTH)", hue="PURCHASEDPLAN1", data=df3_new)
        st.pyplot()
    
    st.subheader("Silhouette Score")
    print(silhouette_score(X, km.labels_))
    silhouette_visualizer(km,X,colors='yellowbrick')
    st.pyplot()
 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
elif section == "Association Rule Mining":
    
    st.header("Association Rule Mining")
    df = pd.read_csv('Insurance_Data.csv')

    df.columns = df.columns.str.upper()
    
    df['SALARY(MONTH)'] = df['ANNUALSALARY']/12
    
    df2 = df.copy()

    df2["AGE"] = df2['AGE'].fillna(round(df2["AGE"].mean(),0))
    df2['NOOFDEPENDENT'] = df2['NOOFDEPENDENT'].fillna(round(df2["NOOFDEPENDENT"].mean(),0))
    df2["FAMILYEXPENSES(MONTH)"] = df2["FAMILYEXPENSES(MONTH)"].fillna(df2.groupby("RESIDENTIALTYPE")['FAMILYEXPENSES(MONTH)'].transform('mean'))
    df2["ANNUALSALARY"] = df2["ANNUALSALARY"].fillna(df2.groupby("RESIDENTIALTYPE")['ANNUALSALARY'].transform('mean'))
    df2["SALARY(MONTH)"] = df2["SALARY(MONTH)"].fillna(df2.groupby("RESIDENTIALTYPE")['SALARY(MONTH)'].transform('mean'))

    categorical_columns_mask = df2.dtypes==object
    categorical_columns = df2.columns[categorical_columns_mask].tolist()

    for col_name in categorical_columns:
        df2[col_name].fillna('NotSpecified', inplace=True)
    
    select_feature11 = st.radio('Select a feature:', ('Purchased Plan1', 'Purchased Plan2'))
    
    if select_feature11 == 'Purchased Plan1':
    
        select_feature12 = st.radio('Select a feature:', ('View 1', 'View 2','View 3'))
        df_arm1 = df2[["CUSTOMER_NEEDS_1","PURCHASEDPLAN1"]]
        dfv = df_arm1.values
        te = TransactionEncoder()
        te_ary = te.fit(dfv).transform(dfv)
        df_arm = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_arm, min_support=0.05, use_colnames=True)
        association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)
        
        if select_feature12 == 'View 1':
            st.write(rules)
            
        elif select_feature12 == 'View 2':
            cnt = 0
            itemsets = []
            for index, rule in rules.iterrows():
                cnt += 1
                st.write("(Rule " + str(cnt) + ") " + list(rule['antecedents'])[0] + " -> " + list(rule['consequents'])[0])
                itemsets.append(list(rule['antecedents'])[0] + " -> " + list(rule['consequents'])[0])
                st.write("Support: " + str(round(rule['support'], 3)))
                st.write("Confidence: " + str(round(rule['confidence'], 3)))
                st.write("Lift: " + str(round(rule['lift'],3)))
                st.write("=====================================")
        
        elif select_feature12 == 'View 3':
            
            cnt = 0
            itemsets = []
            for index, rule in rules.iterrows():
                cnt += 1
                itemsets.append(list(rule['antecedents'])[0] + " -> " + list(rule['consequents'])[0])
            rules['itemsets'] = itemsets
            sns.scatterplot(x="lift", y="confidence", hue="itemsets", data=rules)
            st.pyplot()
    
    elif select_feature11 == 'Purchased Plan2':
        
        select_feature13 = st.radio('Select a feature:', ('View 1', 'View 2','View 3'))
        df_arm2 = df2[["CUSTOMER_NEEDS_2","PURCHASEDPLAN2"]]
        dfv2 = df_arm2.values
        te = TransactionEncoder()
        te_ary2 = te.fit(dfv2).transform(dfv2)
        df_arm2 = pd.DataFrame(te_ary2, columns=te.columns_)
        frequent_itemsets2 = apriori(df_arm2, min_support=0.05, use_colnames=True)
        #print (frequent_itemsets2)
        association_rules(frequent_itemsets2, metric="confidence", min_threshold=0.2)
        rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=0.2)
        rules2 = rules2.sort_values(['support', 'confidence'], ascending =[False, False]) 
        
        if select_feature13 == 'View 1':
                
            st.write(rules2)
                
        elif select_feature13 == 'View 2':
                
            cnt =0
            itemsets2 = []
            for index, rule in rules2.iterrows():
                cnt += 1
                st.write("(Rule " + str(cnt) + ") " + list(rule['antecedents'])[0] + " -> " + list(rule['consequents'])[0])
                itemsets2.append(list(rule['antecedents'])[0] + " -> " + list(rule['consequents'])[0])
                st.write("Support: " + str(round(rule['support'], 3)))
                st.write("Confidence: " + str(round(rule['confidence'], 3)))
                st.write("Lift: " + str(round(rule['lift'],3)))
                st.write("=====================================")
                
        elif select_feature13 == 'View 3':
            
            cnt =0
            itemsets2 = []
            for index, rule in rules2.iterrows():
                cnt += 1
                itemsets2.append(list(rule['antecedents'])[0] + " -> " + list(rule['consequents'])[0])

            
            rules2['itemsets'] = itemsets2
            sns.scatterplot(x="lift", y="confidence", hue="itemsets", data=rules2)
            st.pyplot()



