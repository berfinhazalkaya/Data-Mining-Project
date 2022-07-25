#!/usr/bin/env python
# coding: utf-8

# ### EXPLANATION OF THE PROBLEM
# 
# Most of us know someone who is fighting breast cancer, or at least someone who is fighting breast cancer. Breast cancer is the most common cancer among women and affects about 2 million women each year. Breast cancer causes the most cancer-related deaths among women. It is estimated that 627,000 women died of breast cancer in 2018 alone.
# 
# Breast cancer is a disease that occurs as a result of a tumor caused by the change and uncontrolled proliferation of one of the cell groups that make up the breast tissue. Early diagnosis of breast cancer can greatly increase patients' chances of survival by encouraging early clinical treatment. This has become important for human health and predictions based on some information have become possible.

# INTRODUCING THE DATA SET
# 
# The data set studied was obtained through a data science community, Kaggle (https://www.kaggle.com/raghadalharbi/breast-cancer-gene-expression-profiles-metabric).
# 
# Our dataset has 693 attributes and 1904 rows.

# SOME SPECIFICITY INFORMATION IN THE DATA SET
# 
# • Patient ID -> Object -> Patient ID
# 
# • Age -> Float -> Age of the patient at the time of diagnosis
# 
# • Breast Surgery Type -> Object -> Breast cancer surgery type
# 
#       1-MASTECTOMY refers to a surgery performed to remove all breast tissue from the breast to treat or prevent breast cancer.
#     
#       2- Protection, which means an urgency where only the cancerous part of the breast is removed
#         
# • Cancer Type -> Object -> Breast cancer types:
#     
#                 1-Breast Cancer
#         
#                 2- Breast Sarcoma
#             
#             
# • Detailed Cancer Type -> Object -> Detailed Breast cancer types:
#     
# 1-Breast Invasive Ductal Carcinoma
# 
# 2-Breast Mixed Ductal and Lobular Carcinoma
# 
# 3-Breast Invasive Lobular Carcinoma
# 
# 4-Breast Invasive Mixed Mucinous Carcinoma
# 
# 5-Metaplastic Breast Cancer
# 
# Cellularity -> Object -> Cancer cellularity after chemotherapy refers to the amount of tumor cells in the sample and their arrangement into clusters.
# 
# Chemotherapy -> Object -> Whether the patient received chemotherapy as treatment (yes / no)
# 
# • Pam50 + Claudin Low Subtype -> Object -> Pam 50: whether some estrogen receptor positive (ER positive), HER2 negative breast cancers metastasize or not
#     
# It is a tumor profiling test that helps show (when breast cancer has spread to other organs).
# 
# The Claudin-low breast cancer subtype is most prominently defined by its gene expression characteristics:
#     
# Low expression of cell-cell adhesion genes, high expression of epithelial-mesenchymal transition (EMT) genes, and stem cell-like / less differentiated gene expression patterns
# 
# • Cohort -> Float -> Cohort is a group of subjects who share a descriptive trait (takes a value from 1 to 5)
# 
# • Er Status Measured by IHc -> Float -> Whether estrogen receptors are expressed in cancer cells immune histochemistry (a dye used in pathology targeting the specific antigen will color if available, otherwise it will not color) (positive negative)
# 
# • Er Status -> Object -> Cancer cells are positive or negative for estrogen receptors
# 
# • Neoplasm Histological Grade -> Int -> Determined by pathology by looking at the structure of the cells, whether it looks aggressive or not (takes a value from 1 to 3)
# 
# • Her2 Status Measured by Snap6 -> Object -> Using advanced molecular techniques to evaluate whether the cancer is positive for HER2 (Next generation sequencing type)
# • Death Caused by Cancer -> Object -> Whether the death of the patient is due to cancer or not

# In[6]:


# Main libraries we use(Kullandığımız temel kütüphaneler)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


get_ipython().system('pip install skompiler')


# In[8]:


#Loading data (Verilerin yüklenmesi)
data=pd.read_csv("METABRIC_RNA_Mutation.csv", low_memory=False)


# In[9]:


#To see the data (Verileri görmek için)
data.head()


# In[ ]:





# In[10]:


#Dimensions of data data (Data verisinin boyutları)
data.shape


# Burada da görüldüğü üzere öznitelik verisi fazladır bu yüzden bu verisetinin özenitelik verisini düşürdük.

# In[11]:


#we set the first 32 columns to remain (ilk 32 sütun kalacak şekilde ayarladık)

data=data.drop (data.columns [32:694], axis = 1)


# In[12]:


data.head()


# In[13]:


#After the deletion we tested it again. (Silme işleminden sonra tekrar test ettik.)
data.shape


# In[14]:


# We brought in the column we will use to find the death rate. (ölüm oranlarını bulmak için kullanacağımız sütunu getirdik.)
data=data.drop (data.columns [13:30], axis = 1)


# In[15]:


#test ettik ve sütunun geldiğini gördük
data.head()


# In[16]:


data.shape


# In[17]:


#We deleted this column because we won't be using the data in column 14. (14.sütundaki verileri kullanmayacağımız için bu sütunu sildik.)
data=data.drop (data.columns [14], axis = 1)


# In[18]:


data.shape


# In[19]:


data.head()


# In[20]:


#To find out the data type(Veri tipini öğrenmek için)
data.dtypes


# In[ ]:





# In[21]:


#to show how many types of different values are in it (içinde kaç tür farklı değer olduğunu göstermek için)
data.patient_id.unique()


# In[22]:


#içinde kaç tür farklı değer olduğunu göstermek için
data.er_status.unique()


# In[23]:


#shows the analysis of the data set. (veri setinin analizini gösterir.)
data.describe().T 


# In[24]:


#to see missing data (eksik verileri görmek için)
data.isna().sum()


# In[25]:


#We assign a value to the missing data and assign the data set to a file with the values we just assigned. (eksik verilere bir değer atadık ve yeni atadığımız değerler ile birlikte bir dosyaya atadık veri setini.)
data.fillna(value=-1).to_csv("yeni_data.csv")


# In[26]:


#veri seti kümesini gösterdik
data1=pd.read_csv("yeni_data.csv", low_memory=False)
data1.head()


# In[27]:


#An unnamed column has been added and we have removed it. (Unnamed(isimsiz) bir sütun eklendi onu çıkarttık.)

data1=data1.drop (data1.columns [0], axis = 1)


# In[28]:


data1.head()


# In[29]:


#test etmek amacıyla veri setinin boyutlarına bakıyoruz.
data1.shape


# In[30]:



data1.isna().sum()


# Değer atama işleminden sonra,eksik bir değer olup olmadığını kontrol  ettik ve yukarıdada artık eksik bir verinin olmadığını görüyoruz.

# In[31]:


#age_at_diagnosis attribute variable is float to convert it to int variable (age_at_diagnosis özniteliğin değişkeni float dır bunu int değişkenine çevirmek için)
data1['int_yaş']=data1['age_at_diagnosis'].apply(lambda x:int(x)) 
data1.head()


# In[32]:


#(belli bir hastanın özelliklerini getirmek için) to bring the characteristics of a particular patient
data1.loc[56,['age_at_diagnosis','type_of_breast_surgery','cancer_type']]


# In[33]:


#(bir özniteliği özelliklerine göre gruplandırma) grouping an attribute by its properties
data1.cellularity.value_counts()


# hücresellik
# yüksek olan 939
# orta olan 711
# düşük bölünme olan 200 tane hastamız vardır
# -1 değeri girilmemiş değerlerdir.

# In[34]:


#gruplandırmanın grafiksel olarak gösterimi(graphical representation of grouping)
sns.countplot(data1['cellularity'],label="Count")


# In[35]:


#bir özniteliği özelliklerine göre gruplandırma (grouping an attribute by its properties)
data.chemotherapy.value_counts()  


# 0=kemoterapi görmeyenler
# 1=kemoterapi görenler

# In[36]:


#gruplandırmanın grafiksel olarak gösterimi (graphical representation of grouping)
sns.countplot(data['chemotherapy'],label="Count")


# In[37]:


#bir özniteliği özelliklerine göre gruplandırma
data.death_from_cancer.value_counts()  


# Bu veriye dayanarak 801 kişinin yaşadığını,622 kişinin hastalık öldüğünü,480 kişinin de diğer nedenlerden ölmüştür.

# In[38]:


#gruplandırmanın grafiksel olarak gösterimi
sns.countplot(data1['death_from_cancer'],label="Count")


# In[39]:


#yaşlara göre kemoterapi alma oranları
data1.groupby(by='int_yaş').sum()['chemotherapy']


# In[40]:



data1.groupby(by='int_yaş').sum()['chemotherapy'].plot.barh();


# In[41]:


sns.pairplot(data1);


# In[42]:


data1.corr()


# In[43]:


#KORELASYON KISMI
sns.heatmap(data1.corr())


# Korelasyon analizi; değişkenler arasındaki ilişki, bu ilişkinin yönü ve şiddeti ile ilgili bilgiler sağlayan istatiksel bir yöntemdir.

# Özellik sayımız burada çok fazladır.Daha net bir sonuç görmek için özellik sayımızı azaltalım böylelikle sınıflandırmanın kalitesi de artmış oluyor

# In[44]:


#Burada age_at_diagnosis özniteliğine etki eden en yüksek 4 değeri aldık
data1.corr().nlargest(4,'age_at_diagnosis')


# In[45]:


#etki eden özniteliklerin sadece isimlerini öğrenmek istiyorsak sonuna index yazıp sergilemek için de tolist metodunu kullandım.
data1.corr().nlargest(4,'age_at_diagnosis').index.tolist()


# In[46]:


#şimdi görsel olarak görelim
sns.heatmap(data1.corr().nlargest(4,'age_at_diagnosis'))


# In[47]:


#yaş verisinin çizgi grafiğini çizmek için 
plt.plot(data1.int_yaş)  


# In[48]:


#0 ile 100 arasındaki verilerinin grafiği
plt.plot(data1.int_yaş[0:100])


# In[49]:


data1["death_from_cancer"].replace({'Living':'1',
                                  'Died of Disease':'2',
                                   'Died of Other Causes':'3'
                                    },inplace=True)


# In[50]:


data1["death_from_cancer"]=data1["death_from_cancer"].apply(lambda x:float(x)) 


# In[51]:


data1.info()


# In[52]:


data1.head()


# DOĞRUSAL REGRESYON MODELLERİ

# PCR

# Değişkenlere boyut indirgeme yapıldıktan sonra çıkan bileşenler üzerinde regresyon yapıyoruz.

# In[ ]:


#ilk başta kategorik değişkenleri dummies değişken formatına çevirmek


# In[53]:


dms=pd.get_dummies(data1[['type_of_breast_surgery','cancer_type','cancer_type_detailed','cellularity','pam50_+_claudin-low_subtype','er_status_measured_by_ihc','er_status','her2_status_measured_by_snp6']])


# In[54]:


dms.head()


# In[ ]:


#bağımlı değişkeni ve kategorik değişkenleri çıkarttım


# In[55]:


X_=data1.drop(['type_of_breast_surgery','cancer_type','cancer_type_detailed','cellularity','pam50_+_claudin-low_subtype','er_status_measured_by_ihc','er_status','her2_status_measured_by_snp6','death_from_cancer'],axis=1).astype("float64")


# In[56]:


X_.head()


# In[ ]:


#dummies değişkenler ile birlikte bir birleştirme işlemi yaptım.


# In[57]:


X=pd.concat([X_,dms[['type_of_breast_surgery_BREAST CONSERVING','type_of_breast_surgery_MASTECTOMY',
                    'cancer_type_Breast Cancer',
                     'cancer_type_Breast Sarcoma',
                     'cancer_type_detailed_Breast',
                     'cancer_type_detailed_Breast Invasive Ductal Carcinoma',
                     'cancer_type_detailed_Breast Invasive Lobular Carcinoma',
                     'cancer_type_detailed_Breast Invasive Mixed Mucinous Carcinoma',
                     'cancer_type_detailed_Breast Mixed Ductal and Lobular Carcinoma',
                     'cancer_type_detailed_Metaplastic Breast Cancer',
                     'pam50_+_claudin-low_subtype_Normal',
                     'pam50_+_claudin-low_subtype_claudin-low',
                     'er_status_measured_by_ihc_Negative',
                     'er_status_measured_by_ihc_Positve',
                     'er_status_Negative',
                     'er_status_Positive',
                     'her2_status_measured_by_snp6_GAIN',
                     'her2_status_measured_by_snp6_LOSS',
                     'her2_status_measured_by_snp6_NEUTRAL',
                     'her2_status_measured_by_snp6_UNDEF'
                     
                     
                     
                     
                     
                     
                     
                     
                    ]]],axis=1)


# In[58]:


X.head()  #bağımsız değişkenlerin


# In[59]:


y=data1[['death_from_cancer']]  #bağımlı değişkeni y değişkenine atadım.


# In[60]:


y.head() #bağımlı değişken


# In[61]:


from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=42)


# In[63]:


X_train.shape


# In[64]:


X_test.shape


# In[65]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
pca = PCA()


# In[ ]:


#pca nesnesini kurarak  değişken indirgeme yöntemini kullandım.


# In[66]:


X_reduced_train = pca.fit_transform(scale(X_train))


# In[ ]:


#1.satırın tüm değişkenlerini görmek istiyorum.


# In[67]:


X_reduced_train[0:1,:]


# In[68]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[69]:


pcr_model = lm.fit(X_reduced_train, y_train)  #modeli fit ettik
pcr_model.intercept_ #sabit


# In[70]:


pcr_model.coef_ #değişkenlerin katsayısı


# TAHMİN

# pcr_model ile tahmin işlemi yapıyoruz.

# In[71]:


y_pred = pcr_model.predict(X_reduced_train)


# tahmin edilen değerler

# In[72]:


y_pred[0:5]


# train setimde bulunan hatalarımızı hesapladık y_pred=eğitim setinde tahmin ettiğim degerler y_train=train setinde bulunan değerler

# In[73]:


from sklearn.metrics import mean_squared_error, r2_score
np.sqrt(mean_squared_error(y_train, y_pred)) #ortalama hatası 


# In[74]:


data1["death_from_cancer"].mean()


# In[75]:


r2_score(y_train, y_pred)


# In[76]:


pca2 = PCA() #testin hata oranını bulucaz


# In[77]:


X_reduced_test = pca2.fit_transform(scale(X_test))


# In[78]:


y_pred = pcr_model.predict(X_reduced_test)


# In[79]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[80]:


cv_hata = np.sqrt(-1*cross_val_score(pcr_model, 
                                     X_reduced_train, 
                                     y_train, 
                                     cv=10, 
                                     scoring = "neg_mean_squared_error").mean())


# In[81]:


cv_hata


# MODEL TUNİNG

# burada ele aldığım değişken sayısı değiştikçe hata oranı da değişir ben burada belirli değişkenleri almak istersem cross işlemi yapıp en iyi parametreyi bulmalıyım

# In[82]:


lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train[:,0:10], y_train)
y_pred = pcr_model.predict(X_reduced_test[:,0:10])
print(np.sqrt(mean_squared_error(y_test, y_pred)))


# In[83]:


from sklearn import model_selection


# In[84]:


cv_10 = model_selection.KFold(n_splits = 10,
                             shuffle = True,
                             random_state = 1)


# In[85]:


lm = LinearRegression()


# In[86]:


RMSE = []


# In[87]:


for i in np.arange(1, X_reduced_train.shape[1] + 1):
    
    score = np.sqrt(-1*model_selection.cross_val_score(lm, 
                                                       X_reduced_train[:,:i], 
                                                       y_train.ravel(), 
                                                       cv=cv_10, 
                                                       scoring='neg_mean_squared_error').mean())
    RMSE.append(score)


# In[88]:


plt.plot(RMSE, '-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('ÖLÜM ORANININ NEYDEN KAYNAKLANDIĞINI BULMAK için PCR Model Tuning');


# In[89]:


lm = LinearRegression()


# In[90]:


pcr_model = lm.fit(X_reduced_train[:,0:20], y_train)


# In[91]:


y_pred = pcr_model.predict(X_reduced_train[:,0:20])


# In[92]:


print(np.sqrt(mean_squared_error(y_train, y_pred)))


# In[93]:


y_pred = pcr_model.predict(X_reduced_test[:,0:20])


# In[ ]:





# In[ ]:





# In[94]:


print(np.sqrt(mean_squared_error(y_test, y_pred)))


# In[ ]:





# In[102]:


pip install astor


# In[103]:


from skompiler import skompile


# In[104]:


print(skompile(pcr_model.predict).to('python/code'))


# In[ ]:





# PLS

# In[97]:


from sklearn.cross_decomposition import PLSRegression, PLSSVD


# In[98]:


pls_model = PLSRegression().fit(X_train, y_train)


# In[99]:


pls_model.coef_


# TAHMİN

# In[100]:


X_train.head()


# In[101]:


pls_model.predict(X_train)[0:10]


# In[130]:


y_pred = pls_model.predict(X_train)


# In[131]:


np.sqrt(mean_squared_error(y_train, y_pred))


# In[132]:


r2_score(y_train, y_pred)


# In[133]:


y_pred = pls_model.predict(X_test)


# In[134]:


np.sqrt(mean_squared_error(y_test, y_pred))


# MODEL TUNİNG

# In[135]:


#CV
cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)


#Hata hesaplamak için döngü
RMSE = []

for i in np.arange(1, X_train.shape[1] + 1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls, X_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

#Sonuçların Görselleştirilmesi
plt.plot(np.arange(1, X_train.shape[1] + 1), np.array(RMSE), '-v', c = "r");
plt.xlabel('Bileşen Sayısı');
plt.ylabel('RMSE');
plt.title('DEATH OF CANCER');


# In[136]:


pls_model = PLSRegression(n_components = 2).fit(X_train, y_train)


# In[137]:


y_pred = pls_model.predict(X_test)


# In[138]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


get_ipython().system('pip install skompiler')


# In[ ]:





# In[ ]:


from skompiler import skompile


# In[ ]:


print(skompile(cart_model.predict).to('python/code'))


# PLS MODELİ DAHA AZ HATA VERMEKTEDİR.

# DOĞRUSAL OLMAYAN REGRESYON MODELLERİ

# KNN

# In[139]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor

from warnings import filterwarnings
filterwarnings('ignore')
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=42)


# In[140]:


knn_model = KNeighborsRegressor().fit(X_train, y_train)


# In[141]:


knn_model


# farklı k komşuluk sayılarına karşılık farklı değerler verecektir.buna karşılık olarak optimize edeceğimiz parametre de vardır.

# In[142]:


#HİPERPARAMETRE İLE ADLANDIRABİLECEĞİMİZ PARAMETRE KOMŞULUK SAYISIDIR.burada komşuluk sayısını 5 vermiştir.,
knn_model.n_neighbors 


# In[143]:


knn_model.effective_metric_


# TAHMİN

# In[144]:


y_pred = knn_model.predict(X_test)   #kurmuş olduğumuz knn modeli ile tahmin olayı.test hatasını hesaplayabiliriz.


# In[145]:


y_pred


# In[146]:


np.sqrt(mean_squared_error(y_test, y_pred))  #test hatasını bulduk.


# gözlem yapmak adına rmse değerlerini tutuyoruz her iterasyonda k ye göre modelimizi fit ediyoruz,tahmin yapıcaz ve tahmin işlemlerine bağlı olarak eğitim(train) üzerinde bir hata gözlemleme işlemi yapıcağız.

# In[147]:


RMSE = [] 

for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_train) 
    rmse = np.sqrt(mean_squared_error(y_train,y_pred)) 
    RMSE.append(rmse) 
    print("k =" , k , "için RMSE değeri: ", rmse)


# MODEL TUNİNG

# GridSearchCV üzerinde optimum k sayısını belirleme işlemi gerçekleştiriyoruz. GridSearchCv bir ızgara mantığıyla olası parametre setinin verilip tüm olası kombinasyonların denenmesi anlamına gelir

# In[148]:


from sklearn.model_selection import GridSearchCV  


# 1-30 a kadar komşuluk değerlerini deniyip hangisinin daha iyi olduğuna bakıcağız

# In[149]:


knn_params = {'n_neighbors': np.arange(1,30,1)} 


# In[150]:


knn = KNeighborsRegressor()


# burada modelin nesnesini tanımladık o yüzden fit etmemiz lazım.

# In[151]:


knn_cv_model = GridSearchCV(knn, knn_params, cv = 10)


# In[152]:


knn_cv_model.fit(X_train, y_train)


# In[153]:


knn_cv_model.best_params_["n_neighbors"]  #12 sayısını en iyi parametre değerini buldu.


# In[154]:


RMSE = [] 
RMSE_CV = []
for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_train) 
    rmse = np.sqrt(mean_squared_error(y_train,y_pred)) 
    rmse_cv = np.sqrt(-1*cross_val_score(knn_model, X_train, y_train, cv=10, 
                                         scoring = "neg_mean_squared_error").mean())
    RMSE.append(rmse) 
    RMSE_CV.append(rmse_cv)
    print("k =" , k , "için RMSE değeri: ", rmse, "RMSE_CV değeri: ", rmse_cv )


# 2.tarafta gördüğümüz değerler train hataları (valide edilmiştir)

# In[155]:


knn_tuned = KNeighborsRegressor(n_neighbors = knn_cv_model.best_params_["n_neighbors"]) #tune edilmiş modelin nesnesini oluşturduk.


# In[156]:


knn_tuned.fit(X_train, y_train) #fit ettik.


# In[157]:


np.sqrt(mean_squared_error(y_test, knn_tuned.predict(X_test)))


# XGBOOST

# In[158]:


get_ipython().system('pip install xgboost')


# In[159]:


import xgboost as xgb


# In[160]:


DM_train = xgb.DMatrix(data = X_train, label = y_train)
DM_test = xgb.DMatrix(data = X_test, label = y_test)


# In[161]:


from xgboost import XGBRegressor


# In[162]:


xgb_model = XGBRegressor().fit(X_train, y_train)


# TAHMİN

# In[163]:


y_pred = xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# MODEL TUNİNG

# In[164]:


xgb_model


# In[165]:


xgb_grid = {
     'colsample_bytree': [0.4, 0.5,0.6,0.9,1], 
     'n_estimators':[100, 200, 500, 1000],
     'max_depth': [2,3,4,5,6],
     'learning_rate': [0.1, 0.01, 0.5]
}


# In[166]:


xgb = XGBRegressor()

xgb_cv = GridSearchCV(xgb, 
                      param_grid = xgb_grid, 
                      cv = 10, 
                      n_jobs = -1,
                      verbose = 2)


xgb_cv.fit(X_train, y_train)


# In[167]:


xgb_cv.best_params_


# In[168]:


xgb_tuned = XGBRegressor(colsample_bytree = 0.4, 
                         learning_rate = 0.01, 
                         max_depth = 3, 
                         n_estimators = 500) 

xgb_tuned = xgb_tuned.fit(X_train,y_train)


# In[169]:


y_pred = xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# CART

# In[194]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=42)


# In[195]:


#BİR TANE DEĞİŞKEN SEÇİYORUM.

X_train = pd.DataFrame(X_train["int_yaş"])
X_test = pd.DataFrame(X_test["int_yaş"])


# In[216]:


cart_model = DecisionTreeRegressor(max_leaf_nodes=10,min_samples_split = 2)


# In[217]:


cart_model.fit(X_train, y_train)


# In[218]:


X_grid = np.arange(min(np.array(X_train)),max(np.array(X_train)), 0.01) #regresyon kural noktaları göstermek amacıyla grid oluşturuyor
X_grid = X_grid.reshape((len(X_grid), 1))  
plt.scatter(X_train, y_train, color = 'red')    #death_from_cancer  ve int_yaş değişkenleri için bir scatterplot oluşturuyoruz.
plt.plot(X_grid, cart_model.predict(X_grid), color = 'blue')  #tahmin edilen değerleri oluşturduğumuz grid yardımıyla grafiğe ekliyoruz.
plt.title('CART REGRESON AĞACI')  
plt.xlabel('Tanı yaşı(int_yaş)') 
plt.ylabel('death_of_cancer(ne sebepden öldü)') ;


# oluşturduğun grafiğin kural setini kurucam.Bunu skompile ile yapıcam

# In[199]:


get_ipython().system('pip install skompiler')


# In[222]:


pip install astor


# In[223]:


from skompiler import skompile


# In[224]:



print(skompile(cart_model.predict).to('python/code'))


# TAHMİN

# In[235]:


x = [45]  #şu an bir tahmin sonucunu oluşturduk.


# In[236]:


((((1.491891891891892 if x[0] <= 44.5 else 1.3563829787234043) if x[0] <= 
    51.5 else 1.5398230088495575) if x[0] <= 58.5 else 1.7380952380952381) if
    x[0] <= 62.5 else 1.9741379310344827 if x[0] <= 71.5 else (
    2.4166666666666665 if x[0] <= 73.5 else 2.096774193548387 if x[0] <= 
    75.5 else 2.484848484848485 if x[0] <= 76.5 else 2.2153846153846155) if
    x[0] <= 79.5 else 2.4623655913978495)


# In[237]:


cart_model.predict(X_test)[0:5]  #x testinin ilk 5 tahmin değeri


# In[241]:


cart_model.predict([[45]]) #yukarıdaki bulmuş olduğumuz terimin aynısıdır.


# In[242]:


y_pred =cart_model.predict(X_test)  #ilkel test hatamız.


# In[243]:


np.sqrt(mean_squared_error(y_test, y_pred))  #test hatamız 0.76 olarak çıkmıştır.


# MODEL TUNİNG 

# In[230]:


cart_model = DecisionTreeRegressor()
cart_model.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)


# In[244]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[245]:


cart_params = {"min_samples_split": range(2,100),
               "max_leaf_nodes": range(2,10)}


# In[246]:


cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 10)


# In[247]:


cart_cv_model.fit(X_train, y_train)


# In[248]:


cart_cv_model.best_params_


# In[249]:


cart_tuned = DecisionTreeRegressor(max_leaf_nodes = 3, min_samples_split = 2)


# In[250]:


cart_tuned.fit(X_train, y_train)


# In[251]:


y_pred = cart_tuned.predict(X_test)


# In[252]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[253]:


#final


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# XGBOOST

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# YAPAY SİNİR AĞLARI

# In[254]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)


# In[255]:


from sklearn.preprocessing import StandardScaler  


# In[256]:


scaler = StandardScaler()


# In[257]:


scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[258]:


X_test_scaled[0:5]


# In[259]:


from sklearn.neural_network import MLPClassifier


# In[260]:


mlpc = MLPClassifier().fit(X_train_scaled, y_train)


# In[261]:


y_pred = mlpc.predict(X_test_scaled)
accuracy_score(y_test, y_pred)


# In[ ]:


#MODEL TUNŞNG
mlpc


# In[264]:


mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.005, 0.0001,0.00001],
              "hidden_layer_sizes": [(10,10,10),
                                     (100,100,100),
                                     (100,100),
                                     (3,5), 
                                     (5, 3)],
              "solver" : ["lbfgs","adam","sgd"],
              "activation": ["relu","logistic"]}


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




