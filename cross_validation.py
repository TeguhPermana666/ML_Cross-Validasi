import pandas as pd
data_path="Intermediate_ml\melb_data.csv"
data=pd.read_csv(data_path)
# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
#mager aing transfor ke numeric isi data categorical, dah lah langsung aja pakai numeric semua
X = data[cols_to_use]

# Select target
y = data.Price
"""
Kemudian, kami mendefinisikan pipeline yang menggunakan imputer untuk mengisi nilai 
yang hilang dan model hutan acak untuk membuat prediksi.Meskipun dimungkinkan untuk melakukan validasi silang tanpa saluran pipa,
itu cukup sulit! Menggunakan pipeline akan membuat kode menjadi sangat mudah.
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
my_pipeline=Pipeline(steps=[
    ('preprocesor',SimpleImputer()),
    ('model',RandomForestRegressor(n_estimators=50,random_state=0))
])
#Kita menetapkan cross_validation dengan menggunakan  cross_val_score() dari sckit-learn()
#Kami mengatur jumlah folds dengan parameter cv.
from sklearn.model_selection import cross_val_score
# Kalikan dengan -1 karena sklearn menghitung *negatif* MAE
scores=-1 * cross_val_score(my_pipeline, X,y,cv=10,scoring='neg_mean_absolute_error')
"""
semua data set dibagi kedalam setiap fold (#) yang dilakukan secara berulang sebanyak fold (*)
jadi setiap fold ,100%/10=>10%, berisikan 10% di setiap fold(*)
    #   #   #   #   #   #   #   #   #   #
*   v
*       v
*           v
*               v
*                   v
*                       v
*                           v
*                               v
*                                   v
*                                       v

dimana v adalah ->data validasi
jadi setiap eksperiment(*) dilakukan satu fold (#)untuk melakukan validasi
"""
print("MAE scores:\n", scores)
"""
Parameter penilaian memilih ukuran kualitas model untuk dilaporkan: 
dalam hal ini, kami memilih kesalahan absolut rata-rata negatif (MAE). Dokumen untuk scikit-learn menunjukkan daftar opsi.
Agak mengejutkan bahwa kami menentukan MAE negatif. Scikit-learn memiliki konvensi di mana semua metrik didefinisikan sehingga angka yang tinggi lebih baik. 
Menggunakan negatif di sini memungkinkan mereka untuk konsisten dengan konvensi itu, meskipun MAE negatif hampir tidak pernah terdengar di tempat lain.
Kami biasanya menginginkan ukuran kualitas model tunggal untuk membandingkan model alternatif. Jadi kami mengambil rata-rata di seluruh eksperimen.
"""
print("Rata-Rata MAE Scores:\n",scores.mean())
