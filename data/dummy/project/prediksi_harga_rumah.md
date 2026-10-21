# Prediksi Harga Rumah Jabodetabek

## Latar belakang
Project tugas akhir mata kuliah Machine Learning (semester 6). Tujuannya memperkirakan harga rumah berdasarkan luas tanah, luas bangunan, jumlah kamar, dan lokasi.

## Data
Dataset publik listing rumah Jabodetabek, sekitar 7.000 baris setelah dibersihkan. Data duplikat dan harga ekstrem (outlier) dihapus.

## Metode
Membandingkan regresi linear, random forest, dan gradient boosting dengan scikit-learn. Fitur lokasi di-encode dengan target encoding. Evaluasi memakai 5-fold cross-validation.

## Hasil
Random forest memberi hasil terbaik dengan RMSE sekitar 18% lebih rendah dibanding regresi linear. Fitur paling berpengaruh: luas bangunan dan kecamatan.

## Peran saya
Mengerjakan data cleaning, feature engineering, dan evaluasi model. Project dikerjakan berdua.

## Teknologi
Python, Pandas, scikit-learn, Matplotlib, Jupyter Notebook
