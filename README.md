# SVM-LEXICON
PENELTIAN PENGARUH POLA SENTIMEN NETRAL, POSITIF DAN NEGATIF TIAP KANDIDAT CAPRES 2024 TERHADAP POTENSI KEMENANGAN MENGGUNAKAN LEXICON BASED DAN PENDEKATAN MACHINE LEARNING(SVM)
# 📊 Analisis Sentimen Debat Pilpres 2024: SVM-Lexicon

## 🔍 Hasil Utama Penelitian
Dari total 
### 1. Distribusi Sentimen per Paslon
#### **Debat 1** (Persentase Sentimen)
Persebaran sentimen dibawah ini merupakan hasil setelah diimplementasikan Lexicon Based dan Support Vector Machine.
<img width="1344" height="830" alt="Image" src="https://github.com/user-attachments/assets/bbbe01b3-c6fa-481a-bfc0-721d47c90fbc" />

#### **Debat 2** (Persentase Sentimen)
<img width="1450" height="849" alt="Image" src="https://github.com/user-attachments/assets/797ca383-6ddc-4fb9-865e-900bf3a6b702" />

#### **Debat 3** (Persentase Sentimen)
<img width="1193" height="759" alt="Image" src="https://github.com/user-attachments/assets/db8baacc-5a8c-4027-a5c8-bd299a9a5250" />

#### **Debat 4** (Persentase Sentimen)
<img width="1072" height="735" alt="Image" src="https://github.com/user-attachments/assets/ab982789-a255-46e8-a77f-b48a11d227f1" />

#### **Debat 5** (Persentase Sentimen)
<img width="1177" height="753" alt="Image" src="https://github.com/user-attachments/assets/9afef0b2-de26-4efc-a13f-2b53289ff22b" />

### 2. Hasil Uji Korelasi Spearman
Skala untuk ukuran hubungan antar variabel dalam uji korelasi spearman pada penelitian ini menggunakan patokan dari SPSS yang umum digunakan, yakni :
1.	Nilai koefisien korelasi sebesar 0,00 – 0,25 = hubungan sangat lemah
2.	Nilai koefisien korelasi sebesar 0,26 – 0,50 = hubungan cukup
3.	Nilai koefisien korelasi sebesar 0,51 – 0,75 = hubungan kuat
4.	Nilai koefisien korelasi sebesar 0,76 – 0,99 = hubungan sangat kuat
5.	Nilai koefisien korelasi sebesar 1,00 = hubungan sempurna

Berikut data hasil perhitungan korelasi spearman berdasarkan data persebaran sentimen 
#### **Debat 1**
| Sentimen   | Korelasi | P-value | Interpretasi               |
|------------|----------|---------|----------------------------|
| Positive   | 0.500    | 0.667   | Korelasi sedang (tidak signifikan) |
| Negative   | 0.500    | 0.667   | Korelasi sedang (tidak signifikan) |
| Neutral    | 0.500    | 0.667   | Korelasi sedang (tidak signifikan) |
---
Dapat dilihat bahwa : 
Korelasi moderat (r_s = 0,500) untuk semua jenis sentimen mengindikasikan bahwa terdapat hubungan positif namun tidak sempurna antara sentimen pascadebat pertama dengan hasil pemilu. Nilai p-value yang tinggi (0,667) menunjukkan bahwa korelasi ini tidak signifikan secara statistik, yang berarti hubungan tersebut mungkin terjadi secara kebetulan.
#### **Debat 5**
| Sentimen   | Korelasi | P-value | Interpretasi               |
|------------|----------|---------|----------------------------|
| Positive   | 1.000    | 0.000   | Korelasi sempurna (sangat signifikan) |
| Negative   | 1.000    | 0.000   | Korelasi sempurna (sangat signifikan) |
| Neutral    | 1.000    | 0.000   | Korelasi sempurna (sangat signifikan) |

---
Korelasi sempurna (r_s = 1,000) dengan p-value signifikan (0,000) menunjukkan adanya hubungan kuat antara sentimen pascadebat kelima dengan hasil pemilu. Ini mengindikasikan bahwa urutan peringkat sentimen masyarakat (baik positif, negatif, maupun netral) setelah debat terakhir persis mencerminkan urutan peringkat perolehan suara final.


## 🧠 Implementasi Model

### Arsitektur SVM Hybrid
```python
Pipeline(
    ColumnTransformer([
        ('text', TfidfVectorizer(max_features=1000), 'stemming'),
        ('lex', StandardScaler(), ['InSet_Score', 'Sentistrength_Score'])
    ]),
    SVC(
        kernel='rbf',
        C=0.5,
        class_weight='balanced',
        gamma='scale'
    )
)
