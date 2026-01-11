# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 08:35:34 2025

@author: user
"""

# ==========================================================
# Klasifikasi Kelulusan dengan SVM
# Input manual 5 data (Nilai, Kehadiran, Ekstrakurikuler 0–100)
# ==========================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# === 1. Baca dataset dari file Excel ===
file_path = r'c:\data_siswa.xlsx'
data = pd.read_excel(file_path)

# === 2. Pastikan kolom sesuai ===
if not {'nilai','Kehadiran','Ekstrakurikuler','Status'}.issubset(data.columns):
    raise ValueError("Kolom tidak lengkap! Harus ada: Nilai, Kehadiran, Ekstrakurikuler, Status")

# === 3. Pisahkan fitur dan label ===
X = data[['nilai','Kehadiran','Ekstrakurikuler']]
y = data['Status']

# === 4. Encode label target ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === 5. Split data (train/test) ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# === 6. Buat dan latih model SVM ===
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)

# === 7. Evaluasi model dasar ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
matrix = confusion_matrix(y_test, y_pred)

# === 8. Cetak hasil evaluasi ke kanvas ===
print("=== HASIL KLASIFIKASI DATA LATIH ===")
print("Akurasi:", round(acc*100,2),"%")
print("\nConfusion Matrix:\n", matrix)
print("\nClassification Report:\n", report)

# === 9. Input interaktif untuk 5 data manual ===
print("\n=== INPUT 5 DATA MANUAL UNTUK PREDIKSI ===")
data_manual = []
for i in range(1,6):
    print(f"\nData ke-{i}:")
    nilai = float(input("  nilai siswa (0–100): "))
    kehadiran = float(input("  Kehadiran (0–100): "))
    ekstra = float(input("  nilai Ekstrakurikuler (0–100): "))
    for var, val in [('nilai',nilai),('Kehadiran',kehadiran),('Ekstrakurikuler',ekstra)]:
        if not (0 <= val <= 100):
            raise ValueError(f"{var} harus antara 0–100")
    data_manual.append([nilai, kehadiran, ekstra])

# === 10. Buat DataFrame dari input manual ===
df_manual = pd.DataFrame(data_manual, columns=['nilai','Kehadiran','Ekstrakurikuler'])

# === 11. Prediksi semua data manual ===
prediksi = model.predict(df_manual)
hasil_prediksi = label_encoder.inverse_transform(prediksi)
df_manual['Prediksi'] = hasil_prediksi

# === 12. Cetak hasil prediksi ke kanvas ===
print("\n=== HASIL PREDIKSI 5 DATA MANUAL ===")
print(df_manual.to_string(index=False))

# === 13. Simpan laporan ke PDF ===
pdf_path = r'd:\hasil_klasifikasi_svm_interaktif_v4.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
styles = getSampleStyleSheet()
elements = []

elements.append(Paragraph("Laporan Klasifikasi SVM - Status Kelulusan", styles['Title']))
elements.append(Spacer(1, 12))
elements.append(Paragraph(f"Akurasi Model: {round(acc*100,2)}%", styles['Normal']))
elements.append(Spacer(1, 12))

# Confusion Matrix
elements.append(Paragraph("Confusion Matrix:", styles['Heading3']))
conf_table = Table([['', *label_encoder.classes_]] +
                   [[label_encoder.classes_[i]] + list(matrix[i]) for i in range(len(matrix))])
conf_table.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
    ('GRID',(0,0),(-1,-1),1,colors.black)
]))
elements.append(conf_table)
elements.append(Spacer(1, 12))

# Classification Report
elements.append(Paragraph("Classification Report:", styles['Heading3']))
for line in report.split('\n'):
    elements.append(Paragraph(line, styles['Code']))
elements.append(Spacer(1, 12))

# Hasil prediksi manual
elements.append(Paragraph("Hasil Prediksi 5 Data Manual:", styles['Heading3']))
tabel_manual = Table(
    [['No','nilai','Kehadiran','Ekstrakurikuler','Prediksi']] +
    [[i+1, row['nilai'], row['Kehadiran'], row['Ekstrakurikuler'], row['Prediksi']]
     for i,row in df_manual.iterrows()]
)
tabel_manual.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.lightblue),
    ('GRID',(0,0),(-1,-1),1,colors.black),
    ('FONTSIZE',(0,0),(-1,-1),9)
]))
elements.append(tabel_manual)
elements.append(Spacer(1, 12))

elements.append(Paragraph(
    "Catatan: Setiap nilai Ekstrakurikuler berada pada rentang 0–100 sebagai skor keaktifan siswa.",
    styles['Italic']
))

doc.build(elements)
print(f"\n📄 Laporan PDF berhasil disimpan di: {pdf_path}")
