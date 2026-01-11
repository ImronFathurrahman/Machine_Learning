# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 18:56:38 2025

@author: user
"""

# ==============================================
# REGRESI LINEAR DENGAN 3 VARIABEL BEBAS
# Dosen: Saludin Muis
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------------------------------
# 1. Membaca dataset dari file Excel
# ------------------------------------------------
file_path = r'c:\UTS_ML.xlsx'  # Sesuaikan dengan lokasi Bapak
data = pd.read_excel(file_path)

# Asumsikan kolom: HeatFlux (Y), X1, X2, X3
Y = data.iloc[:, 0].values.reshape(-1, 1)
X = data.iloc[:, 1:4].values

# ------------------------------------------------
# 2. Melatih model regresi linear
# ------------------------------------------------
model = LinearRegression()
model.fit(X, Y)

# Prediksi dan evaluasi
Y_pred = model.predict(X)
r2 = r2_score(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)

# ------------------------------------------------
# 3. Menampilkan hasil di layar
# ------------------------------------------------
print("=== HASIL REGRESI LINEAR ===")
print("Koefisien (b1, b2, b3):", model.coef_[0])
print("Intercept (b0):", model.intercept_[0])
print("R²:", r2)
print("RMSE:", rmse)

# ------------------------------------------------
# 4. Menampilkan grafik di kanvas
# ------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(Y, label='Nilai Aktual', marker='o')
plt.plot(Y_pred, label='Nilai Prediksi', marker='x')
plt.title("Perbandingan Nilai Aktual vs Prediksi")
plt.xlabel("Indeks Data")
plt.ylabel("HeatFlux")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------
# 5. Menyimpan hasil ke file PDF
# ------------------------------------------------
pdf_path = r'd:\hasil_uts.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

judul = Paragraph("<b>Laporan Hasil Regresi Linear</b>", styles['Title'])
story.append(judul)
story.append(Spacer(1, 12))

isi = f"""
<b>Model:</b><br/>
HeatFlux = {model.intercept_[0]:.4f} 
+ ({model.coef_[0][0]:.4f})*X1 
+ ({model.coef_[0][1]:.4f})*X2 
+ ({model.coef_[0][2]:.4f})*X3<br/><br/>
<b>R²:</b> {r2:.4f}<br/>
<b>RMSE:</b> {rmse:.4f}<br/>
"""
story.append(Paragraph(isi, styles['Normal']))
story.append(Spacer(1, 12))

doc.build(story)
print(f"Hasil regresi berhasil disimpan ke: {pdf_path}")
