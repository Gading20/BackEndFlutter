#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import pymongo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Koneksi ke MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017") # Ganti dengan URL MongoDB Anda
db = client['Quran'] # Ganti dengan nama database Anda
collection = db['audio'] # Ganti dengan nama koleksi Anda

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Aplikasi Analisis Data",  # Judul halaman
    page_icon=":chart_with_upwards_trend:",  # Ikon halaman
    layout="wide",  # Layout halaman
    initial_sidebar_state="expanded"  # Status awal sidebar
)

# Sidebar
option = st.sidebar.selectbox(
    'Menu Utama',
    ('Halaman Utama', 'Dataframe', 'Analisis Data')
)

# Fungsi untuk mengambil data dari MongoDB
def get_mongo_data():
    cursor = collection.find()
    df = pd.DataFrame(list(cursor)) # Ubah data dari cursor menjadi dataframe
    return df

# Halaman Utama
if option == 'Halaman Utama':
    st.title("Selamat Datang di Aplikasi Analisis Data")
    st.write("Aplikasi ini memungkinkan Anda untuk menganalisis dan memvisualisasikan data dengan Streamlit.")

# Halaman Dataframe
elif option == 'Dataframe':
    st.title("Visualisasi Dataframe")

    # Ambil satu dokumen dari MongoDB untuk mendapatkan nama kolom
    sample_doc = collection.find_one()
    if sample_doc:
        column_names = list(sample_doc.keys())  # Ambil nama-nama kolom dari dokumen sample
    else:
        st.error("Tidak ada data yang ditemukan di koleksi MongoDB.")

    # Ambil data dari MongoDB
    df = get_mongo_data()

    # Tampilkan jumlah data dalam bentuk grafik batang
    st.subheader("Jumlah Data")
    st.write("Grafik di bawah menunjukkan jumlah data unik dalam kolom-kolom tertentu:")
    st.bar_chart(df.nunique(), use_container_width=True)

    # Tampilkan tabel dengan jumlah data unik
    st.subheader("Tabel Jumlah Data")
    st.write("Tabel di bawah menampilkan jumlah data unik dalam kolom-kolom tertentu:")
    st.table(df.nunique())

    # Tampilkan pesan berdasarkan persentase
    st.subheader("Pesan")
    if df['_id'].nunique():
        st.error("Tampilan data yang telah melakukan deteksi di aplikasi Al-Quran kami")

# Halaman Analisis Data
elif option == 'Analisis Data':
    st.title('Aplikasi Analisis Data')
    st.write('Selamat datang di aplikasi analisis data sederhana!')

    # Ambil data dari MongoDB
    cursor = collection.find()
    df = pd.DataFrame(list(cursor)) # Ubah data dari cursor menjadi dataframe

    st.sidebar.header('Pengaturan')
    option = st.sidebar.selectbox(
        'Pilih sebuah opsi:',
        ('Tampilkan Data', 'Tampilkan Grafik')
    )

    if option == 'Tampilkan Data':
        st.subheader('Data')
        st.write(df)
    else:
        st.subheader('Grafik')
        # Grafik bar
        st.write('### Grafik Bar Nilai')
        fig, ax = plt.subplots()
        ax.bar(df['file_path'], df['predicted_class'])
        st.pyplot(fig)
