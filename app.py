import streamlit as st
import pandas as pd
import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


# Data dan konfigurasi
df = pd.read_excel("Coffee Shop Sales.xlsx")
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

st.set_page_config(page_title="Coffee Shop Sales Dashboard", layout="wide")

# Gaya CSS untuk tampilan yang lebih menarik
st.markdown("""
    <style>
    .block-container {padding: 1.5rem;}
    .title-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #8B4513;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #F5F5DC;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .divider {
        border-top: 3px solid #8B4513;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Gambar header
image = Image.open('coffe.jpg')
st.image(image, use_container_width=True, caption="Coffee Shop Dashboard")

# Judul utama
st.markdown('<div class="title-header">☕ Coffee Shop Interactive Sales Dashboard ☕</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📌 Filters")
locations = df['store_location'].unique()
selected_locations = st.sidebar.multiselect(
    "Select Store Location(s):",
    options=locations,
    default=locations
)

# Filter data berdasarkan lokasi
filtered_data = df[df['store_location'].isin(selected_locations)]

# Kolom untuk metrik utama
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("📈 Total Transactions", f"{filtered_data['transaction_qty'].sum():,}")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("💰 Total Revenue", f"${(filtered_data['Revenue'].sum()):,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("📦 Unique Products", filtered_data['product_type'].nunique())
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Visualisasi utama
col4, col5 = st.columns([0.6, 0.4])
with col4:
    st.subheader("📊 Transaction Trends")
    daily_qty = filtered_data.groupby('transaction_date')['transaction_qty'].sum().reset_index()
    fig_line = px.line(
        daily_qty,
        x='transaction_date',
        y='transaction_qty',
        title="Daily Transaction Quantity",
        markers=True,
        template="plotly_dark"
    )
    st.plotly_chart(fig_line, use_container_width=True)

with col5:
    st.subheader("🏪 Transaction Quantity by Store Location")
    location_qty = filtered_data.groupby('store_location')['transaction_qty'].sum().reset_index()
    fig_bar = px.bar(
        location_qty,
        x='store_location',
        y='transaction_qty',
        text='transaction_qty',
        title="Transaction Quantity by Store",
        color='transaction_qty',
        color_continuous_scale='Tealgrn',
        template="plotly_dark"
    )
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Visualisasi tambahan
col6, col7 = st.columns([0.5, 0.5])
with col6:
    st.subheader("📌 Transaction Quantity by Product Category")
    category_qty = filtered_data.groupby('product_type')['transaction_qty'].sum().reset_index()
    fig_pie = px.pie(
        category_qty,
        names='product_type',
        values='transaction_qty',
        title="Product Category Breakdown",
        color_discrete_sequence=px.colors.sequential.RdBu,
        template="plotly_dark"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col7:
    st.subheader("💲 Revenue by Store Location")
    revenue_by_location = filtered_data.groupby('store_location')['Revenue'].sum().reset_index()
    fig_revenue = px.bar(
        revenue_by_location,
        x='store_location',
        y='Revenue',
        text='Revenue',
        title="Total Revenue by Store Location",
        color='Revenue',
        color_continuous_scale='Blues',
        template="plotly_dark"
    )
    fig_revenue.update_traces(texttemplate='$%{text:,.2f}', textposition='outside')
    st.plotly_chart(fig_revenue, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

   
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

# Fungsi untuk mendeteksi outliers menggunakan IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Fungsi untuk menangani outliers dengan imputasi median
def handle_outliers_with_imputation(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Ganti outliers dengan median
    median = data[column].median()
    data.loc[data[column] < lower_bound, column] = median
    data.loc[data[column] > upper_bound, column] = median
    
    return data

# Fungsi untuk melakukan prediksi SARIMA
def sarima_forecast(data, order=(1,1,2), seasonal_order=(0,1,1,7), forecast_days=30):
    """ Melakukan prediksi menggunakan SARIMA dan mengembalikan hasilnya """
    try:
        sarima_model = SARIMAX(data['transaction_qty'], order=order, seasonal_order=seasonal_order)
        sarima_fit = sarima_model.fit(disp=False)

        # Prediksi untuk data historis
        data['SARIMA_pred'] = sarima_fit.predict(start=0, end=len(data)-1)

        # Prediksi untuk masa depan
        future_dates = pd.date_range(start=data['transaction_date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        future_preds = sarima_fit.predict(start=len(data), end=len(data) + forecast_days - 1)

        # Evaluasi model
        y_true = data['transaction_qty'].dropna()
        y_pred = data['SARIMA_pred'].dropna()
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return data, future_dates, future_preds, rmse, mae, mape
    except Exception as e:
        st.error(f"Error in SARIMA forecasting: {e}")
        return None, None, None, None, None, None

# SARIMA untuk tiap lokasi setelah penanganan outliers
sarima_results = []
for location in selected_locations:
    location_data = filtered_data[filtered_data['store_location'] == location].groupby('transaction_date')['transaction_qty'].sum().reset_index()
    
    # Handle outliers
    location_data = handle_outliers_with_imputation(location_data, 'transaction_qty')
    
    # SARIMA forecasting
    sarima_data, future_dates, future_preds, rmse, mae, mape = sarima_forecast(location_data)

    if sarima_data is not None:
        # Simpan hasil
        sarima_results.append({
            "location": location,
            "data": sarima_data,
            "future_dates": future_dates,
            "future_preds": future_preds,
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        })

# Visualisasi hasil SARIMA
st.markdown("## 📊 SARIMA Prediction")

for result in sarima_results:
    st.subheader(f"SARIMA Forecast for {result['location']}")
    
    fig_sarima = go.Figure()
    
    # Data aktual
    fig_sarima.add_trace(go.Scatter(
        x=result['data']['transaction_date'],
        y=result['data']['transaction_qty'],
        mode='lines',
        name="Actual Sales",
        line=dict(color="blue", width=2)
    ))

    # Prediksi historis
    fig_sarima.add_trace(go.Scatter(
        x=result['data']['transaction_date'],
        y=result['data']['SARIMA_pred'],
        mode='lines',
        name="SARIMA Prediction",
        line=dict(color="green", width=2, dash='dot')
    ))

    # Prediksi masa depan
    fig_sarima.add_trace(go.Scatter(
        x=result['future_dates'],
        y=result['future_preds'],
        mode='lines',
        name="Future Prediction",
        line=dict(color="red", width=2, dash='dash')
    ))

    fig_sarima.update_layout(
        title=f"Sales Forecast for {result['location']}",
        xaxis_title="Date",
        yaxis_title="Transaction Quantity",
        template="plotly_white",
        legend_title="Legend"
    )
    
    st.plotly_chart(fig_sarima, use_container_width=True)

    # Metrik evaluasi
    st.markdown(f"**Model Evaluation for {result['location']}:**")
    st.write(f"📌 RMSE: {result['rmse']:.2f}")
    st.write(f"📌 MAE: {result['mae']:.2f}")
    st.write(f"📌 MAPE: {result['mape']:.2f}%")
    st.markdown("---")

#DATA TABEL
st.markdown("### 📋 Filtered Data")
st.dataframe(filtered_data, use_container_width=True)

import streamlit as st
import pandas as pd
from collections import Counter

# Membaca data dari file
df = pd.read_excel("Coffee Shop Sales.xlsx")

# Preprocessing data
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Sidebar input untuk memilih product_type
selected_product_type = st.sidebar.selectbox("Pilih Kategori Produk:", df['product_type'].unique())

# Menampilkan rekomendasi produk detail berdasarkan product_type yang dipilih
st.markdown(f"### Rekomendasi Produk Detail untuk {selected_product_type}")

# Filter data berdasarkan product_type yang dipilih
filtered_data = df[df['product_type'] == selected_product_type]

# Menghitung total kuantitas untuk setiap product_detail
product_detail_qty = filtered_data.groupby('product_detail')['transaction_qty'].sum().reset_index()

# Mengurutkan produk detail berdasarkan kuantitas tertinggi
sorted_product_detail = product_detail_qty.sort_values(by='transaction_qty', ascending=False)

# Menampilkan 3 produk detail yang paling banyak terjual
st.subheader("🔝 3 Produk Teratas yang Paling Banyak Terjual")
top_3_sold = sorted_product_detail.head(3)
for index, row in top_3_sold.iterrows():
    st.write(f"📦 **{row['product_detail']}** - Kuantitas Terjual: {row['transaction_qty']}")

# Menampilkan 3 produk detail yang sedikit terjual
st.subheader("🔻 3 Produk Teratas yang Sedikit Terjual")
bottom_3_sold = sorted_product_detail.tail(3)
for index, row in bottom_3_sold.iterrows():
    st.write(f"📦 **{row['product_detail']}** - Kuantitas Terjual: {row['transaction_qty']}")


st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

# Fungsi untuk memuat dataset dengan caching
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('dataset_astoria_updated.xlsx')
        df['product_detail'] = df['product_detail'].str.strip()
        df.dropna(subset=['new_invoice_id'], inplace=True)
        df['new_invoice_id'] = df['new_invoice_id'].astype(str)
        df = df[~df['new_invoice_id'].str.contains('C')]
        return df
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Membentuk basket
    basket = (df.groupby(['new_invoice_id', 'product_detail'])['transaction_qty']
              .sum().unstack().reset_index().fillna(0)
              .set_index('new_invoice_id'))

    def encode_units(x):
        return 1 if x >= 1 else 0

    basket_sets = basket.applymap(encode_units)

    # Apriori Algorithm
    min_support = st.sidebar.slider("Minimum Support", 0.01, 0.1, 0.025, 0.005)
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(frozenset)

    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.7)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    st.title("Rekomendasi Menu Kopi ☕")
    if rules.empty:
        st.warning("Tidak ditemukan aturan asosiasi. Coba turunkan nilai minimum support atau threshold.")
    else:
        st.write("Aturan asosiasi ditemukan:")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                     .sort_values(by='lift', ascending=False))

    # Sistem rekomendasi interaktif
    selected_products = st.text_input("Masukkan produk yang dibeli (pisahkan dengan koma):").strip()
    if selected_products:
        selected_products = {p.strip() for p in selected_products.split(',')}
        recommendations = set()
        
        for _, row in rules.iterrows():
            antecedents = set(row['antecedents'].split(', '))
            consequents = set(row['consequents'].split(', '))
            if antecedents & selected_products:
                recommendations.update(consequents)
        
        recommendations -= selected_products  # Hapus produk yang sudah dibeli
        
        if recommendations:
            st.success(f"Rekomendasi produk: {', '.join(list(recommendations)[:3])}")
        else:
            st.warning("Tidak ada rekomendasi untuk produk yang dipilih.")




