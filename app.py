import streamlit as st
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Coffee Shop Sales Dashboard", layout="wide")

# Load Data (hanya sekali)
@st.cache_data
def load_data():
    df = pd.read_excel("Coffee Shop Sales.xlsx")
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

df = load_data()

# Sidebar Navigation dengan Selectbox
menu = st.sidebar.selectbox("Menu:", ["📊 Dashboard", "🔮 Prediksi SARIMA", "📌 Rekomendasi Produk", "🤝 Sistem Rekomendasi Produk Kopi"], index=0)


# Mapping gambar latar belakang berdasarkan menu
background_images = {
    "📊 Dashboard": "background_dashboard.jpg",
    "🔮 Prediksi SARIMA": "background_sarima.jpg",
    "📌 Rekomendasi Produk": "background_rekomendasi.jpg",
    "🤝 Sistem Rekomendasi Produk Kopi": "background_asosiasi.jpg"
}

# Terapkan CSS untuk latar belakang
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("{background_images[menu]}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#######==================================DASHBOARD=======================================================

if menu == "📊 Dashboard":
    st.markdown(
    "<h1 style='text-align: center;'>☕ Coffee Shop Sales Dashboard ☕</h1>", 
    unsafe_allow_html=True
    )
    image = Image.open('coffe.jpg')
    st.image(image, use_container_width=True)

    # Filter lokasi
    # Pilihan Lokasi Toko
    locations = df['store_location'].unique()

    # Multiselect untuk memilih beberapa lokasi, secara default semua lokasi terpilih
    selected_locations = st.multiselect("Select Store Location(s):", options=locations, default=locations)

    # Filter Data Berdasarkan Lokasi yang Dipilih
    filtered_data = df[df['store_location'].isin(selected_locations)]

   
    
    # Gaya CSS untuk tampilan yang lebih menarik
    st.markdown("""<style>
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
        .small-subheader {
        font-size: 22px !important; /* Atur ukuran font di sini */
        font-weight: bold;
        color: #8B4513; /* Warna tetap bisa disesuaikan */
    }
    </style>""", unsafe_allow_html=True)

    # KPI Metrics
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
        st.markdown('<div class="small-subheader">📊 Transaction Trends</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="small-subheader">🏪 Transaction Quantity by Store Location</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="small-subheader">📌 Transaction Quantity by Product Category</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="small-subheader">💲 Revenue by Store Location</div>', unsafe_allow_html=True)
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

#####=====================PREDIKSI SARIMA=====================================================

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go

# Fungsi untuk menangani outliers dengan imputasi median
def handle_outliers_with_imputation(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data = data.copy()
    median = data[column].median()
    data.loc[data[column] < lower_bound, column] = median
    data.loc[data[column] > upper_bound, column] = median

    return data

# Fungsi untuk melakukan prediksi SARIMA
def sarima_forecast(data, order=(1,1,2), seasonal_order=(0,1,1,7), forecast_days=30):
    sarima_model = SARIMAX(data['transaction_qty'], order=order, seasonal_order=seasonal_order)
    sarima_fit = sarima_model.fit(disp=False)

    # Prediksi untuk data historis
    data['SARIMA_pred'] = sarima_fit.predict(start=0, end=len(data)-1)

    # Prediksi untuk masa depan
    future_dates = pd.date_range(start=data['transaction_date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    future_preds = sarima_fit.predict(start=len(data), end=len(data) + forecast_days - 1)

    # Evaluasi model
    rmse = np.sqrt(mean_squared_error(data['transaction_qty'], data['SARIMA_pred']))
    mae = mean_absolute_error(data['transaction_qty'], data['SARIMA_pred'])
    mape = np.mean(np.abs((data['transaction_qty'] - data['SARIMA_pred']) / data['transaction_qty'])) * 100

    return data, future_dates, future_preds, rmse, mae, mape

# Menu SARIMA
if 'menu' in locals() and menu == "🔮 Prediksi SARIMA":
    st.title("🔮 Prediksi SARIMA untuk Penjualan")

    # Pilihan Lokasi Toko
    locations = df['store_location'].unique()
    selected_location = st.selectbox("Pilih Lokasi Toko:", options=locations, index=0)

    # Filter data berdasarkan lokasi yang dipilih
    filtered_data = df[df['store_location'] == selected_location]

    if filtered_data.empty:
        st.warning("🚫 Tidak ada data yang tersedia untuk lokasi yang dipilih.")
    else:
        st.markdown("## 📊 Hasil Prediksi SARIMA")

        location_data = filtered_data.groupby('transaction_date')['transaction_qty'].sum().reset_index()

        # Penanganan outliers
        location_data = handle_outliers_with_imputation(location_data, 'transaction_qty')

        # SARIMA Forecasting
        try:
            sarima_data, future_dates, future_preds, rmse, mae, mape = sarima_forecast(location_data)
        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan saat melakukan prediksi: {e}")
            sarima_data = None

        if sarima_data is not None and not sarima_data.empty:
            st.subheader(f"📈 Prediksi SARIMA untuk {selected_location}")

            fig_sarima = go.Figure()

            # Data Aktual
            fig_sarima.add_trace(go.Scatter(
                x=sarima_data['transaction_date'],
                y=sarima_data['transaction_qty'],
                mode='lines',
                name="Penjualan Aktual",
                line=dict(color="blue", width=2)
            ))

            # Prediksi Historis
            fig_sarima.add_trace(go.Scatter(
                x=sarima_data['transaction_date'],
                y=sarima_data['SARIMA_pred'],
                mode='lines',
                name="Prediksi SARIMA",
                line=dict(color="green", width=2, dash='dot')
            ))

            # Prediksi Masa Depan
            fig_sarima.add_trace(go.Scatter(
                x=future_dates,
                y=future_preds,
                mode='lines',
                name="Prediksi Masa Depan",
                line=dict(color="red", width=2, dash='dash')
            ))

            fig_sarima.update_layout(
                title=f"📊 Prediksi Penjualan untuk {selected_location}",
                xaxis_title="🗕️ Tanggal",
                yaxis_title="📦 Jumlah Transaksi",
                template="plotly_white",
                legend_title="Legenda",
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )

            st.plotly_chart(fig_sarima, use_container_width=True)

            # Metrik Evaluasi
            st.markdown(f"### 📋 Evaluasi Model untuk {selected_location}:")
            st.metric(label="RMSE", value=f"{rmse:.2f}")
            st.metric(label="MAE", value=f"{mae:.2f}")
            st.metric(label="MAPE", value=f"{mape:.2f}%")
            st.markdown("---")
        else:
            st.warning("⚠️ Data prediksi tidak tersedia. Coba pilih lokasi lain.")


elif menu == "📌 Rekomendasi Produk":
    st.title("📌 Rekomendasi Produk")

    locations = df['store_location'].unique()
    selected_locations = st.multiselect("Select Store Location(s):", options=locations, default=locations)

    selected_product_type = st.selectbox("Pilih Kategori Produk:", df['product_type'].unique())
    filtered_data = df[df['product_type'] == selected_product_type]
    product_detail_qty = filtered_data.groupby('product_detail')['transaction_qty'].sum().reset_index()
    sorted_product_detail = product_detail_qty.sort_values(by='transaction_qty', ascending=False)

    st.subheader("🔝 Produk Terlaris")
    for _, row in sorted_product_detail.head(3).iterrows():
        st.write(f"📦 *{row['product_detail']}* - {row['transaction_qty']} terjual")
    
    st.subheader("🔻 Produk Kurang Laku")
    for _, row in sorted_product_detail.tail(3).iterrows():
        st.write(f"📦 *{row['product_detail']}* - {row['transaction_qty']} terjual")

elif menu == "🤝 Sistem Rekomendasi Produk Kopi":
    st.title("🤝 Analisis Asosiasi dengan Apriori")

    # ============================= PILIHAN DATASET =============================
    dataset_option = st.selectbox(
        "Pilih Dataset",
        ("Dataset Astoria", "Dataset Hell's Kitchen", "Dataset LowerManhattan")
    )

    @st.cache_data
    def load_assoc_data(dataset_option):
        if dataset_option == "Dataset Astoria":
            df = pd.read_excel('dataset_astoria_updated.xlsx')
        elif dataset_option == "Dataset Hell's Kitchen":
            df = pd.read_excel('dataset_hellskitchen.xlsx')
        else:
            df = pd.read_excel('dataset_lowermh.xlsx')

        df.dropna(subset=['new_invoice_id'], inplace=True)
        df['new_invoice_id'] = df['new_invoice_id'].astype(str)
        df = df[~df['new_invoice_id'].str.contains('C')]
        return df

    df_assoc = load_assoc_data(dataset_option)

    # ============================= MINIMUM SUPPORT BERBEDA UNTUK SETIAP DATASET =============================
    if dataset_option == "Dataset Astoria":
        min_support_default = 0.05
        st.info("⚠️ Untuk Dataset Astoria, disarankan minimum support antara 0.01 hingga 0.02.")
    elif dataset_option == "Dataset Hell's Kitchen":
        min_support_default = 0.03
        st.info("⚠️ Untuk Dataset Hell's Kitchen, disarankan minimum support antara 0.01 hingga 0.05.")
    else:
        min_support_default = 0.07
        st.info("⚠️ Untuk Dataset LowerManhattan, disarankan minimum support antara 0.02 hingga 0.05.")

    min_support = st.slider("🔧 Minimum Support", 0.01, 0.1, min_support_default, 0.005)

    # ============================= PROSES APRIORI =============================
    basket = df_assoc.groupby(['new_invoice_id', 'product_detail'])['transaction_qty'].sum().unstack().fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.7)

    # ============================= TAMPILKAN HASIL ASSOCIATION RULES =============================
    if rules.empty:
        st.warning("❌ Tidak ditemukan aturan asosiasi dengan support ini.")
    else:
        st.subheader("📌 Aturan Asosiasi Terbentuk")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        # ============================= SISTEM REKOMENDASI =============================
        st.subheader("🎯 Rekomendasi Produk")

        # Pastikan input pengguna dikonversi menjadi set
        selected_products = st.text_input("Masukkan produk yang dibeli (pisahkan dengan koma):")
        if selected_products:
            selected_products = set(p.strip() for p in selected_products.split(','))  # Pastikan ini adalah set

        recommendations = {}  # Dictionary untuk menyimpan skor rekomendasi

        import ast

        for _, row in rules.iterrows():
            # Konversi dari string ke set jika diperlukan
            antecedents = row['antecedents']
            if isinstance(antecedents, str):
                antecedents = set(ast.literal_eval(antecedents))
            else:
                antecedents = set(antecedents)

            consequents = set(map(str, row['consequents']))

            if selected_products & antecedents:
                for product in consequents:
                    recommendations[product] = recommendations.get(product, 0) + row['confidence']

            recommendations = {prod: score for prod, score in recommendations.items() if prod not in selected_products}


        # Urutkan rekomendasi berdasarkan confidence (descending) dan ambil 3 teratas
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:3]

        # Tampilkan hasil rekomendasi
        if top_recommendations:
            st.success("✅ Produk yang direkomendasikan untuk Anda:")
            for product, score in top_recommendations:
                st.write(f"🔹 **{product}** (Confidence: {score:.2f})")
        else:
            st.warning("⚠️ Tidak ada rekomendasi yang cocok dengan produk yang dipilih.")
