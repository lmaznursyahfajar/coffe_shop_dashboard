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


# Sidebar Navigation
menu = st.sidebar("Menu:", ["📊 Dashboard", "🔮 Prediksi SARIMA", "📌 Rekomendasi Produk", "🤝 Sistem Rekomendasi Produk Kopi"])
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
    st.title("☕ Coffee Shop Sales Dashboard ☕")
    image = Image.open('coffe.jpg')
    st.image(image, use_container_width=True)

    # Filter lokasi
    locations = df['store_location'].unique()
    selected_locations = st.sidebar.multiselect("Select Store Location(s):", options=locations, default=locations)
    filtered_data = df[df['store_location'].isin(selected_locations)]
    
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

#####=====================PREDIKSI SARIMA=====================================================

elif menu == "🔮 Prediksi SARIMA":
    st.title("🔮 Prediksi SARIMA untuk Penjualan")

    locations = df['store_location'].unique()
    selected_locations = st.sidebar.multiselect(
        "Pilih Lokasi Toko:",
        options=locations,
        default=locations
    )
    
    # Fungsi untuk menangani outliers dengan imputasi median
    def handle_outliers_with_imputation(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        median = data[column].median()
        data.loc[data[column] < lower_bound, column] = median
        data.loc[data[column] > upper_bound, column] = median
        
        return data

    # Fungsi untuk melakukan prediksi SARIMA
    def sarima_forecast(data, order=(1,1,2), seasonal_order=(0,1,1,7), forecast_days=30):
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
            st.error(f"Error dalam prediksi SARIMA: {e}")
            return None, None, None, None, None, None
    
    # Filter data berdasarkan lokasi yang dipilih
    filtered_data = df[df['store_location'].isin(selected_locations)]

    if filtered_data.empty:
        st.warning("Tidak ada data yang tersedia untuk lokasi yang dipilih.")
    else:
        st.markdown("## 📊 Hasil Prediksi SARIMA")
        
        for location in selected_locations:
            location_data = filtered_data[filtered_data['store_location'] == location].groupby('transaction_date')['transaction_qty'].sum().reset_index()
            
            # Handle outliers
            location_data = handle_outliers_with_imputation(location_data, 'transaction_qty')
            
            # SARIMA forecasting
            sarima_data, future_dates, future_preds, rmse, mae, mape = sarima_forecast(location_data)
            
            if sarima_data is not None:
                st.subheader(f"Prediksi SARIMA untuk {location}")
                
                fig_sarima = go.Figure()
                
                # Data aktual
                fig_sarima.add_trace(go.Scatter(
                    x=sarima_data['transaction_date'],
                    y=sarima_data['transaction_qty'],
                    mode='lines',
                    name="Penjualan Aktual",
                    line=dict(color="blue", width=2)
                ))

                # Prediksi historis
                fig_sarima.add_trace(go.Scatter(
                    x=sarima_data['transaction_date'],
                    y=sarima_data['SARIMA_pred'],
                    mode='lines',
                    name="Prediksi SARIMA",
                    line=dict(color="green", width=2, dash='dot')
                ))

                # Prediksi masa depan
                fig_sarima.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_preds,
                    mode='lines',
                    name="Prediksi Masa Depan",
                    line=dict(color="red", width=2, dash='dash')
                ))

                fig_sarima.update_layout(
                    title=f"Prediksi Penjualan untuk {location}",
                    xaxis_title="Tanggal",
                    yaxis_title="Jumlah Transaksi",
                    template="plotly_white",
                    legend_title="Legenda"
                )

                st.plotly_chart(fig_sarima, use_container_width=True)
                
                # Metrik evaluasi
                st.markdown(f"### Evaluasi Model untuk {location}:")
                st.write(f"📌 RMSE: {rmse:.2f}")
                st.write(f"📌 MAE: {mae:.2f}")
                st.write(f"📌 MAPE: {mape:.2f}%")
                st.markdown("---")

elif menu == "📌 Rekomendasi Produk":
    st.title("📌 Rekomendasi Produk")

    locations = df['store_location'].unique()
    selected_locations = st.sidebar.multiselect(
        "Pilih Lokasi Toko:",
        options=locations,
        default=locations
    )

    selected_product_type = st.sidebar.selectbox("Pilih Kategori Produk:", df['product_type'].unique())
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
        ("Dataset Astoria", "Dataset Hell's KItchen", "Dataset LowerManhattan")
    )

    @st.cache_data
    def load_assoc_data(dataset_option):
        if dataset_option == "Dataset Astoria":
            df = pd.read_excel('dataset_astoria_updated.xlsx')
        elif dataset_option == "dataset_hellskitchen":
            df = pd.read_excel('dataset_hellskitchen.xlsx')
        else:
            df = pd.read_excel('dataset_lowermh.xlsx')
        
        df.dropna(subset=['new_invoice_id'], inplace=True)
        df['new_invoice_id'] = df['new_invoice_id'].astype(str)
        df = df[~df['new_invoice_id'].str.contains('C')]
        return df

    df_assoc = load_assoc_data(dataset_option)

    # ============================= PROSES APRIORI =============================
    basket = df_assoc.groupby(['new_invoice_id', 'product_detail'])['transaction_qty'].sum().unstack().fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

    min_support = st.slider("🔧 Minimum Support", 0.01, 0.1, 0.05, 0.005)
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

    selected_products = st.text_input("Masukkan produk yang dibeli (pisahkan dengan koma):")
    if selected_products:
        selected_products = {p.strip() for p in selected_products.split(',')}
    recommendations = {}  # Gunakan dictionary untuk menyimpan skor

    for _, row in rules.iterrows():
        antecedents = set(row['antecedents'])
        consequents = set(row['consequents'])
        
        # Cek apakah produk yang dipilih ada di antecedents
        if antecedents & selected_products:
            for product in consequents:
                # Tambahkan skor confidence untuk setiap produk rekomendasi
                recommendations[product] = recommendations.get(product, 0) + row['confidence']

    # Hindari merekomendasikan produk yang sudah dibeli
    recommendations = {prod: score for prod, score in recommendations.items() if prod not in selected_products}

    # Urutkan rekomendasi berdasarkan skor confidence (descending) dan ambil 3 teratas
    top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:3]

    if top_recommendations:
        st.success("✅ Produk yang direkomendasikan untuk Anda:")
        for product, score in top_recommendations:
            st.write(f"🔹 **{product}** (Confidence: {score:.2f})")
    else:
        st.warning("⚠️ Tidak ada rekomendasi yang cocok dengan produk yang dipilih.")
