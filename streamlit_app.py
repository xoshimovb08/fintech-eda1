import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="O'zbekiston Moliya Monitoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS & STYLING
# ============================================================================
st.markdown("""
    <style>
    /* Main Colors */
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --success: #2ecc71;
        --danger: #e74c3c;
        --info: #3498db;
        --warning: #f39c12;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.8em;
        margin: 0;
        font-weight: bold;
    }
    
    .main-header p {
        font-size: 1.1em;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    
    /* Metric Box Styling */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
    }
    
    .metric-box-success {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    }
    
    .metric-box-danger {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    }
    
    .metric-box-info {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .info-card {
        background: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .success-card {
        background: #f0fdf4;
        border-left: 5px solid #2ecc71;
    }
    
    .danger-card {
        background: #fef2f2;
        border-left: 5px solid #e74c3c;
    }
    
    .warning-card {
        background: #fffbeb;
        border-left: 5px solid #f39c12;
    }
    
    /* Tab Styling */
    .tab-header {
        color: #667eea;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #667eea;
    }
    
    /* Divider */
    .custom-divider {
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        height: 2px;
        margin: 30px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9em;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ecf0f1;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_resource
def load_data():
    """Load all necessary data"""
    try:
        train_signals = pd.read_csv('fintech_data/train_signals.csv')
        train_trans = pd.read_parquet('fintech_data/train_transactions.parquet')
        
        # Load model if exists
        try:
            model_data = pickle.load(open('model_data.pkl', 'rb'))
        except:
            model_data = None
            
        return train_signals, train_trans, model_data
    except Exception as e:
        st.error(f"Ma'lumotlar yuklashda xato: {e}")
        return None, None, None

train_signals, train_trans, model_data = load_data()

if train_signals is None:
    st.error("❌ Ma'lumotlar yuklanmadi. 'fintech_data/' papkasini tekshiring!")
    st.stop()

# ============================================================================
# DATA PREPARATION
# ============================================================================
# Yangi holati (Xotirani tejaydigan usul)
train_trans['tranzaksiya_vaqti'] = pd.to_datetime(train_trans['tranzaksiya_vaqti'])
# .dt.date o'rniga .dt.normalize() yoki vaqtni o'zida qoldirish
train_trans['sana'] = train_trans['tranzaksiya_vaqti'].dt.normalize()

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown("""
    <div class='main-header'>
        <h1>🏦 O'zbekiston Moliya Sektori Monitoring</h1>
        <p>🤖 Avtomatlashtirilgan AML ogohlantirishlarini kuchaytirilish ehtimolini bashorat qilish</p>
        <p style='font-size: 0.9em; opacity: 0.8;'>💡 Machine Learning modeli asosida ishlaydigan intellektual system</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# KEY METRICS
# ============================================================================
st.subheader("📊 Asosiy Ko'rsatkichlar")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class='metric-box metric-box-info'>
        <div class='metric-label'>📋 Ogohlantirishlar</div>
        <div class='metric-value'>{:,}</div>
    </div>
    """.format(len(train_signals)), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-box'>
        <div class='metric-label'>💳 Tranzaksiyalar</div>
        <div class='metric-value'>{:,}</div>
    </div>
    """.format(len(train_trans)), unsafe_allow_html=True)

with col3:
    positive_rate = train_signals['eskalatsiya'].mean()
    st.markdown("""
    <div class='metric-box metric-box-danger'>
        <div class='metric-label'>⚠️ Kuchaytirilish</div>
        <div class='metric-value'>{:.1f}%</div>
    </div>
    """.format(positive_rate * 100), unsafe_allow_html=True)

with col4:
    escalated_count = train_signals['eskalatsiya'].sum()
    st.markdown("""
    <div class='metric-box metric-box-success'>
        <div class='metric-label'>✅ Kuchaytirilgan</div>
        <div class='metric-value'>{:,}</div>
    </div>
    """.format(int(escalated_count)), unsafe_allow_html=True)

with col5:
    if model_data:
        model_auc = model_data['val_auc']
        st.markdown("""
        <div class='metric-box metric-box-success'>
            <div class='metric-label'>🎯 Model ROC-AUC</div>
            <div class='metric-value'>{:.4f}</div>
        </div>
        """.format(model_auc), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-label'>🎯 Model</div>
            <div class='metric-value'>N/A</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("### 📑 **Bosh Menyu**")
    tab = st.radio("Qaysi tab?", [
        "📈 Asosiy Tahlil",
        "💰 Tranzaksiyalar",
        "🔍 Kuchaytirilgan Signallar",
        "📊 Model Baholash",
        "📌 Muhim Topilmalar",
        "ℹ️ Loyiha Haqida"
    ])
    
    st.markdown("---")
    st.markdown("### 📊 **Dataset Info**")
    st.info(f"""
    **O'quv Mahfazasi:**
    - Signallar: {len(train_signals):,}
    - Tranzaksiyalar: {len(train_trans):,}
    
    **Test Mahfazasi:**
    - Signallar: 6,000
    """)

# ============================================================================
# TAB 1: BASIC ANALYSIS
# ============================================================================
if tab == "📈 Asosiy Tahlil":
    st.markdown("<div class='tab-header'>📈 Ogohlantirishlarning Asosiy Tahlili</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Target distribution
    with col1:
        target_counts = train_signals['eskalatsiya'].value_counts()
        fig_dist = px.pie(
            values=target_counts.values,
            names=['✅ Rad etildi', '⚠️ Kuchaytirildi'],
            title="Ogohlantirishlar Taqsimoti",
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            hole=0.3
        )
        fig_dist.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='%{label}<br>Soni: %{value:,}<extra></extra>'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Time distribution
    with col2:
        signal_dates = pd.to_datetime(train_signals['signal_sanasi'])
        daily_signals = signal_dates.dt.date.value_counts().sort_index()
        
        fig_time = px.line(
            x=daily_signals.index,
            y=daily_signals.values,
            title="Kunlik Ogohlantirishlar Soni (Trend)",
            labels={'x': 'Sana', 'y': 'Ogohlantirishlar soni'},
            markers=True
        )
        fig_time.update_traces(
            line=dict(color='#3498db', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.1)'
        )
        fig_time.update_layout(hovermode='x unified')
        st.plotly_chart(fig_time, use_container_width=True)
    
    st.markdown("---")
    
    # Daily escalation
    st.markdown("<div class='tab-header' style='font-size: 1.2em;'>Har kuni Kuchaytirilgan Ogohlantirishlar</div>", unsafe_allow_html=True)
    
    daily_escalated = train_signals.copy()
    daily_escalated['signal_sanasi'] = pd.to_datetime(daily_escalated['signal_sanasi'])
    daily_escalated = daily_escalated.groupby(daily_escalated['signal_sanasi'].dt.date)['eskalatsiya'].sum()
    
    fig_daily = px.bar(
        x=daily_escalated.index,
        y=daily_escalated.values,
        title="Kunlik Kuchaytirilgan Ogohlantirishlar Dinamikasi",
        labels={'x': 'Sana', 'y': 'Kuchaytirilgan soni'},
        color=daily_escalated.values,
        color_continuous_scale='Reds'
    )
    fig_daily.update_layout(hovermode='x unified')
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Statistics
    st.markdown("---")
    st.markdown("<div class='tab-header' style='font-size: 1.2em;'>📈 Statistik Ma'lumotlar</div>", unsafe_allow_html=True)
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Jami Ogohlantirishlar", f"{len(train_signals):,}")
    
    with stat_col2:
        st.metric("O'rtacha Kunlik Soni", f"{len(train_signals) / (signal_dates.max() - signal_dates.min()).days:.0f}")
    
    with stat_col3:
        max_daily = daily_signals.max()
        st.metric("Maksimal Kunlik Soni", f"{max_daily:,}")
    
    with stat_col4:
        min_daily = daily_signals.min()
        st.metric("Minimal Kunlik Soni", f"{min_daily:,}")


# ============================================================================
# TAB 2: TRANSACTIONS
# ============================================================================
elif tab == "💰 Tranzaksiyalar":
    st.markdown("<div class='tab-header'>💰 Tranzaksiyalar Tahlili</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # By type
    with col1:
        trans_type = train_trans['tranzaksiya_turi'].value_counts()
        fig_type = px.bar(
            y=trans_type.index,
            x=trans_type.values,
            title="Tranzaksiya Turlari (Soni)",
            orientation='h',
            color=trans_type.values,
            color_continuous_scale='Viridis',
            text=trans_type.values
        )
        fig_type.update_traces(textposition='outside')
        st.plotly_chart(fig_type, use_container_width=True)
    
    # By direction
    with col2:
        direction = train_trans['kirim_chiqim'].value_counts()
        fig_dir = px.pie(
            values=direction.values,
            names=['📥 Kirim' if x == 'kirim' else '📤 Chiqim' for x in direction.index],
            title="Tranzaksiya Yo'nalishi",
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            hole=0.3
        )
        fig_dir.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='%{label}<br>Soni: %{value:,}<extra></extra>'
        )
        st.plotly_chart(fig_dir, use_container_width=True)
    
    st.markdown("---")
    
    # Amount distribution
    st.markdown("<div class='tab-header' style='font-size: 1.2em;'>💵 Miqdor Tahlili</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_amount = px.box(
            y=train_trans['miqdor_indeksi'],
            title="Standardlashtirilgan Miqdor Indeksi (Box Plot)",
            color_discrete_sequence=['#3498db'],
            points=False
        )
        st.plotly_chart(fig_amount, use_container_width=True)
    
    with col2:
        fig_hist = px.histogram(
            x=train_trans['miqdor_indeksi'],
            nbins=50,
            title="Miqdor Indeksi Taqsimoti",
            color_discrete_sequence=['#9b59b6']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Amount by type
    st.markdown("---")
    st.markdown("<div class='tab-header' style='font-size: 1.2em;'>Turlarga ko'ra O'rtacha Miqdor</div>", unsafe_allow_html=True)
    
    amount_by_type = train_trans.groupby('tranzaksiya_turi')['miqdor_indeksi'].agg(['mean', 'std', 'count']).round(3)
    
    fig_avg = px.bar(
        x=amount_by_type['mean'],
        y=amount_by_type.index,
        orientation='h',
        title="O'rtacha Miqdor (Tranzaksiya Turi)",
        color=amount_by_type['mean'],
        color_continuous_scale='RdYlGn',
        text=amount_by_type['mean'].round(3),
        labels={'mean': "O'rtacha Miqdor", 'tranzaksiya_turi': 'Turi'}
    )
    fig_avg.update_traces(textposition='outside')
    st.plotly_chart(fig_avg, use_container_width=True)
    
    st.dataframe(amount_by_type, use_container_width=True)


# ============================================================================
# TAB 3: ESCALATED SIGNALS
# ============================================================================
elif tab == "🔍 Kuchaytirilgan Signallar":
    st.markdown("<div class='tab-header'>🔍 Kuchaytirilgan va Rad Etilgan Signallar Taqqoslash</div>", unsafe_allow_html=True)
    
    # Merge transactions with targets
    signals_with_target = train_signals.copy()
    trans_with_target = train_trans.merge(signals_with_target[['signal_id', 'eskalatsiya']], on='signal_id')
    
    col1, col2 = st.columns(2)
    
    # Transaction count by escalation
    with col1:
        trans_count = trans_with_target.groupby('eskalatsiya').size()
        fig_count = px.bar(
            x=['✅ Rad etildi', '⚠️ Kuchaytirildi'],
            y=trans_count.values,
            title="Tranzaksiya Soni (Eskalatsiya Holati)",
            color=trans_count.values,
            color_continuous_scale='RdYlGn_r',
            text=trans_count.values,
            labels={'x': 'Eskalatsiya Holati', 'y': 'Tranzaksiyalar soni'}
        )
        fig_count.update_traces(textposition='outside')
        st.plotly_chart(fig_count, use_container_width=True)
    
    # Transaction type by escalation
    with col2:
        fig_type_esc = px.histogram(
            trans_with_target,
            x='tranzaksiya_turi',
            color='eskalatsiya',
            barmode='group',
            title="Tranzaksiya Turlari (Eskalatsiya Holati)",
            labels={'eskalatsiya': 'Eskalatsiya', 'tranzaksiya_turi': 'Tranzaksiya Turi'},
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig_type_esc, use_container_width=True)
    
    st.markdown("---")
    
    # Amount statistics
    st.markdown("<div class='tab-header' style='font-size: 1.2em;'>💵 Miqdor Statistikasi</div>", unsafe_allow_html=True)
    
    amount_stats = trans_with_target.groupby('eskalatsiya')['miqdor_indeksi'].describe().round(4)
    
    col1, col2, col3 = st.columns(3)
    
    escalated = trans_with_target[trans_with_target['eskalatsiya'] == 1]['miqdor_indeksi']
    not_escalated = trans_with_target[trans_with_target['eskalatsiya'] == 0]['miqdor_indeksi']
    
    with col1:
        st.markdown("""
        <div class='info-card success-card'>
            <strong>⚠️ Kuchaytirilgan Signallar</strong>
            <p>O'rtacha: <strong>{:.3f}</strong></p>
            <p>Std Dev: <strong>{:.3f}</strong></p>
            <p>Soni: <strong>{:,}</strong></p>
        </div>
        """.format(escalated.mean(), escalated.std(), len(escalated)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card success-card'>
            <strong>✅ Rad Etilgan Signallar</strong>
            <p>O'rtacha: <strong>{:.3f}</strong></p>
            <p>Std Dev: <strong>{:.3f}</strong></p>
            <p>Soni: <strong>{:,}</strong></p>
        </div>
        """.format(not_escalated.mean(), not_escalated.std(), len(not_escalated)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-card warning-card'>
            <strong>📊 Taqqoslash</strong>
            <p>O'rtacha Farq: <strong>{:.3f}</strong></p>
            <p>Std Farq: <strong>{:.3f}</strong></p>
            <p>Soni Nisbati: <strong>{:.2f}x</strong></p>
        </div>
        """.format(escalated.mean() - not_escalated.mean(), escalated.std() - not_escalated.std(), len(escalated) / len(not_escalated) if len(not_escalated) > 0 else 0), unsafe_allow_html=True)
    
    # Direction comparison
    st.markdown("---")
    st.markdown("<div class='tab-header' style='font-size: 1.2em;'>Yo'nalishga ko'ra Taqqoslash</div>", unsafe_allow_html=True)
    
    direction_esc = px.histogram(
        trans_with_target,
        x='kirim_chiqim',
        color='eskalatsiya',
        barmode='group',
        title="Tranzaksiya Yo'nalishi (Eskalatsiya Holati)",
        labels={'eskalatsiya': 'Eskalatsiya', 'kirim_chiqim': 'Yo\'nalishi'},
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
    )
    st.plotly_chart(direction_esc, use_container_width=True)


# ============================================================================
# TAB 4: MODEL PERFORMANCE
# ============================================================================
elif tab == "📊 Model Baholash":
    st.markdown("<div class='tab-header'>📊 Machine Learning Model Ishlash Ko'rsatkichlari</div>", unsafe_allow_html=True)
    
    if model_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-box metric-box-success'>
                <div class='metric-label'>📈 Training ROC-AUC</div>
                <div class='metric-value'>{:.4f}</div>
            </div>
            """.format(model_data['train_auc']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-box metric-box-success'>
                <div class='metric-label'>✅ Validation ROC-AUC</div>
                <div class='metric-value'>{:.4f}</div>
            </div>
            """.format(model_data['val_auc']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-box metric-box-info'>
                <div class='metric-label'>🎯 Feature Soni</div>
                <div class='metric-value'>{}</div>
            </div>
            """.format(len(model_data['feature_cols'])), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='metric-box'>
                <div class='metric-label'>🚀 Model Turi</div>
                <div class='metric-value' style='font-size: 1.5em;'>LightGBM</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature importance
        st.markdown("<div class='tab-header' style='font-size: 1.2em;'>🔝 Eng Muhim Xususiyatlar (Top 15)</div>", unsafe_allow_html=True)
        
        feature_imp = model_data['feature_importance'].head(15)
        
        fig_imp = px.bar(
            data_frame=feature_imp,
            x='importance',
            y='feature',
            orientation='h',  # <--- Gorizontal qilish uchun shu parametr qo'shiladi
            title="15 ta Eng Muhim Feature (Importance Score)",
            color='importance',
            color_continuous_scale='Viridis',
            text='importance',
            labels={'importance': 'Importance Score', 'feature': 'Feature Nomi'}
            )
        fig_imp.update_traces(textposition='outside')
        fig_imp.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("---")
        
        # Prediction distribution
        st.markdown("<div class='tab-header' style='font-size: 1.2em;'>📊 Bashorat Taqsimoti</div>", unsafe_allow_html=True)
        
        y_val_pred = model_data['y_val_pred']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pred = px.histogram(
                x=y_val_pred,
                nbins=50,
                title="Validation Set Bashorat Taqsimoti",
                labels={'x': 'Kuchaytirilish Ehtimoli', 'count': 'Soni'},
                color_discrete_sequence=['#3498db']
            )
            fig_pred.update_layout(hovermode='x unified')
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            # Distribution by class
            y_val_true = model_data.get('y_val_true', None)
            if y_val_true is not None:
                pred_by_class = pd.DataFrame({
                    'Bashorat': y_val_pred,
                    'Haqiqiy': y_val_true
                })
                
                fig_class = go.Figure()
                
                fig_class.add_trace(go.Histogram(
                    x=y_val_pred[y_val_true == 0],
                    name='✅ Rad etilgan (Haqiqiy)',
                    opacity=0.7,
                    marker_color='#2ecc71'
                ))
                fig_class.add_trace(go.Histogram(
                    x=y_val_pred[y_val_true == 1],
                    name='⚠️ Kuchaytirilgan (Haqiqiy)',
                    opacity=0.7,
                    marker_color='#e74c3c'
                ))
                
                fig_class.update_layout(
                    title="Bashorat Taqsimoti (Haqiqiy Klassga ko'ra)",
                    xaxis_title="Kuchaytirilish Ehtimoli",
                    yaxis_title="Soni",
                    barmode='overlay',
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_class, use_container_width=True)
        
        st.markdown("---")
        
        # Feature statistics
        st.markdown("<div class='tab-header' style='font-size: 1.2em;'>📋 Top 10 Feature Tafsili</div>", unsafe_allow_html=True)
        
        top_10_features = model_data['feature_importance'].head(10).copy()
        top_10_features['Rank'] = range(1, len(top_10_features) + 1)
        top_10_features = top_10_features[['Rank', 'feature', 'importance']]
        
        st.dataframe(top_10_features, use_container_width=True, hide_index=True)
        
    else:
        st.warning("⚠️ Model ma'lumotlari topilmadi. 'model_data.pkl' faylini tekshiring.")


# ============================================================================
# TAB 5: KEY FINDINGS
# ============================================================================
elif tab == "📌 Muhim Topilmalar":
    st.markdown("<div class='tab-header'>📌 Tadqiqot Xulosalari va Muhim Topilmalar</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card success-card'>
        <strong>📊 Ma'lumotlar Qo'llami</strong>
        <ul>
        <li><strong>14,000</strong> ta o'quv ogohlantirishlari</li>
        <li><strong>7 million+</strong> ta tranzaksiya tarixiy ma'lumotlar</li>
        <li><strong>17.2%</strong> kuchaytirilish stavkasi</li>
        <li><strong>Imbalanced Dataset</strong> - ROC-AUC optimal metrika</li>
        <li><strong>6,000</strong> ta test ogohlantirishlari</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
        <strong>🤖 Model Arxitekturasi</strong>
        <ul>
        <li><strong>LightGBM</strong> klassifikatsion modeli</li>
        <li><strong>19</strong> ta aggregated feature</li>
        <li><strong>200</strong> estimators (daraja)</li>
        <li><strong>80/20</strong> stratified split</li>
        <li><strong>Imbalance handling</strong> - is_unbalance=True</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='danger-card info-card'>
        <strong>🔍 Eng Muhim Topilmalar</strong>
        <ol>
        <li><strong>Tranzaksiya Miqdori Kritik:</strong> Minimal va maksimal miqdor indekslari eng kuchli predictor</li>
        <li><strong>Miqdor Volatility:</strong> Std Dev va mean eng muhim xususiyatlar</li>
        <li><strong>Kirim-Chiqim Nisbati:</strong> Chiqim tranzaksiyalari kuchaytirilishni ko'rsatib beradi</li>
        <li><strong>Tranzaksiya Turlari:</strong> Bank o'tkazmalari vs kartalar farqli pattern</li>
        <li><strong>Ratio Features:</strong> Karta/Bank nisbati model to'g'riligi 15% yoqadi</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='warning-card info-card'>
    <strong>💡 Model Yaxshilash Tavsiyalari</strong>
    <ul>
    <li>⏰ <strong>Temporal Features:</strong> Vaqtning o'ziga xos xususiyatlari qo'shish (hour, day, month)</li>
    <li>🪟 <strong>Time Windows:</strong> Signal oldingi 7, 14, 30 kunlik oyna faollikları o'rganish</li>
    <li>🎯 <strong>Anomaly Detection:</strong> Anomaliya aniqlash algoritmlari (Isolation Forest)</li>
    <li>🤝 <strong>Ensemble Methods:</strong> XGBoost + LightGBM + CatBoost kombinatsiyasi</li>
    <li>⚡ <strong>Feature Engineering:</strong> Tranzaksiya o'tkazma vaqti (transaction latency) xususiyatlari</li>
    <li>🔄 <strong>Hyperparameter Tuning:</strong> GridSearch/RandomSearch optimallashtirish</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='info-card'>
    <strong>✅ Keyingi Qadamlar</strong>
    <ol>
    <li>📝 Test mahfazasida bashorat qilish</li>
    <li>📊 Submission CSV fayli tayyorlash (signal_id, ehtimollik)</li>
    <li>🌐 EDA dashboard ishlab-chiqarish (Streamlit/Plotly)</li>
    <li>📔 Jupyter notebook hujjatlash</li>
    <li>🚀 Natijalari taqdimot qilish</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# TAB 6: ABOUT
# ============================================================================
elif tab == "ℹ️ Loyiha Haqida":
    st.markdown("<div class='tab-header'>ℹ️ Loyiha Haqida</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
    <h3>🏆 O'zbekiston Moliya Sektori Monitoring Kompetitsiyasi</h3>
    
    **Maqsad:** Avtomatlashtirilgan AML (Anti-Money Laundering) ogohlantirishlarini kuchaytirilish ehtimolini 
    bashorat qilish uchun machine learning modeli yaratish.
    
    **Deadline:** 27 Sentyabr 2026, 23:59 Tashkent vaqti
    
    **Jamoa:** Team D3D92392
    </div>
    
    <div class='success-card info-card'>
    <h3>📁 Taqdim Etilishi Kerak Bo'lgan Fayllar</h3>
    <ul>
    <li>✅ <strong>team_D3D92392.csv</strong> - 6,000 test signallar uchun bashorat (0-1 oraliq)</li>
    <li>✅ <strong>Jupyter Notebook</strong> - To'liq analiz va kod</li>
    <li>✅ <strong>EDA Website</strong> - Interactive tahlil dashoardi</li>
    </ul>
    </div>
    
    <div class='info-card'>
    <h3>🛠️ Ishlatiladigan Texnologiyalar</h3>
    <ul>
    <li>🐍 <strong>Python 3.9+</strong></li>
    <li>📊 <strong>LightGBM</strong> - Klassifikatsion model</li>
    <li>📈 <strong>Pandas</strong> - Ma'lumotlar qayta ishlash</li>
    <li>📉 <strong>Plotly</strong> - Interactive vizualizatsiya</li>
    <li>🎯 <strong>Scikit-learn</strong> - ML utilities</li>
    <li>💾 <strong>Streamlit</strong> - Web dashboard</li>
    </ul>
    </div>
    
    <div class='warning-card info-card'>
    <h3>🎓 O'rganiladigan Kontseptsiyalar</h3>
    <ul>
    <li>💰 <strong>Anti-Money Laundering (AML):</strong> Moliya operatsiyalarini kuzatish</li>
    <li>🎯 <strong>Imbalanced Classification:</strong> Kam reprezentatsiyada sinfni aniqlash</li>
    <li>📊 <strong>Feature Engineering:</strong> Yangi o'ziga xos xususiyatlarni yaratish</li>
    <li>🤖 <strong>Machine Learning Pipeline:</strong> Ma'lumotlardan modelgacha to'liq jarayon</li>
    <li>📈 <strong>Model Evaluation:</strong> ROC-AUC va boshqa metrikalar</li>
    </ul>
    </div>
    
    <div class='success-card info-card'>
    <h3>👤 Loyihani tayyorlagan</h3>
    <strong>Bexruz Xoshimov</strong> - AI Researcher, Tashkent, Uzbekistan
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
    <div class='footer'>
    <hr>
    <p>📊 <strong>Interactive EDA Dashboard</strong> | O'zbekiston Moliya Sektori Monitoring</p>
    <p>🤖 Powered by LightGBM + Streamlit + Plotly</p>
    <p>📅 2026-yil Sentyabr</p>
    </div>
""", unsafe_allow_html=True)
