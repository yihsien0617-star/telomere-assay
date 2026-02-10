import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 設定網頁標題
st.set_page_config(page_title="實驗室端粒分析平台", layout="wide", page_icon="🧬")
st.title("🧬 實驗室端粒趨勢分析平台")
st.markdown("---")

# --- 側邊欄：設定與資料載入 ---
st.sidebar.header("📂 資料與模型設定")
uploaded_file = st.sidebar.file_uploader("請上傳端粒數據 (CSV)", type=['csv'])

# 模型建立基準選擇
st.sidebar.subheader("🧮 端粒年齡推算模型")
model_basis = st.sidebar.radio(
    "選擇常模建立基準：",
    ("僅用首次檢測數據 (推薦)", "使用所有檢測數據"),
    help="推薦使用「首次檢測」作為基準，以排除後續干預措施對常模的影響。"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # 欄位檢查
        required_cols = ['姓名', '篩檢日期', '端粒長度', '備註', '年紀']
        if not all(col in df.columns for col in required_cols):
            st.error(f"資料格式錯誤！請確保 CSV 包含以下欄位：{required_cols}")
            st.stop()

        # === 資料清洗 ===
        df['備註'] = df['備註'].fillna("未分類").astype(str) # 修復 TypeError
        df['篩檢日期'] = pd.to_datetime(df['篩檢日期'], errors='coerce')
        df = df.dropna(subset=['篩檢日期', '姓名', '端粒長度', '年紀'])
        df = df.sort_values(by='篩檢日期')
        
        # === 核心運算：計算每人的「首末次變化量」 ===
        # 找出每個人最早與最晚的紀錄
        person_stats = []
        for name, group in df.groupby('姓名'):
            if len(group) >= 2:
                group = group.sort_values('篩檢日期')
                first_rec = group.iloc[0]
                last_rec = group.iloc[-1]
                
                change = last_rec['端粒長度'] - first_rec['端粒長度']
                duration = (last_rec['篩檢日期'] - first_rec['篩檢日期']).days
                
                person_stats.append({
                    '姓名': name,
                    '備註': first_rec['備註'], # 歸屬團隊以第一次為準
                    '首次數值': first_rec['端粒長度'],
                    '末次數值': last_rec['端粒長度'],
                    '變化量': change,
                    '監測天數': duration
                })
        
        df_changes = pd.DataFrame(person_stats)
        
        # === 建立端粒年齡常模 ===
        if model_basis == "僅用首次檢測數據 (推薦)":
            train_df = df.sort_values('篩檢日期').drop_duplicates(subset='姓名', keep='first')
        else:
            train_df = df

        valid_train = train_df[(train_df['年紀'] > 0) & (train_df['端粒長度'] > 0)]

        if len(valid_train) > 5:
            slope, intercept = np.polyfit(valid_train['年紀'], valid_train['端粒長度'], 1)
            def calculate_telomere_age(length):
                if slope == 0: return 0
                return (length - intercept) / slope
            
            # 應用到所有資料
            df['端粒年齡'] = df['端粒長度'].apply(calculate_telomere_age)
        else:
            st.warning("有效樣本數不足，無法建立模型。")
            slope, intercept = 0, 0
            df['端粒年齡'] = df['年紀']

    except Exception as e:
        st.error(f"資料處理失敗: {e}")
        st.stop()

    # --- 主要分析頁籤 ---
    tab1, tab2, tab3 = st.tabs(["📊 團隊成效總覽 (首末對比)", "👤 個人追蹤 & 端粒年齡", "📉 常模相關性"])

    # === Tab 1: 團隊成效 (首末對比) ===
    with tab1:
        st.header("團隊整體改善成效評估")
        st.markdown("此頁面僅統計 **「至少檢測過 2 次」** 的成員，計算其 `(最新數據 - 最初數據)` 的差異。")
        
        all_teams = sorted(list(df['備註'].unique()))
        sel_teams_ov = st.multiselect("選擇團隊", all_teams, default=all_teams)
        
        if not df_changes.empty and sel_teams_ov:
            df_ch_filtered = df_changes[df_changes['備註'].isin(sel_teams_ov)]
            
            col_kpi1, col_kpi2 = st.columns(2)
            with col_kpi1:
                avg_change = df_ch_filtered['變化量'].mean()
                st.metric("選定團隊平均變化量", f"{avg_change:+.3f}", 
                          help="正值代表端粒增長，負值代表縮短")
            with col_kpi2:
                # 計算進步比例
                improve_count = len(df_ch_filtered[df_ch_filtered['變化量'] > 0])
                total_count = len(df_ch_filtered)
                ratio = (improve_count / total_count * 100) if total_count > 0 else 0
                st.metric("成員進步比例", f"{ratio:.1f}% ({improve_count}/{total_count}人)")

            # 圖表 1: 團隊平均變化排行 (Bar Chart)
            team_agg = df_ch_filtered.groupby('備註')['變化量'].mean().reset_index().sort_values('變化量', ascending=False)
            fig_bar = px.bar(team_agg, x='備註', y='變化量', color='變化量',
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             title="各團隊「平均變化量」排行 (越高越好)",
                             text_auto='.3f')
            fig_bar.add_hline(y=0, line_color="black", line_width=1)
            st.plotly_chart(fig_bar, use_container_width=True)

            # 圖表 2: 個人變化量分佈 (Box Plot)
            st.subheader("團隊成員變化分佈細節")
            fig_box = px.box(df_ch_filtered, x='備註', y='變化量', color='備註', points="all",
                             hover_data=['姓名', '首次數值', '末次數值'],
                             title="團隊成員變化量分佈 (檢視個體差異)")
            fig_box.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_box, use_container_width=True)
            
        else:
            st.info("目前沒有足夠的「重複檢測」資料可供對比，或未選擇團隊。")

    # === Tab 2: 個人追蹤 (新增圖表) ===
    with tab2:
        st.subheader("個人詳細檢測報告")
        
        c1, c2 = st.columns(2)
        with c1:
            all_teams_list = sorted(list(df['備註'].unique()))
            sel_team = st.selectbox("1. 選擇團隊", all_teams_list, key="ind_team")
        with c2:
            people_list = sorted(df[df['備註'] == sel_team]['姓名'].unique())
            sel_person = st.selectbox("2. 選擇人員", people_list, key="ind_person")

        if sel_person:
            person_df = df[df['姓名'] == sel_person].sort_values('篩檢日期')
            latest = person_df.iloc[-1]
            
            # KPI
            real_age = latest['年紀']
            bio_age = latest['端粒年齡']
            diff = bio_age - real_age
            
            k1, k2, k3 = st.columns(3)
            k1.metric("實際年齡", f"{real_age:.0f} 歲")
            k2.metric("推算端粒年齡", f"{bio_age:.1f} 歲", 
                      delta=f"{diff:.1f} 歲 (差距)", delta_color="inverse")
            k3.metric("最新端粒長度", f"{latest['端粒長度']:.3f}")

            # --- 圖表區 (左右並排) ---
            g1, g2 = st.columns(2)
            
            with g1:
                # 圖表 A: 趨勢折線圖
                fig_line = px.line(person_df, x="篩檢日期", y="端粒長度", markers=True, 
                                title=f"📈 {sel_person} 端粒變化趨勢")
                # 同齡平均線
                expected_val = slope * real_age + intercept
                fig_line.add_hline(y=expected_val, line_dash="dash", line_color="gray", annotation_text="同齡平均")
                st.plotly_chart(fig_line, use_container_width=True)

            with g2:
                # 圖表 B: (新增) 各時間點柱狀圖
                # 將日期轉字串以免柱狀圖過細
                person_df['日期標籤'] = person_df['篩檢日期'].dt.strftime('%Y-%m-%d')
                fig_bar_ind = px.bar(person_df, x="日期標籤", y="端粒長度", 
                                     text="端粒長度", color="端粒長度",
                                     title=f"📊 {sel_person} 各時間點測試結果",
                                     color_continuous_scale="Viridis")
                fig_bar_ind.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                # 設定 Y 軸範圍讓變化看起來明顯一點 (optional)
                y_min = person_df['端粒長度'].min() * 0.8
                y_max = person_df['端粒長度'].max() * 1.1
                fig_bar_ind.update_layout(yaxis_range=[y_min, y_max])
                
                st.plotly_chart(fig_bar_ind, use_container_width=True)

            # 資料表
            st.dataframe(person_df[['篩檢日期', '端粒長度', '端粒年齡', '備註']].style.format({"端粒長度": "{:.3f}", "端粒年齡": "{:.1f}"}))

    # === Tab 3: 常模 ===
    with tab3:
        st.subheader("常模與相關性分析")
        sel_corr_teams = st.multiselect("篩選顯示團隊", all_teams, default=all_teams, key="corr_team")
        
        if sel_corr_teams:
            df_corr = df[df['備註'].isin(sel_corr_teams)]
            fig_scatter = px.scatter(df_corr, x="年紀", y="端粒長度", color="備註",
                                     hover_data=["姓名", "端粒年齡"],
                                     title=f"資料庫分佈 (模型基準: {model_basis})")
            
            # 畫常模線
            x_range = np.linspace(df['年紀'].min(), df['年紀'].max(), 100)
            y_pred = slope * x_range + intercept
            fig_scatter.add_traces(go.Scatter(x=x_range, y=y_pred, mode='lines', 
                                              name='常模趨勢線', line=dict(color='red', dash='dash')))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
else:
    st.info("👈 請從左側選單上傳 CSV 檔案")
