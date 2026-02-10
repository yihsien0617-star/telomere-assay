import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 設定網頁標題與排版
st.set_page_config(page_title="實驗室端粒分析平台", layout="wide", page_icon="🧬")

# 標題區
st.title("🧬 實驗室端粒趨勢分析平台")
st.markdown("---")

# --- 側邊欄：資料上傳 ---
st.sidebar.header("📂 資料載入")
uploaded_file = st.sidebar.file_uploader("請上傳端粒數據 (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # 欄位檢查
        required_cols = ['姓名', '篩檢日期', '端粒長度', '備註', '年紀']
        if not all(col in df.columns for col in required_cols):
            st.error(f"資料格式錯誤！請確保 CSV 包含以下欄位：{required_cols}")
            st.stop()

        # 資料清洗
        df['篩檢日期'] = pd.to_datetime(df['篩檢日期'], errors='coerce')
        df = df.dropna(subset=['篩檢日期', '姓名', '端粒長度', '年紀'])
        df = df.sort_values(by='篩檢日期')
        
        # === 核心演算法：計算端粒年齡 ===
        # 使用全體數據建立標準常模 (Linear Regression: Length = m * Age + c)
        # 排除異常值 (選擇性)
        valid_data = df[(df['年紀'] > 0) & (df['端粒長度'] > 0)]
        if len(valid_data) > 1:
            # 計算迴歸係數
            slope, intercept = np.polyfit(valid_data['年紀'], valid_data['端粒長度'], 1)
            
            # 定義推算函數： (Length - c) / m = Estimated Age
            def calculate_telomere_age(length):
                if slope == 0: return 0
                return (length - intercept) / slope

            # 應用到資料集
            df['端粒年齡'] = df['端粒長度'].apply(calculate_telomere_age)
        else:
            st.warning("數據不足，無法建立端粒年齡模型。")
            df['端粒年齡'] = df['年紀'] # Fallback
            slope, intercept = 0, 0

    except Exception as e:
        st.error(f"讀取檔案失敗: {e}")
        st.stop()

    # --- 主要分析頁籤 ---
    tab1, tab2, tab3 = st.tabs(["📊 團隊趨勢概覽", "👤 個人追蹤 & 端粒年齡", "📉 年齡相關性分析"])

    # === Tab 1: 團隊趨勢 ===
    with tab1:
        st.subheader("團隊整體表現")
        
        # 團隊篩選器
        all_teams = sorted(list(df['備註'].unique()))
        selected_teams_overview = st.multiselect("選擇要顯示的團隊", all_teams, default=all_teams)
        
        if selected_teams_overview:
            df_team = df[df['備註'].isin(selected_teams_overview)]
            
            # 依月份平均趨勢
            df_team['月份'] = df_team['篩檢日期'].dt.to_period('M').astype(str)
            team_trend = df_team.groupby(['備註', '月份'])['端粒長度'].mean().reset_index()
            
            fig_trend = px.line(team_trend, x="月份", y="端粒長度", color="備註", markers=True,
                                title="團隊平均端粒長度變化")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # 分佈圖
            fig_box = px.box(df_team, x="備註", y="端粒長度", color="備註", points="all", 
                             hover_data=["姓名", "年紀"], title="團隊數值分佈")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("請選擇至少一個團隊。")

    # === Tab 2: 個人追蹤 (新增：團隊篩選 & 端粒年齡) ===
    with tab2:
        st.subheader("個人詳細檢測報告")
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            # 1. 先選團隊
            all_teams_list = sorted(list(df['備註'].unique()))
            selected_team_individual = st.selectbox("1. 選擇所屬團隊", all_teams_list)
        
        with col_sel2:
            # 2. 再選該團隊的人員
            people_in_team = sorted(df[df['備註'] == selected_team_individual]['姓名'].unique())
            selected_person = st.selectbox("2. 選擇人員姓名", people_in_team)
            
        if selected_person:
            person_data = df[df['姓名'] == selected_person].sort_values('篩檢日期')
            latest_record = person_data.iloc[-1]
            
            # --- 顯示端粒年齡指標 ---
            st.markdown("#### 🧬 端粒年齡分析 (最新一次檢測)")
            
            # 邏輯：如果端粒年齡 < 實際年齡 = 年輕 (綠色)；反之 = 老化 (紅色)
            real_age = latest_record['年紀']
            bio_age = latest_record['端粒年齡']
            diff = bio_age - real_age 
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("實際年齡", f"{real_age:.0f} 歲")
            with m2:
                # 逆向思考：端粒越長 -> 端粒年齡越小 -> 狀態越好
                # diff 為負值代表生理比實際年輕
                color = "normal"
                if diff < -2: color = "normal" # 綠色/正常 (Streamlit metric 邏輯: delta正值為綠，這裡我們要反過來呈現語意，用文字輔助)
                
                st.metric("推算端粒年齡", f"{bio_age:.1f} 歲", 
                          delta=f"{diff:.1f} 歲 (與實際年齡差距)",
                          delta_color="inverse") # inverse: 數值變小(年輕)是好事(綠色)
            with m3:
                st.metric("最新端粒長度", f"{latest_record['端粒長度']:.3f}")

            # 繪圖
            fig_person = px.line(person_data, x="篩檢日期", y="端粒長度", markers=True, 
                                 title=f"{selected_person} 的歷史變化")
            
            # 畫出該年齡層的平均線 (基準線)
            # 計算該實際年齡對應的標準長度: Length = m * Age + c
            expected_len = slope * real_age + intercept
            fig_person.add_hline(y=expected_len, line_dash="dash", line_color="gray", 
                                 annotation_text=f"同齡平均水準 ({expected_len:.2f})")
            
            st.plotly_chart(fig_person, use_container_width=True)
            
            st.caption(f"註：端粒年齡是根據資料庫全體樣本 ({len(df)} 筆數據) 的迴歸曲線推算。")

    # === Tab 3: 年齡相關性 (新增：團隊篩選) ===
    with tab3:
        st.subheader("年齡 vs. 端粒長度迴歸分析")
        
        # 團隊篩選
        selected_teams_corr = st.multiselect("篩選顯示團隊", all_teams, default=all_teams, key="corr_team_select")
        
        if selected_teams_corr:
            df_corr = df[df['備註'].isin(selected_teams_corr)]
            
            fig_scatter = px.scatter(df_corr, x="年紀", y="端粒長度", color="備註", 
                                     hover_data=["姓名", "端粒年齡"], trendline="ols",
                                     title="年齡相關性分佈圖")
            
            # 顯示迴歸公式 (全體)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info(f"全體數據庫迴歸模型： 端粒長度 = {slope:.4f} × 年齡 + {intercept:.4f}")
            st.markdown("*(斜率為負值代表隨年齡增長，端粒長度呈現下降趨勢)*")
        else:
            st.warning("請選擇至少一個團隊以進行分析。")

else:
    st.info("👈 請從左側選單上傳 CSV 檔案")
