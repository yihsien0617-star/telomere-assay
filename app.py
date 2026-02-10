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

# --- 側邊欄：設定與資料載入 ---
st.sidebar.header("📂 資料與模型設定")
uploaded_file = st.sidebar.file_uploader("請上傳端粒數據 (CSV)", type=['csv'])

# 新增：模型建立基準選擇
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

        # === 1. 資料清洗與格式修復 (修復 TypeError) ===
        # 強制將備註轉為字串，並填補空白值
        df['備註'] = df['備註'].fillna("未分類").astype(str)
        
        # 日期處理
        df['篩檢日期'] = pd.to_datetime(df['篩檢日期'], errors='coerce')
        
        # 移除無效資料
        df = df.dropna(subset=['篩檢日期', '姓名', '端粒長度', '年紀'])
        df = df.sort_values(by='篩檢日期')
        
        # === 2. 建立端粒年齡常模 (Linear Regression) ===
        # 根據使用者選擇篩選訓練資料
        if model_basis == "僅用首次檢測數據 (推薦)":
            # 針對每個人，只保留日期最早的那一筆
            train_df = df.sort_values('篩檢日期').drop_duplicates(subset='姓名', keep='first')
            st.sidebar.success(f"已選用 {len(train_df)} 筆「首次檢測」資料建立模型")
        else:
            train_df = df
            st.sidebar.info(f"使用全體 {len(train_df)} 筆資料建立模型")

        # 排除異常值 (例如年齡或長度 <= 0)
        valid_train = train_df[(train_df['年紀'] > 0) & (train_df['端粒長度'] > 0)]

        if len(valid_train) > 5:
            # 計算迴歸係數: Length = slope * Age + intercept
            slope, intercept = np.polyfit(valid_train['年紀'], valid_train['端粒長度'], 1)
            
            # 定義推算函數: Age = (Length - intercept) / slope
            def calculate_telomere_age(length):
                if slope == 0: return 0
                return (length - intercept) / slope

            # 應用到「所有」資料 (即使不是首次測量，也用這個公式算)
            df['端粒年齡'] = df['端粒長度'].apply(calculate_telomere_age)
            
            # 顯示模型公式
            st.sidebar.markdown(f"**模型公式：**\n`y = {slope:.4f}x + {intercept:.2f}`")
        else:
            st.warning("有效樣本數不足，無法建立端粒年齡模型。")
            slope, intercept = 0, 0
            df['端粒年齡'] = df['年紀']

    except Exception as e:
        st.error(f"資料處理失敗: {e}")
        st.stop()

    # --- 主要分析頁籤 ---
    tab1, tab2, tab3 = st.tabs(["📊 團隊趨勢概覽", "👤 個人追蹤 & 端粒年齡", "📉 常模與相關性分析"])

    # === Tab 1: 團隊趨勢 ===
    with tab1:
        st.subheader("團隊整體表現")
        
        all_teams = sorted(list(df['備註'].unique())) # 這裡已經修復了
        selected_teams_overview = st.multiselect("選擇要顯示的團隊", all_teams, default=all_teams)
        
        if selected_teams_overview:
            df_team = df[df['備註'].isin(selected_teams_overview)]
            
            # 依月份平均
            df_team['月份'] = df_team['篩檢日期'].dt.to_period('M').astype(str)
            team_trend = df_team.groupby(['備註', '月份'])['端粒長度'].mean().reset_index()
            
            fig_trend = px.line(team_trend, x="月份", y="端粒長度", color="備註", markers=True,
                                title="團隊平均端粒長度變化")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            fig_box = px.box(df_team, x="備註", y="端粒長度", color="備註", points="all", 
                             hover_data=["姓名", "年紀"], title="團隊數值分佈")
            st.plotly_chart(fig_box, use_container_width=True)

    # === Tab 2: 個人追蹤 (含篩選功能) ===
    with tab2:
        st.subheader("個人詳細檢測報告")
        
        c1, c2 = st.columns(2)
        with c1:
            # 階層式選單
            all_teams_list = sorted(list(df['備註'].unique()))
            sel_team = st.selectbox("1. 選擇團隊", all_teams_list, key="ind_team")
        with c2:
            people_list = sorted(df[df['備註'] == sel_team]['姓名'].unique())
            sel_person = st.selectbox("2. 選擇人員", people_list, key="ind_person")

        if sel_person:
            person_df = df[df['姓名'] == sel_person].sort_values('篩檢日期')
            latest = person_df.iloc[-1]
            
            # KPI 指標
            real_age = latest['年紀']
            bio_age = latest['端粒年齡']
            diff = bio_age - real_age
            
            # 顏色邏輯：端粒年齡 < 實際年齡 = 綠色 (好)
            delta_color = "inverse" # inverse 表示數值越小越好
            
            k1, k2, k3 = st.columns(3)
            k1.metric("實際年齡", f"{real_age:.0f} 歲")
            k2.metric("端粒生物年齡", f"{bio_age:.1f} 歲", 
                      delta=f"{diff:.1f} 歲 (差距)", delta_color=delta_color)
            k3.metric("最新端粒長度", f"{latest['端粒長度']:.3f}")
            
            # 個人趨勢圖
            fig_p = px.line(person_df, x="篩檢日期", y="端粒長度", markers=True, 
                            title=f"{sel_person} 端粒變化歷程")
            
            # 加上同齡平均線 (使用模型預測該年齡的應有長度)
            expected_val = slope * real_age + intercept
            fig_p.add_hline(y=expected_val, line_dash="dash", line_color="gray",
                            annotation_text=f"同齡平均 ({expected_val:.2f})")
            
            st.plotly_chart(fig_p, use_container_width=True)
            st.dataframe(person_df[['篩檢日期', '端粒長度', '端粒年齡', '備註']].style.format({"端粒長度": "{:.3f}", "端粒年齡": "{:.1f}"}))

    # === Tab 3: 常模與相關性 ===
    with tab3:
        st.subheader("年齡 vs. 端粒長度 (常模分析)")
        
        st.markdown(f"""
        **目前模型基準**：{model_basis}
        - 藍色點：每個人的實際數據
        - 紅色線：根據數據庫推算出的老化趨勢線
        """)
        
        # 篩選顯示團隊
        sel_corr_teams = st.multiselect("篩選團隊", all_teams, default=all_teams, key="corr_team")
        
        if sel_corr_teams:
            df_corr = df[df['備註'].isin(sel_corr_teams)]
            
            # 為了凸顯「基準數據」，我們可以把用來訓練的數據特別標示出來
            # 這裡簡單畫出所有數據的分佈，並加上迴歸線
            fig_scatter = px.scatter(df_corr, x="年紀", y="端粒長度", color="備註",
                                     hover_data=["姓名", "端粒年齡"],
                                     title="資料庫分佈圖")
            
            # 手動畫出模型的迴歸線 (紅線)，這樣不受篩選影響，顯示的是「全域常模」
            x_range = np.linspace(df['年紀'].min(), df['年紀'].max(), 100)
            y_pred = slope * x_range + intercept
            
            fig_scatter.add_traces(go.Scatter(x=x_range, y=y_pred, mode='lines', 
                                              name='常模趨勢線', line=dict(color='red', dash='dash')))

            st.plotly_chart(fig_scatter, use_container_width=True)
            
else:
    st.info("👈 請從左側選單上傳 CSV 檔案")
