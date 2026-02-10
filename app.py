import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 設定網頁標題與排版
st.set_page_config(page_title="實驗室端粒分析平台", layout="wide", page_icon="🧬")

# 標題區
st.title("🧬 實驗室端粒趨勢分析平台")
st.markdown("---")

# --- 側邊欄：資料上傳與設定 ---
st.sidebar.header("📂 資料載入")
uploaded_file = st.sidebar.file_uploader("請上傳端粒數據 (CSV)", type=['csv'])

# 範例資料下載連結 (可選)
# st.sidebar.info("如果沒有檔案，請使用格式範本...")

if uploaded_file is not None:
    # 1. 讀取資料
    try:
        df = pd.read_csv(uploaded_file)
        
        # 欄位檢查 (確保格式正確)
        required_cols = ['姓名', '篩檢日期', '端粒長度', '備註', '年紀']
        if not all(col in df.columns for col in required_cols):
            st.error(f"資料格式錯誤！請確保 CSV 包含以下欄位：{required_cols}")
            st.stop()

        # 日期處理
        df['篩檢日期'] = pd.to_datetime(df['篩檢日期'], errors='coerce')
        # 移除日期無效的資料
        df = df.dropna(subset=['篩檢日期', '姓名', '端粒長度'])
        df = df.sort_values(by='篩檢日期')
        
        # 整理出每人的測量次數
        counts = df['姓名'].value_counts()
        df['測量次數'] = df['姓名'].map(counts)

    except Exception as e:
        st.error(f"讀取檔案失敗: {e}")
        st.stop()

    # --- 數據概覽 KPI ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("總樣本數", len(df))
    with col2:
        st.metric("監測人數", df['姓名'].nunique())
    with col3:
        st.metric("平均端粒長度", f"{df['端粒長度'].mean():.2f}")
    with col4:
        # 計算有多少人有進步 (末次 > 首次)
        improvers = 0
        valid_people = 0
        for name, group in df[df['測量次數'] >= 2].groupby('姓名'):
            if group.iloc[-1]['端粒長度'] > group.iloc[0]['端粒長度']:
                improvers += 1
            valid_people += 1
        st.metric("進步人數 (數值增加)", f"{improvers} / {valid_people}")

    st.markdown("---")

    # --- 主要分析頁籤 ---
    tab1, tab2, tab3 = st.tabs(["📊 團隊趨勢概覽", "👤 個人變化追蹤", "📉 年齡相關性"])

    # === Tab 1: 團隊趨勢 ===
    with tab1:
        col_t1, col_t2 = st.columns([1, 3])
        with col_t1:
            # 篩選器
            all_teams = list(df['備註'].unique())
            selected_teams = st.multiselect("選擇團隊", all_teams, default=all_teams[:3] if len(all_teams)>3 else all_teams)
        
        # 過濾資料
        df_team = df[df['備註'].isin(selected_teams)]

        # 圖表 1: 團隊分佈箱型圖
        fig_box = px.box(df_team, x="備註", y="端粒長度", color="備註", 
                         title="各團隊端粒長度分佈 (Box Plot)", points="all", hover_data=["姓名"])
        st.plotly_chart(fig_box, use_container_width=True)

        # 圖表 2: 時間趨勢圖
        # 將日期轉為月份字串以便分組
        df_team['月份'] = df_team['篩檢日期'].dt.to_period('M').astype(str)
        team_trend = df_team.groupby(['備註', '月份'])['端粒長度'].mean().reset_index()
        
        fig_trend = px.line(team_trend, x="月份", y="端粒長度", color="備註", markers=True,
                            title="團隊平均端粒長度變化趨勢")
        st.plotly_chart(fig_trend, use_container_width=True)

    # === Tab 2: 個人變化 ===
    with tab2:
        st.subheader("🔍 個人數據深度分析")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            # 找出變化量最大的前幾名
            df_change_list = []
            for name, group in df[df['測量次數']>=2].groupby('姓名'):
                group = group.sort_values('篩檢日期')
                start = group.iloc[0]['端粒長度']
                end = group.iloc[-1]['端粒長度']
                change = end - start
                team = group.iloc[0]['備註']
                df_change_list.append({'姓名': name, '團隊': team, '變化量': change, '首次': start, '末次': end})
            
            df_changes = pd.DataFrame(df_change_list)
            
            if not df_changes.empty:
                df_changes = df_changes.sort_values(by='變化量', ascending=False)
                
                # 顯示進步榜
                st.write("🏆 **進步最多 (Top 5)**")
                st.dataframe(df_changes.head(5).style.format({'變化量': '{:+.2f}', '首次': '{:.2f}', '末次': '{:.2f}'}))
                
                # 顯示退步榜
                st.write("⚠️ **需關注 (Bottom 5)**")
                st.dataframe(df_changes.tail(5).sort_values('變化量').style.format({'變化量': '{:+.2f}', '首次': '{:.2f}', '末次': '{:.2f}'}))
            else:
                st.info("目前沒有足夠的重複測量數據來計算變化量。")

        with c2:
            # 指定人員查詢
            search_name = st.selectbox("搜尋特定人員詳細歷程", df['姓名'].unique())
            person_data = df[df['姓名'] == search_name].sort_values('篩檢日期')
            
            # 繪製個人折線圖
            fig_person = px.line(person_data, x="篩檢日期", y="端粒長度", markers=True, 
                                 title=f"{search_name} ({person_data.iloc[0]['備註']}) 的檢測歷程")
            
            # 加入正常參考區間 (示意圖，假設 2.5-8.0)
            fig_person.add_hrect(y0=2.5, y1=8.0, line_width=0, fillcolor="green", opacity=0.1, annotation_text="一般區間")
            
            st.plotly_chart(fig_person, use_container_width=True)
            st.dataframe(person_data[['篩檢日期', '端粒長度', '年紀', '備註']])

    # === Tab 3: 年齡相關性 ===
    with tab3:
        st.subheader("📉 生理年齡與端粒長度")
        fig_scatter = px.scatter(df, x="年紀", y="端粒長度", color="備註", 
                                 hover_name="姓名", trendline="ols",
                                 title="年齡 vs 端粒長度迴歸分析")
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    # 歡迎畫面
    st.info("👈 請從左側選單上傳您的 CSV 檔案以開始分析")
    st.markdown("""
    ### 準備您的 CSV 檔案
    請確保您的 Excel 另存為 CSV 檔案，並包含以下欄位名稱：
    - `姓名`
    - `篩檢日期` (例如: 2025-01-01)
    - `端粒長度` (數值)
    - `備註` (作為團隊名稱)
    - `年紀`
    """)
