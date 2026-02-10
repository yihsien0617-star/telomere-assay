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
        df['備註'] = df['備註'].fillna("未分類").astype(str)
        df['篩檢日期'] = pd.to_datetime(df['篩檢日期'], errors='coerce')
        df = df.dropna(subset=['篩檢日期', '姓名', '端粒長度', '年紀'])
        df = df.sort_values(by='篩檢日期')
        
        # === 核心運算：計算每人的「首末次變化量」 ===
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
                    '備註': first_rec['備註'],
                    '首次數值': first_rec['端粒長度'],
                    '末次數值': last_rec['端粒長度'],
                    '變化量': change,
                    '監測天數': duration,
                    '最後檢測日': last_rec['篩檢日期']
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
            df['端粒年齡'] = df['端粒長度'].apply(calculate_telomere_age)
        else:
            slope, intercept = 0, 0
            df['端粒年齡'] = df['年紀']

    except Exception as e:
        st.error(f"資料處理失敗: {e}")
        st.stop()

    # --- 主要分析頁籤 ---
    tab1, tab2, tab3 = st.tabs(["📊 團隊改善成效 (含趨勢)", "👤 個人追蹤 (多選比較)", "📉 常模相關性"])

    # === Tab 1: 團隊成效 ===
    with tab1:
        st.header("團隊整體改善成效評估")
        
        all_teams = sorted(list(df['備註'].unique()))
        sel_teams_ov = st.multiselect("選擇團隊", all_teams, default=all_teams)
        
        if not df_changes.empty and sel_teams_ov:
            df_ch_filtered = df_changes[df_changes['備註'].isin(sel_teams_ov)]
            
            # KPI
            c1, c2 = st.columns(2)
            c1.metric("選定團隊平均變化量", f"{df_ch_filtered['變化量'].mean():+.3f}")
            c2.metric("總監測人數", f"{len(df_ch_filtered)} 人")

            # 圖表 A: 團隊平均變化排行 (Bar Chart) - 總結性
            st.subheader("1. 團隊總體改善幅度排行")
            team_agg = df_ch_filtered.groupby('備註')['變化量'].mean().reset_index().sort_values('變化量', ascending=False)
            fig_bar = px.bar(team_agg, x='備註', y='變化量', color='變化量',
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             text_auto='.3f', title="團隊平均變化量 (數值越高代表改善越多)")
            fig_bar.add_hline(y=0, line_color="black")
            st.plotly_chart(fig_bar, use_container_width=True)

            # 圖表 B: 團隊變化趨勢折線圖 (Line Chart) - 時間性
            st.subheader("2. 團隊改善趨勢變化 (按月)")
            # 我們需要原始 df 配合變化量概念。這裡簡單做法：計算每個月各團隊的「平均端粒長度」
            # 更好的做法是：計算相對於該團隊「基準線」的變化，但這裡先呈現絕對數值趨勢比較直觀
            df_trend_source = df[df['備註'].isin(sel_teams_ov)].copy()
            df_trend_source['月份'] = df_trend_source['篩檢日期'].dt.to_period('M').astype(str)
            
            # Group by 團隊與月份
            team_month_trend = df_trend_source.groupby(['備註', '月份'])['端粒長度'].mean().reset_index()
            
            fig_line_team = px.line(team_month_trend, x='月份', y='端粒長度', color='備註', markers=True,
                                    title="各團隊平均端粒長度隨時間變化趨勢")
            st.plotly_chart(fig_line_team, use_container_width=True)

            # 圖表 C: 個體分佈細節
            st.subheader("3. 團隊成員個體變化分佈")
            fig_box = px.box(df_ch_filtered, x='備註', y='變化量', color='備註', points="all",
                             hover_data=['姓名'], title="團隊內部差異 (Box Plot)")
            fig_box.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_box, use_container_width=True)
            
        else:
            st.info("資料不足或未選擇團隊。")

    # === Tab 2: 個人追蹤 (改為多選勾選) ===
    with tab2:
        st.subheader("個人詳細檢測報告 (可多選比較)")
        
        # 1. 選擇團隊 (單選，避免名單太長)
        all_teams_list = sorted(list(df['備註'].unique()))
        sel_team_ind = st.selectbox("1. 選擇所屬團隊", all_teams_list, key="ind_team_multi")
        
        # 2. 選擇人員 (多選 Checkbox/Multiselect)
        people_in_team = sorted(df[df['備註'] == sel_team_ind]['姓名'].unique())
        
        # 預設全選太亂，預設不選或是選第一個
        sel_people = st.multiselect("2. 勾選要顯示的人員 (可多選對比)", people_in_team, default=people_in_team[:1] if people_in_team else None)

        if sel_people:
            # 篩選出這些人的資料
            multi_person_df = df[df['姓名'].isin(sel_people)].sort_values('篩檢日期')
            
            # 圖表 A: 多人趨勢比較 (折線圖)
            st.markdown("#### 📈 端粒變化趨勢對比")
            fig_multi_line = px.line(multi_person_df, x="篩檢日期", y="端粒長度", color="姓名", markers=True,
                                     title=f"人員趨勢比較 ({sel_team_ind})")
            
            # 如果只選了一個人，才畫同齡平均線，不然太亂
            if len(sel_people) == 1:
                person_age = multi_person_df.iloc[-1]['年紀']
                expected_val = slope * person_age + intercept
                fig_multi_line.add_hline(y=expected_val, line_dash="dash", line_color="gray", annotation_text="同齡平均")
            
            st.plotly_chart(fig_multi_line, use_container_width=True)

            # 圖表 B: 檢測結果並排 (Grouped Bar Chart)
            st.markdown("#### 📊 各次檢測數值細節")
            # 為了讓 bar chart 好看，將日期轉字串
            multi_person_df['日期標籤'] = multi_person_df['篩檢日期'].dt.strftime('%Y-%m-%d')
            
            fig_multi_bar = px.bar(multi_person_df, x="日期標籤", y="端粒長度", color="姓名", barmode="group",
                                   text="端粒長度", title="各時間點檢測結果對比")
            fig_multi_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_multi_bar, use_container_width=True)

            # 資料表
            st.dataframe(multi_person_df[['姓名', '篩檢日期', '端粒長度', '端粒年齡', '備註']].style.format({"端粒長度": "{:.3f}", "端粒年齡": "{:.1f}"}))
        else:
            st.info("請勾選至少一位人員以顯示圖表。")

    # === Tab 3: 常模 ===
    with tab3:
        st.subheader("常模與相關性分析")
        sel_corr_teams = st.multiselect("篩選顯示團隊", all_teams, default=all_teams, key="corr_team")
        if sel_corr_teams:
            df_corr = df[df['備註'].isin(sel_corr_teams)]
            fig_scatter = px.scatter(df_corr, x="年紀", y="端粒長度", color="備註", hover_data=["姓名"],
                                     title=f"資料庫分佈 (模型基準: {model_basis})")
            x_range = np.linspace(df['年紀'].min(), df['年紀'].max(), 100)
            y_pred = slope * x_range + intercept
            fig_scatter.add_traces(go.Scatter(x=x_range, y=y_pred, mode='lines', name='常模趨勢線', line=dict(color='red', dash='dash')))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
else:
    st.info("👈 請從左側選單上傳 CSV 檔案")
