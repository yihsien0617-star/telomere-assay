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

# 修改點 1: 允許上傳 CSV 或 Excel
uploaded_file = st.sidebar.file_uploader("請上傳端粒數據 (CSV/Excel)", type=['csv', 'xlsx', 'xls'])

# 模型建立基準選擇
st.sidebar.subheader("🧮 端粒年齡推算模型")
model_basis = st.sidebar.radio(
    "選擇常模建立基準：",
    ("僅用首次檢測數據 (推薦)", "使用所有檢測數據"),
    help="推薦使用「首次檢測」作為基準，以排除後續干預措施對常模的影響。"
)

if uploaded_file is not None:
    try:
        # 修改點 2: 判斷檔案類型並讀取
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            # 預設讀取 Excel 的第一個工作表
            df = pd.read_excel(uploaded_file)
        
        # 欄位檢查
        required_cols = ['姓名', '篩檢日期', '端粒長度', '備註', '年紀']
        
        # 檢查欄位是否存在，若缺欄位可能是讀錯工作表
        if not all(col in df.columns for col in required_cols):
            st.error(f"資料格式錯誤！請確保檔案包含以下欄位：{required_cols}")
            st.warning("如果您上傳的是 Excel，請確保數據位於「第一個工作表」。")
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
                    '首次日期': first_rec['篩檢日期'],
                    '末次日期': last_rec['篩檢日期']
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

    # --- 定義 Tab ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 團隊改善成效", 
        "👤 個人追蹤 (多選比較)", 
        "📉 常模相關性", 
        "🔄 團隊前後測追蹤"
    ])

    # === Tab 1: 團隊成效 ===
    with tab1:
        st.header("團隊整體改善成效評估")
        all_teams = sorted(list(df['備註'].unique()))
        sel_teams_ov = st.multiselect("選擇團隊", all_teams, default=all_teams)
        
        if not df_changes.empty and sel_teams_ov:
            df_ch_filtered = df_changes[df_changes['備註'].isin(sel_teams_ov)]
            
            c1, c2 = st.columns(2)
            c1.metric("選定團隊平均變化量", f"{df_ch_filtered['變化量'].mean():+.3f}")
            c2.metric("總監測人數", f"{len(df_ch_filtered)} 人")

            st.subheader("1. 團隊總體改善幅度排行")
            team_agg = df_ch_filtered.groupby('備註')['變化量'].mean().reset_index().sort_values('變化量', ascending=False)
            fig_bar = px.bar(team_agg, x='備註', y='變化量', color='變化量',
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             text_auto='.3f')
            fig_bar.add_hline(y=0, line_color="black")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("2. 團隊改善趨勢變化 (按月)")
            df_trend_source = df[df['備註'].isin(sel_teams_ov)].copy()
            df_trend_source['月份'] = df_trend_source['篩檢日期'].dt.to_period('M').astype(str)
            team_month_trend = df_trend_source.groupby(['備註', '月份'])['端粒長度'].mean().reset_index()
            fig_line_team = px.line(team_month_trend, x='月份', y='端粒長度', color='備註', markers=True)
            st.plotly_chart(fig_line_team, use_container_width=True)
        else:
            st.info("資料不足或未選擇團隊。")

    # === Tab 2: 個人追蹤 ===
    with tab2:
        st.subheader("個人詳細檢測報告 (可多選比較)")
        all_teams_list = sorted(list(df['備註'].unique()))
        sel_team_ind = st.selectbox("1. 選擇所屬團隊", all_teams_list, key="ind_team_multi")
        people_in_team = sorted(df[df['備註'] == sel_team_ind]['姓名'].unique())
        sel_people = st.multiselect("2. 勾選要顯示的人員", people_in_team, default=people_in_team[:1] if people_in_team else None)

        if sel_people:
            multi_person_df = df[df['姓名'].isin(sel_people)].sort_values('篩檢日期')
            
            st.markdown("#### 📈 端粒變化趨勢對比")
            fig_multi_line = px.line(multi_person_df, x="篩檢日期", y="端粒長度", color="姓名", markers=True)
            st.plotly_chart(fig_multi_line, use_container_width=True)

            st.markdown("#### 📊 各次檢測數值細節")
            multi_person_df['日期標籤'] = multi_person_df['篩檢日期'].dt.strftime('%Y-%m-%d')
            fig_multi_bar = px.bar(multi_person_df, x="日期標籤", y="端粒長度", color="姓名", barmode="group", text="端粒長度")
            fig_multi_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_multi_bar, use_container_width=True)
        else:
            st.info("請勾選至少一位人員。")

    # === Tab 3: 常模 ===
    with tab3:
        st.subheader("常模與相關性分析")
        sel_corr_teams = st.multiselect("篩選顯示團隊", all_teams, default=all_teams, key="corr_team")
        if sel_corr_teams:
            df_corr = df[df['備註'].isin(sel_corr_teams)]
            fig_scatter = px.scatter(df_corr, x="年紀", y="端粒長度", color="備註", hover_data=["姓名"])
            x_range = np.linspace(df['年紀'].min(), df['年紀'].max(), 100)
            y_pred = slope * x_range + intercept
            fig_scatter.add_traces(go.Scatter(x=x_range, y=y_pred, mode='lines', name='常模趨勢線', line=dict(color='red', dash='dash')))
            st.plotly_chart(fig_scatter, use_container_width=True)

    # === Tab 4: 團隊前後測追蹤 ===
    with tab4:
        st.header("🔄 團隊前後測結果對比")
        target_team = st.selectbox("請選擇要分析的團隊", all_teams, key="pre_post_team_select")
        team_changes = df_changes[df_changes['備註'] == target_team]

        if not team_changes.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("平均首次數值", f"{team_changes['首次數值'].mean():.3f}")
            c2.metric("平均末次數值", f"{team_changes['末次數值'].mean():.3f}")
            c3.metric("平均變化量", f"{team_changes['變化量'].mean():+.3f}")

            st.markdown("---")
            st.subheader(f"{target_team} - 成員前後測變化斜率圖")
            
            fig_slope = go.Figure()
            for i, row in team_changes.iterrows():
                color = 'green' if row['變化量'] >= 0 else 'red'
                fig_slope.add_trace(go.Scatter(
                    x=['首次', '末次'], y=[row['首次數值'], row['末次數值']],
                    mode='lines+markers', name=row['姓名'],
                    line=dict(color=color, width=2), marker=dict(size=8),
                    hovertext=f"{row['姓名']}: {row['變化量']:+.3f}"
                ))
                # 數值標籤
                fig_slope.add_trace(go.Scatter(
                    x=['末次'], y=[row['末次數值']], mode='text',
                    text=[f"{row['姓名']} ({row['末次數值']:.2f})"],
                    textposition="middle right", showlegend=False
                ))

            fig_slope.update_layout(xaxis=dict(showgrid=False), yaxis=dict(title="端粒長度"), showlegend=False)
            st.plotly_chart(fig_slope, use_container_width=True)

            st.subheader("詳細數據表")
            # 這裡就是上次修正錯誤的地方，確保 background_gradient 可以運作
            st.dataframe(team_changes[['姓名', '首次數值', '末次數值', '變化量']].sort_values('變化量', ascending=False)
                         .style.format({'首次數值':'{:.3f}','末次數值':'{:.3f}','變化量':'{:+.3f}'})
                         .background_gradient(subset=['變化量'], cmap='RdYlGn'))
        else:
            st.warning(f"團隊 {target_team} 沒有足夠的前後測數據。")

else:
    st.info("👈 請從左側選單上傳 CSV 或 Excel 檔案")
