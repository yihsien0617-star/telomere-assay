import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 設定網頁標題
st.set_page_config(page_title="實驗室端粒分析平台", layout="wide", page_icon="🧬")
st.title("🧬 實驗室端粒趨勢分析平台")
st.markdown("---")

# --- 輔助函式：讀取檔案 ---
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    # 欄位檢查
    required_cols = ['姓名', '篩檢日期', '端粒長度', '備註', '年紀']
    if not all(col in df.columns for col in required_cols):
        return None, f"缺少必要欄位: {list(set(required_cols) - set(df.columns))}"
    
    # 清洗
    df['備註'] = df['備註'].fillna("未分類").astype(str)
    df['篩檢日期'] = pd.to_datetime(df['篩檢日期'], errors='coerce')
    df = df.dropna(subset=['篩檢日期', '姓名', '端粒長度', '年紀'])
    df = df.sort_values(by='篩檢日期')
    return df, None

# --- 側邊欄：資料載入 ---
st.sidebar.header("📂 1. 分析目標資料 (必填)")
uploaded_file = st.sidebar.file_uploader("上傳要製作報告的數據 (CSV/Excel)", type=['csv', 'xlsx', 'xls'], key="main_upload")

st.sidebar.markdown("---")
st.sidebar.header("🧮 2. 端粒年齡推算模型")

uploaded_ref_file = st.sidebar.file_uploader("上傳「常模資料庫」數據 (選填)", type=['csv', 'xlsx', 'xls'], 
                                             key="ref_upload", help="若未上傳，系統將直接使用上方的分析數據來建立模型。")

model_basis = st.sidebar.radio(
    "常模篩選基準：",
    ("僅用首次檢測數據 (推薦)", "使用所有檢測數據"),
    help="推薦只取每人的「首次」數據建立模型，以排除後續干預影響。"
)

# --- 主程式邏輯 ---
if uploaded_file is not None:
    # 1. 讀取資料
    df, err = load_data(uploaded_file)
    if err:
        st.error(err)
        st.stop()
        
    # 2. 決定模型資料來源
    if uploaded_ref_file is not None:
        df_ref, err_ref = load_data(uploaded_ref_file)
        if err_ref:
            st.warning(f"常模讀取失敗: {err_ref}，已降級使用主資料。")
            df_ref = df.copy()
            source_name = "主分析資料"
        else:
            source_name = "獨立常模資料庫"
    else:
        df_ref = df.copy()
        source_name = "主分析資料 (自建常模)"

    # 3. 建立模型
    if model_basis == "僅用首次檢測數據 (推薦)":
        train_df = df_ref.sort_values('篩檢日期').drop_duplicates(subset='姓名', keep='first')
    else:
        train_df = df_ref

    valid_train = train_df[(train_df['年紀'] > 0) & (train_df['端粒長度'] > 0)]

    if len(valid_train) > 5:
        slope, intercept = np.polyfit(valid_train['年紀'], valid_train['端粒長度'], 1)
        st.sidebar.success(f"✅ 模型已建立 ({len(valid_train)} 人)\n`y = {slope:.4f}x + {intercept:.2f}`")

        def calculate_telomere_age(length):
            if slope == 0: return 0
            return (length - intercept) / slope
        
        # 應用模型
        df['端粒年齡'] = df['端粒長度'].apply(calculate_telomere_age)
        df_ref['端粒年齡'] = df_ref['端粒長度'].apply(calculate_telomere_age)
    else:
        st.warning("⚠️ 樣本不足，無法建立模型。")
        slope, intercept = 0, 0
        df['端粒年齡'] = df['年紀']
        df_ref['端粒年齡'] = df_ref['年紀']

    # === 計算前後測統計 (新增：抓取端粒年齡) ===
    person_stats = []
    for name, group in df.groupby('姓名'):
        if len(group) >= 2:
            group = group.sort_values('篩檢日期')
            first_rec = group.iloc[0]
            last_rec = group.iloc[-1]
            person_stats.append({
                '姓名': name,
                '備註': first_rec['備註'],
                '首次數值': first_rec['端粒長度'],
                '末次數值': last_rec['端粒長度'],
                '首次年齡': first_rec['端粒年齡'],  # 新增
                '末次年齡': last_rec['端粒年齡'],  # 新增
                '變化量': last_rec['端粒長度'] - first_rec['端粒長度'],
                '首次日期': first_rec['篩檢日期'],
                '末次日期': last_rec['篩檢日期']
            })
    df_changes = pd.DataFrame(person_stats)
    all_teams = sorted(list(df['備註'].unique()))

    # --- 頁籤 ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 團隊改善成效", 
        "👤 個人追蹤 (多選比較)", 
        "📉 資料庫常模分析", 
        "🔄 團隊前後測追蹤"
    ])

    # === Tab 1: 團隊成效 ===
    with tab1:
        st.header("團隊整體改善成效評估")
        sel_teams_ov = st.multiselect("選擇團隊", all_teams, default=all_teams)
        
        if not df_changes.empty and sel_teams_ov:
            df_ch_filtered = df_changes[df_changes['備註'].isin(sel_teams_ov)]
            c1, c2 = st.columns(2)
            c1.metric("平均變化量", f"{df_ch_filtered['變化量'].mean():+.3f}")
            c2.metric("總監測人數", f"{len(df_ch_filtered)} 人")

            st.subheader("1. 團隊改善幅度排行")
            team_agg = df_ch_filtered.groupby('備註')['變化量'].mean().reset_index().sort_values('變化量', ascending=False)
            fig_bar = px.bar(team_agg, x='備註', y='變化量', color='變化量', color_continuous_scale=px.colors.diverging.Tealrose, text_auto='.3f')
            fig_bar.add_hline(y=0, line_color="black")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("2. 團隊改善趨勢 (按月)")
            df_trend = df[df['備註'].isin(sel_teams_ov)].copy()
            df_trend['月份'] = df_trend['篩檢日期'].dt.to_period('M').astype(str)
            fig_line = px.line(df_trend.groupby(['備註', '月份'])['端粒長度'].mean().reset_index(), x='月份', y='端粒長度', color='備註', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("資料不足。")

    # === Tab 2: 個人追蹤 (更新：底部表格) ===
    with tab2:
        st.subheader("個人詳細檢測報告")
        sel_team_ind = st.selectbox("1. 選擇團隊", all_teams, key="ind_team")
        people_in_team = sorted(df[df['備註'] == sel_team_ind]['姓名'].unique())
        sel_people = st.multiselect("2. 勾選人員", people_in_team, default=people_in_team[:1] if people_in_team else None)

        if sel_people:
            multi_df = df[df['姓名'].isin(sel_people)].sort_values('篩檢日期')
            
            st.markdown("#### 📈 端粒變化趨勢")
            fig_line_p = px.line(multi_df, x="篩檢日期", y="端粒長度", color="姓名", markers=True)
            st.plotly_chart(fig_line_p, use_container_width=True)

            st.markdown("#### 📊 檢測數值細節")
            multi_df['日期'] = multi_df['篩檢日期'].dt.strftime('%Y-%m-%d')
            fig_bar_p = px.bar(multi_df, x="日期", y="端粒長度", color="姓名", barmode="group", text="端粒長度")
            fig_bar_p.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_bar_p, use_container_width=True)
            
            # 新增：詳細數據表
            st.markdown("#### 📋 詳細數據列表")
            st.dataframe(multi_df[['姓名', '篩檢日期', '端粒長度', '端粒年齡', '備註']]
                         .style.format({"端粒長度": "{:.3f}", "端粒年齡": "{:.1f}"}))
        else:
            st.info("請勾選人員。")

    # === Tab 3: 常模分析 ===
    with tab3:
        st.subheader(f"常模分析 (來源: {source_name})")
        sel_corr_teams = st.multiselect("篩選顯示團隊", all_teams, default=all_teams, key="corr_team")
        
        fig_scatter = go.Figure()
        # 背景
        if source_name == "獨立常模資料庫":
            fig_scatter.add_trace(go.Scatter(x=df_ref['年紀'], y=df_ref['端粒長度'], mode='markers', name='常模資料庫', marker=dict(color='lightgray', size=5)))
        
        # 前景
        if sel_corr_teams:
            df_target = df[df['備註'].isin(sel_corr_teams)]
            for team_name in sel_corr_teams:
                sub = df_target[df_target['備註'] == team_name]
                fig_scatter.add_trace(go.Scatter(
                    x=sub['年紀'], y=sub['端粒長度'], mode='markers', name=team_name,
                    text=sub['姓名'], hovertemplate="<b>%{text}</b><br>長度: %{y:.3f}<br>端粒年齡: %{customdata:.1f}",
                    customdata=sub['端粒年齡']
                ))

        # 趨勢線
        x_range = np.linspace(df_ref['年紀'].min(), df_ref['年紀'].max(), 100)
        y_pred = slope * x_range + intercept
        fig_scatter.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='趨勢線', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # === Tab 4: 前後測追蹤 (更新：表格欄位) ===
    with tab4:
        st.header("🔄 團隊前後測結果對比")
        target_team = st.selectbox("選擇團隊", all_teams, key="pp_team")
        team_changes = df_changes[df_changes['備註'] == target_team]

        if not team_changes.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("平均首次數值", f"{team_changes['首次數值'].mean():.3f}")
            c2.metric("平均末次數值", f"{team_changes['末次數值'].mean():.3f}")
            c3.metric("平均變化", f"{team_changes['變化量'].mean():+.3f}")
            
            st.subheader(f"{target_team} - 變化斜率圖")
            fig_slope = go.Figure()
            for i, row in team_changes.iterrows():
                color = 'green' if row['變化量'] >= 0 else 'red'
                fig_slope.add_trace(go.Scatter(
                    x=['首次', '末次'], y=[row['首次數值'], row['末次數值']],
                    mode='lines+markers', name=row['姓名'], line=dict(color=color),
                    hovertext=f"{row['姓名']}: {row['變化量']:+.3f}"
                ))
            fig_slope.update_layout(xaxis=dict(showgrid=False), yaxis=dict(title="端粒長度"), showlegend=False)
            st.plotly_chart(fig_slope, use_container_width=True)

            st.subheader("詳細數據表 (含端粒年齡)")
            # 更新：加入端粒年齡欄位
            display_cols = ['姓名', '首次數值', '首次年齡', '末次數值', '末次年齡', '變化量']
            st.dataframe(team_changes[display_cols].sort_values('變化量', ascending=False)
                         .style.format({
                             '首次數值':'{:.3f}', '末次數值':'{:.3f}', '變化量':'{:+.3f}',
                             '首次年齡':'{:.1f}', '末次年齡':'{:.1f}'
                          })
                         .background_gradient(subset=['變化量'], cmap='RdYlGn'))
        else:
            st.warning("無前後測數據。")

else:
    st.info("👈 請上傳資料")
