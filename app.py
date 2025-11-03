
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px  # <<< MODIFIED: 確保引入 Plotly
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
@@ -216,7 +216,7 @@
                'Invalid': '#cccccc', 'Error': '#888888'
            }

            # --- 1. 情感分佈圓餅圖 (您的版本，可正常運作) ---
            # --- 1. 情感分佈圓餅圖 (No changes) ---
            st.subheader("1. Sentiment Distribution (Pie)")
            sentiment_series = df_result['sentiment'].dropna().astype(str)
            sentiment_counts = sentiment_series.value_counts()
@@ -236,109 +236,99 @@
            else:
                st.info("No sentiment data available for pie chart.")

            # <<< MODIFIED BLOCK START: 修正並優化每日趨勢圖 >>>

            # --- 每日趨勢圖的數據準備 (使用更穩健的 reindex) ---
            # <<< MODIFIED BLOCK START: 實現兩張獨立的每日趨勢圖 >>>
            
            st.subheader("2. Daily Sentiment Trend")
            
            # --- 數據準備 (共用) ---
            if 'published_at_hk' in df_result.columns:
                df_result['date'] = df_result['published_at_hk'].dt.date
            else:
                df_result['date'] = df_result['published_at'].dt.date

            daily = df_result.groupby(['date', 'sentiment']).size().unstack().fillna(0)
            # 使用 reindex 確保所有情感類別都存在且順序正確，即使某些類別沒有數據
            daily = daily.reindex(columns=sentiments_order).dropna(axis=1, how='all')

            if not daily.empty:
                # --- 2a. 每日情感趨勢 (方案一：互動式 Plotly 圖表 - 推薦) ---
                st.subheader("2. Daily Sentiment Trend (Interactive Chart)")
                st.markdown("**(推薦)** 此圖表可縮放、平移和懸停查看數據，完美解決標籤擁擠問題。")
                
                # Plotly 需要 "long-form" data，所以進行轉換
                # 將數據從 "wide" 轉為 "long" 格式，方便 Plotly 使用
                daily_long = daily.reset_index().melt(id_vars='date', var_name='sentiment', value_name='count')

                fig_plotly = px.area(
                # --- 圖表 2a: 每日情感趨勢 (折線圖) ---
                st.markdown("#### 每日情感趨勢 (折線圖)")
                st.markdown("此圖表展示各情感類別每日的留言數量變化，適合比較不同情感的熱度趨勢。")
                
                fig_line = px.line(
                    daily_long,
                    x='date',
                    y='count',
                    color='sentiment',
                    title='Daily Comment Volume by Sentiment',
                    title='Daily Comment Volume Trend by Sentiment',
                    labels={'date': 'Date', 'count': 'Number of Comments', 'sentiment': 'Sentiment'},
                    color_discrete_map=colors_map,
                    category_orders={'sentiment': [col for col in sentiments_order if col in daily.columns]}
                )
                fig_plotly.update_layout(legend_title_text='Sentiment')
                st.plotly_chart(fig_plotly, use_container_width=True)

                # --- 2b. 每日情感趨勢 (方案二：優化 Matplotlib 靜態圖) ---
                with st.expander("查看靜態 Matplotlib 優化圖表"):
                    st.markdown("此為使用 Matplotlib 繪製的靜態堆疊面積圖，透過智慧日期格式化解決了標籤重疊問題。")
                    
                    fig2, ax2 = plt.subplots(figsize=(12, 5))
                    
                    # 使用 Pandas 內建的 plot 功能，更簡潔穩健
                    daily.plot(kind='area', stacked=True, ax=ax2, 
                               color=[colors_map[col] for col in daily.columns],
                               linewidth=0.5)
                    
                    ax2.set_title('Daily Comment Volume by Sentiment (Static)', fontsize=16)
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Number of Comments')
                    
                    # 核心優化：使用自動日期定位器和格式化器
                    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
                    formatter = mdates.ConciseDateFormatter(locator)
                    ax2.xaxis.set_major_locator(locator)
                    ax2.xaxis.set_major_formatter(formatter)
                    
                    ax2.legend(title='Sentiment')
                    ax2.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
                    fig2.autofmt_xdate()
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                fig_line.update_layout(legend_title_text='Sentiment')
                st.plotly_chart(fig_line, use_container_width=True)

                # --- 圖表 2b: 每日留言總量 (堆疊長條圖) ---
                st.markdown("#### 每日留言總量及情感分佈 (堆疊長條圖)")
                st.markdown("此圖表展示每日的總留言量，並以顏色區分其中各種情感的佔比。")

                fig_bar = px.bar(
                    daily_long,
                    x='date',
                    y='count',
                    color='sentiment',
                    title='Daily Comment Volume by Sentiment (Stacked)',
                    labels={'date': 'Date', 'count': 'Number of Comments', 'sentiment': 'Sentiment'},
                    color_discrete_map=colors_map,
                    category_orders={'sentiment': [col for col in sentiments_order if col in daily.columns]}
                )
                fig_bar.update_layout(legend_title_text='Sentiment', barmode='stack')
                st.plotly_chart(fig_bar, use_container_width=True)

            else:
                st.info("Not enough daily sentiment data to display the trend chart.")
                st.info("Not enough daily sentiment data to display the trend charts.")

            # <<< MODIFIED BLOCK END >>>

            # --- 3. 各主題情感佔比 (您的版本，稍作穩健性修改) ---
            # --- 3. 各主題情感佔比 (No changes) ---
            st.subheader("3. Sentiment Share by Topic")
            topic_sentiment = df_result.groupby(['topic', 'sentiment']).size().unstack().fillna(0)
            # 同樣使用 reindex 確保欄位和順序
            topic_sentiment = topic_sentiment.reindex(columns=sentiments_order).dropna(axis=1, how='all')

            if not topic_sentiment.empty:
                # 過濾掉總和為0的主題，避免除以零的錯誤
                topic_sentiment = topic_sentiment[topic_sentiment.sum(axis=1) > 0]

                if not topic_sentiment.empty:
                    topic_sentiment_percent = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0).fillna(0) * 100

                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    topic_sentiment_percent.plot(
                        kind='bar',
                        stacked=True,
                        ax=ax3,
                        color=[colors_map[col] for col in topic_sentiment_percent.columns]
                    )
                    ax3.set_title('Sentiment Share by Topic', fontsize=16)
                    ax3.set_xlabel('Topic')
                    ax3.set_ylabel('Percentage (%)')
                    ax3.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
                    plt.xticks(rotation=45, ha='right')
                    ax3.legend(title='Sentiment')
                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                else:
                    st.info("No topic data with comments to display the chart.")
            else:
                st.info("Not enough topic sentiment data to display the stacked bar chart.")

            # --- 4. 下載分析明細 (No changes) ---
            st.subheader("4. 下載分析明細")
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 下載全部分析明細 (CSV)",
                csv,
                file_name=f"{movie_title}_analysis_details.csv",
                mime='text/csv'
            )
