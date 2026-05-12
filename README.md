# 大学生消费行为画像与消费趋势预测

本项目用于完成“大学生消费行为画像与消费趋势预测”的数据处理、模型分析、图表生成。分析内容围绕校园消费流水和问卷调查统计结果展开，主要包括问卷数据提取、消费特征构建、KMeans++ 消费画像聚类、SARIMA/随机森林/LSTM 趋势预测。

## 项目内容

项目主要完成以下工作：

- 从问卷 Word 报告中提取题目、选项、比例和有效填写人次。
- 对校园消费流水进行清洗、脱敏和学生层面特征聚合。
- 基于肘部法则和聚类评价指标确定消费画像聚类数量。
- 将学生消费群体按消费金额水平命名为高消费、中消费和低消费三类。
- 构建日度消费金额和交易笔数序列，并比较 SARIMA、随机森林、LSTM 和融合模型的预测效果。
- 输出图表、结果表格。

## 文件说明

| 文件或目录 | 说明 |
|---|---|
| `extract_docx_data.py` | 问卷 Word 报告提取脚本，将 DOCX 转换为结构化 CSV |
| `analysis_pipeline.py` | 主要分析脚本，完成消费数据清洗、特征构建、聚类、预测和图表生成 |
| `大学生消费情况调查问卷2－默认报告.docx` | 问卷调查统计报告 |
| `2023年9-12月理学院(1).xlsx` | 校园消费流水数据 |
| `output/extracted/` | 问卷提取后的结构化数据 |
| `output/analysis/` | 模型分析结果、图表和中间指标 |

## 环境依赖

建议使用 Python 3.10 及以上版本。主要依赖如下：

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels torch python-docx openpyxl
```

其中：

- `python-docx` 用于读取问卷 Word 报告。
- `openpyxl` 用于读取 Excel 消费流水。
- `scikit-learn` 用于 KMeans++ 聚类、PCA、随机森林和评价指标。
- `statsmodels` 用于 SARIMA 时间序列预测。
- `torch` 用于 LSTM 模型。

## 运行步骤

第一次运行或问卷 Word 文件发生变化时，先提取问卷数据：

```bash
python extract_docx_data.py
```

该步骤会生成：

- `output/extracted/questionnaire_summary.csv`
- `output/extracted/questionnaire_questions.csv`

随后运行主分析脚本：

```bash
python analysis_pipeline.py
```

该步骤会读取已提取的问卷 CSV 和消费流水 Excel，生成聚类结果、预测结果和图表。

## 输出结果

主分析结果保存在 `output/analysis/` 目录下。

| 输出文件 | 说明 |
|---|---|
| `student_features_with_clusters.csv` | 脱敏后的学生级消费特征与聚类类别 |
| `cluster_profile.csv` | 各消费类别的均值特征和人数占比 |
| `daily_trend.csv` | 日度消费金额、交易笔数和活跃学生数 |
| `forecast_metrics.csv` | 各预测模型的 MAE、RMSE、MAPE 指标 |
| `daily_forecast.csv` | 测试集预测结果和未来预测结果 |
| `elbow_method/cluster_evaluation.csv` | 不同聚类数下的评价指标 |

主要图表保存在 `output/analysis/figures/` 和 `output/analysis/elbow_method/`。

| 图表 | 路径 |
|---|---|
| 日度消费金额趋势 | `output/analysis/figures/daily_total_amount_trend.png` |
| 问卷月消费占比最高项 | `output/analysis/figures/questionnaire_top_consumption_items.png` |
| 消费画像群体人数分布 | `output/analysis/figures/cluster_distribution.png` |
| 最终聚类 PCA 投影 | `output/analysis/figures/cluster_pca.png` |
| 各画像群体餐次金额占比 | `output/analysis/figures/cluster_meal_share.png` |
| 聚类中心特征热力图 | `output/analysis/figures/cluster_center_heatmap.png` |
| SSE 肘部法则图 | `output/analysis/elbow_method/cluster_elbow_inertia.png` |
| SSE 边际下降率图 | `output/analysis/elbow_method/cluster_inertia_drop_rate.png` |
| 轮廓系数图 | `output/analysis/elbow_method/cluster_silhouette_score.png` |
| CH 指数与 DB 指数图 | `output/analysis/elbow_method/cluster_validity_indices.png` |
| 随机森林消费金额预测 | `output/analysis/figures/forecast_random_forest_total_amount_test.png` |
| 随机森林交易笔数预测 | `output/analysis/figures/forecast_random_forest_transaction_count_test.png` |
| LSTM 消费金额预测 | `output/analysis/figures/forecast_lstm_total_amount_test.png` |
| LSTM 交易笔数预测 | `output/analysis/figures/forecast_lstm_transaction_count_test.png` |

## 核心结果

消费流水清洗后共保留 145,051 条记录，覆盖 919 名学生，样本日期范围为 2023-12-01 至 2024-04-18。消费总额为 586,279.80，单笔消费均值为 4.04。

聚类分析采用 KMeans++ 模型，候选聚类数设置为 $k=2$ 至 $k=8$。结合 SSE 肘部法则、边际下降率和聚类解释性，最终选择 $k=3$。三类消费群体按人均总消费金额由高到低命名为：

| 类别 | 人数 | 占比 | 人均总消费金额 | 人均交易笔数 |
|---|---:|---:|---:|---:|
| 类1-高消费 | 426 | 46.35% | 1102.45 | 278.30 |
| 类2-中消费 | 423 | 46.03% | 261.93 | 59.28 |
| 类3-低消费 | 70 | 7.62% | 83.47 | 20.30 |

趋势预测部分以日度消费金额和日度交易笔数为预测对象，测试集为最后 14 天。模型比较结果显示，随机森林滞后特征模型在两个预测目标上表现最好：

| 预测目标 | 最优模型 | MAE | RMSE | MAPE |
|---|---|---:|---:|---:|
| 消费金额 | 随机森林滞后特征 | 835.65 | 1060.80 | 29.81% |
| 交易笔数 | 随机森林滞后特征 | 203.37 | 255.42 | 28.32% |

LSTM 已纳入模型比较并生成独立预测图，但由于当前日度序列长度有限，预测误差高于随机森林。因此，本文将 LSTM 作为深度学习对照模型和非线性补充模型，不作为本次最优推荐模型。

## 注意事项

- 聚类类别名称仅表示消费金额层级，不直接代表经济困难、消费风险或其他管理语义。
- 问卷数据为汇总统计结果，不能与学生个体消费流水一一匹配。
- 当前趋势预测样本跨度有限，LSTM 结果应谨慎解释。
- 若重新运行脚本，`output/analysis/` 下的结果文件和图表会按当前数据重新生成。

