# 集成學習 Ensemble Learning

隨機森林 (Bagging) 與 XGBoost (Boosting) 的 R 實作練習。

| 檔案 | 內容 |
|---|---|
| `01_RF_Test.R` | 課堂範例：隨機森林 — Iris 分類、Boston Housing 迴歸、mtry 影響 |
| `02_XGBoost_Test.R` | 課堂範例：XGBoost — Iris 分類、Boston Housing 迴歸、max_depth 影響 |
| `03_Ensemble_Baseball.R` | 作業：以 Baseball 資料預測 `Logsalary`，比較 CART / 隨機森林 / XGBoost |
| `data/ball.txt` | Baseball 資料 (263 位球員) |
| `output/` | 執行紀錄、圖形 (PDF)、模型比較表 |

## 執行

```r
install.packages(c("rpart", "rpart.plot", "randomForest", "xgboost", "caTools",
                   "ggplot2", "GGally", "gridExtra", "mlbench"))
```

```bash
Rscript 01_RF_Test.R
Rscript 02_XGBoost_Test.R
Rscript 03_Ensemble_Baseball.R
```

## Baseball 結果 (測試集 20%)

| Model | MSE | RMSE | R² |
|---|---|---|---|
| Regression Tree (CART) | 0.2325 | 0.4822 | 0.7007 |
| Random Forest (Bagging) | 0.1778 | 0.4217 | 0.7711 |
| XGBoost (Boosting) | 0.1827 | 0.4274 | 0.7648 |

- 兩種集成方法都明顯優於單一決策樹。
- Bagging 主要降低變異 (Variance)；Boosting 依序修正殘差，主要降低偏差 (Bias)。
- 最重要的變數為職業生涯累積表現：Hitcr、Batcr、Runcr、Rbcr。
