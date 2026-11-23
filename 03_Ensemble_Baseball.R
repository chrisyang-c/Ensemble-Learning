#############################################################################
## 集成學習 (Ensemble Learning) 作業
## 資料：Baseball (ball.txt) —— 預測 1987 年球員年薪對數 Logsalary
## 比較：單一迴歸樹 (CART) vs 隨機森林 (Bagging) vs XGBoost (Boosting)
## 執行方式：在本資料夾下執行  Rscript 03_Ensemble_Baseball.R
#############################################################################

## Load Packages
library(rpart)        # CART
library(rpart.plot)   # Tree plot
library(randomForest) # Random Forest
library(xgboost)      # XGBoost
library(caTools)      # Data splitting
library(ggplot2)      # Visualization
library(gridExtra)    # Plot arrangement

dir.create("output", showWarnings = FALSE)
pdf("output/Ensemble_Baseball_plots.pdf", width = 9, height = 7)

#############################################################################
## 1. 資料讀取與整理
#############################################################################
Baseball <- read.table("data/ball.txt", header = TRUE, sep = "",
                       quote = "\"", dec = ".", stringsAsFactors = TRUE)
dim(Baseball)
str(Baseball)

## 移除：Id、Name（識別碼）、Salary（= exp(Logsalary)，會洩漏答案）
Baseball <- subset(Baseball, select = -c(Id, Name, Salary))
Baseball <- na.omit(Baseball)
dim(Baseball)
summary(Baseball$Logsalary)

#############################################################################
## 2. 切分訓練 / 測試資料 (80% / 20%)
#############################################################################
set.seed(123)
split <- sample.split(Baseball$Logsalary, SplitRatio = 0.8)
train_data <- subset(Baseball, split == TRUE)
test_data  <- subset(Baseball, split == FALSE)
dim(train_data)
dim(test_data)

mse  <- function(pred, y) mean((pred - y)^2)
rmse <- function(pred, y) sqrt(mse(pred, y))
y.test <- test_data$Logsalary

#############################################################################
## 3. 單一迴歸樹 (CART) —— 基準模型
#############################################################################
set.seed(123)
tree.fit <- rpart(Logsalary ~ ., method = "anova", data = train_data)
rpart.plot(tree.fit, extra = 101, main = "Regression Tree: Logsalary")
tree.fit$variable.importance

pred.tree <- predict(tree.fit, test_data)
mse.tree  <- mse(pred.tree, y.test)
mse.tree

#############################################################################
## 4. 隨機森林 (Bagging：Bootstrap 抽樣 + 隨機挑特徵 + 平均)
#############################################################################
p <- ncol(train_data) - 1          # 預測變數個數
set.seed(123)
rf.fit <- randomForest(Logsalary ~ ., data = train_data,
                       ntree = 500,
                       mtry = max(1, floor(p / 3)),   # 迴歸預設 p/3
                       nodesize = 5,
                       importance = TRUE)
print(rf.fit)

## OOB (袋外資料) MSE 隨樹的數量下降
plot(rf.fit, main = "Random Forest: OOB MSE vs Number of Trees")

## 變數重要度
importance(rf.fit)
varImpPlot(rf.fit, main = "Random Forest Variable Importance: Baseball")

pred.rf <- predict(rf.fit, newdata = test_data)
mse.rf  <- mse(pred.rf, y.test)
mse.rf

## mtry 對 OOB MSE 的影響
mtry.values <- 1:p
oob.mse <- numeric(length(mtry.values))
for (i in seq_along(mtry.values)) {
  set.seed(123)
  fit <- randomForest(Logsalary ~ ., data = train_data,
                      ntree = 500, mtry = mtry.values[i])
  oob.mse[i] <- tail(fit$mse, 1)
}
plot(mtry.values, oob.mse, type = "b", pch = 19,
     xlab = "Number of Variables (mtry)", ylab = "OOB MSE",
     main = "Effect of mtry on Random Forest")
best.mtry <- mtry.values[which.min(oob.mse)]
best.mtry

#############################################################################
## 5. XGBoost (Boosting：每棵樹修正前一棵的殘差)
#############################################################################
## 類別變數轉 dummy；在全資料上建矩陣以確保訓練/測試欄位一致
X.all  <- model.matrix(Logsalary ~ . - 1, data = Baseball)
x.train <- X.all[split == TRUE, ]
x.test  <- X.all[split == FALSE, ]

dtrain <- xgb.DMatrix(data = x.train, label = train_data$Logsalary)
dtest  <- xgb.DMatrix(data = x.test,  label = y.test)

params <- list(objective = "reg:squarederror",
               eval_metric = "rmse",
               max_depth = 3,
               eta = 0.05,               # 學習率 < 1：一次只補一小步
               subsample = 0.8,
               colsample_bytree = 0.8,
               tree_method = "hist")

## 以交叉驗證選擇樹的數量 (避免過度擬合)
set.seed(123)
cv <- xgb.cv(params = params, data = dtrain, nrounds = 1000,
             nfold = 5, early_stopping_rounds = 30, verbose = 0)
cv.log <- as.data.frame(cv$evaluation_log)
best.nrounds <- which.min(cv.log$test_rmse_mean)
best.nrounds

set.seed(123)
xgb.fit <- xgb.train(params = params, data = dtrain,
                     nrounds = best.nrounds,
                     evals = list(train = dtrain, test = dtest),
                     verbose = 0)
print(xgb.fit)

importance.xgb <- xgb.importance(model = xgb.fit)
print(importance.xgb)
xgb.plot.importance(importance_matrix = importance.xgb, top_n = 15,
                    main = "XGBoost Variable Importance: Baseball")

## 訓練 / 測試 RMSE 隨 boosting 回合變化
ev <- as.data.frame(attributes(xgb.fit)$evaluation_log)
if (nrow(ev) > 0) {
  matplot(ev$iter, ev[, c("train_rmse", "test_rmse")], type = "l", lty = 1,
          col = c("blue", "red"), xlab = "Boosting Rounds", ylab = "RMSE",
          main = "XGBoost: Train vs Test RMSE")
  legend("topright", c("train", "test"), col = c("blue", "red"), lty = 1)
}

pred.xgb <- predict(xgb.fit, newdata = dtest)
mse.xgb  <- mse(pred.xgb, y.test)
mse.xgb

#############################################################################
## 6. 模型比較
#############################################################################
results <- data.frame(
  Model = c("Regression Tree (CART)", "Random Forest (Bagging)", "XGBoost (Boosting)"),
  MSE   = c(mse.tree, mse.rf, mse.xgb),
  RMSE  = sqrt(c(mse.tree, mse.rf, mse.xgb)),
  R2    = 1 - c(mse.tree, mse.rf, mse.xgb) / mean((y.test - mean(y.test))^2)
)
results$MSE  <- round(results$MSE, 4)
results$RMSE <- round(results$RMSE, 4)
results$R2   <- round(results$R2, 4)
print(results)
write.csv(results, "output/model_comparison.csv", row.names = FALSE)

## 預測值 vs 實際值
pv <- function(pred, title, col) {
  ggplot(data.frame(Predicted = pred, Observed = y.test),
         aes(x = Predicted, y = Observed)) +
    geom_point(color = col) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = 2) +
    coord_fixed(xlim = c(4, 8), ylim = c(4, 8)) +
    labs(title = title) + theme_bw()
}
grid.arrange(pv(pred.tree, "Regression Tree", "gray30"),
             pv(pred.rf,   "Random Forest",   "darkgreen"),
             pv(pred.xgb,  "XGBoost",         "blue"),
             ncol = 3)

dev.off()

#############################################################################
## 結論 (set.seed(123) 之執行結果，測試集)
##   Model                     MSE     RMSE    R2
##   Regression Tree (CART)    0.2325  0.4822  0.7007
##   Random Forest (Bagging)   0.1778  0.4217  0.7711
##   XGBoost (Boosting)        0.1827  0.4274  0.7648
## - 單一迴歸樹結構簡單、可解釋，但變異大、測試誤差最高。
## - 隨機森林以 Bootstrap + 隨機特徵 + 平均，降低變異 (Variance)，MSE 下降約 24%。
## - XGBoost 以殘差接力修正 + 學習率 + 正則化，降低偏差 (Bias)；以 CV 決定樹的數量
##   以避免過度擬合，表現與隨機森林相近 (本資料僅約 260 筆，Bagging 已足夠穩定)。
## - 變數重要度：職業生涯累積表現 (Hitcr, Batcr, Runcr, Rbcr) 最能解釋年薪。
#############################################################################
