## 原始碼來源：課堂範例 RF_Test.r（已改為 macOS / Rscript 可直接執行：移除 Windows 路徑與 windows()，圖形輸出至 output/）
dir.create("output", showWarnings = FALSE)
pdf("output/RF_Test_plots.pdf", width = 9, height = 7)
##################### Load Packages (For Analysis) ####################
## Libraries Path

## Install Packages (Run Once)
#install.packages(c("randomForest", "caTools", "ggplot2","GGally", "gridExtra", "mlbench"))

## Load Packages
library(randomForest) # For Random Forest
library(caTools)      # For Data Splitting
library(ggplot2)      # For Data Visualization
library(GGally)       # For Scatter Plot Matrix
library(gridExtra)    # For Plot Arrangement
library(mlbench)      # For Boston Housing Data


#############################################################################
## Random Forest
## Iris Data (For Classification)
#############################################################################

## Descriptive Statistics
data("iris")
dim(iris)

summary(iris)
str(iris)
head(iris, n = 10)

## Boxplot
boxplot(iris[, -5],
        col = c("red", "blue", "green", "yellow"),
        main = "Boxplot of Iris Data")

## Scatter Plot Matrix
ggpairs(iris,
        columns = 1:4,
        aes(color = Species, alpha = 0.7)) +
        theme_bw() +
        theme(legend.position = "right")

## Boxplots by Species
p1 <- ggplot(iris, aes(x = Species, y = Sepal.Length)) +
  geom_boxplot(aes(fill = Species)) +
  labs(title = "Sepal Length") +
  theme_bw() +
  theme(legend.position = "none")

p2 <- ggplot(iris, aes(x = Species, y = Sepal.Width)) +
  geom_boxplot(aes(fill = Species)) +
  labs(title = "Sepal Width") +
  theme_bw() +
  theme(legend.position = "none")

p3 <- ggplot(iris, aes(x = Species, y = Petal.Length)) +
  geom_boxplot(aes(fill = Species)) +
  labs(title = "Petal Length") +
  theme_bw() +
  theme(legend.position = "none")

p4 <- ggplot(iris, aes(x = Species, y = Petal.Width)) +
  geom_boxplot(aes(fill = Species)) +
  labs(title = "Petal Width") +
  theme_bw() +
  theme(legend.position = "none")

grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)


########################################################################
## Random Forest Classification
########################################################################

## Training Data and Testing Data
set.seed(123)

split.iris <- sample.split(iris$Species, SplitRatio = 0.8)

train.iris <- subset(iris, split.iris == TRUE)
test.iris <- subset(iris, split.iris == FALSE)

dim(train.iris)
dim(test.iris)

## Random Forest Classification Model
set.seed(123)

rf.iris <- randomForest(
  Species ~ .,
  data = train.iris,
  ntree = 500,
  mtry = 2,
  importance = TRUE
)

## Model Summary
print(rf.iris)
summary(rf.iris)

## Out-of-Bag (OOB) Error Rate
rf.iris$err.rate

## OOB Error Plot
plot(rf.iris,
     main = "Random Forest Classification: OOB Error")

## Variable Importance
importance(rf.iris)

varImpPlot(rf.iris,
           main = "Variable Importance: Iris")

## Prediction
pred.rf.iris <- predict(rf.iris, newdata = test.iris)

## Confusion Matrix
table(Predicted = pred.rf.iris,
      Actual = test.iris$Species)

## Classification Accuracy
accuracy.rf.iris <- mean(pred.rf.iris == test.iris$Species)
accuracy.rf.iris

## Classification Error Rate
error.rf.iris <- 1 - accuracy.rf.iris
error.rf.iris

## Predicted Class Probabilities
prob.rf.iris <- predict(
  rf.iris,
  newdata = test.iris,
  type = "prob"
)

head(prob.rf.iris)


#############################################################################
## Random Forest
## Boston Housing Data (For Regression)
#############################################################################

## Load Data
data("BostonHousing", package = "mlbench")

dim(BostonHousing)
summary(BostonHousing)
str(BostonHousing)
head(BostonHousing, n = 10)


########################################################################
## Training Data and Testing Data
########################################################################

set.seed(123)

split <- sample.split(
  BostonHousing$medv,
  SplitRatio = 0.8
)

train_data <- subset(BostonHousing, split == TRUE)
test_data <- subset(BostonHousing, split == FALSE)

dim(train_data)
dim(test_data)


########################################################################
## Multiple Linear Regression
########################################################################

lm.fit <- glm(
  medv ~ .,
  family = gaussian(link = "identity"),
  data = train_data
)

summary(lm.fit)

## Prediction
pred.lm.fit <- predict(
  lm.fit,
  newdata = test_data
)

## Mean Squared Error
mse.lm.fit <- mean(
  (pred.lm.fit - test_data$medv)^2
)
mse.lm.fit

## Root Mean Squared Error
rmse.lm.fit <- sqrt(mse.lm.fit)
rmse.lm.fit

## Predicted vs. Observed Values
data.frame(
  pred.medv = pred.lm.fit,
  true.medv = test_data$medv
)

plot(pred.lm.fit, test_data$medv,
     xlim = c(0, 50),
     ylim = c(0, 50),
     xlab = "Predicted Values",
     ylab = "Observed Values",
     main = "Multiple Linear Regression",
     pch = 19)

abline(a = 0, b = 1, col = "red", lty = 2)


########################################################################
## Random Forest Regression
########################################################################

## Random Forest Regression Model
set.seed(123)

rf.fit <- randomForest(
  medv ~ .,
  data = train_data,
  ntree = 500,
  mtry = 4,
  nodesize = 5,
  importance = TRUE
)

## Model Summary
print(rf.fit)
summary(rf.fit)

## OOB Mean Squared Error
rf.fit$mse

## OOB Error Plot
plot(rf.fit,
     main = "Random Forest Regression: OOB MSE")

## Variable Importance
importance(rf.fit)

varImpPlot(rf.fit,
           main = "Variable Importance: Boston Housing")

## Prediction
pred.rf.fit <- predict(
  rf.fit,
  newdata = test_data
)

## Mean Squared Error
mse.rf.fit <- mean(
  (pred.rf.fit - test_data$medv)^2
)
mse.rf.fit

## Root Mean Squared Error
rmse.rf.fit <- sqrt(mse.rf.fit)
rmse.rf.fit

## Predicted vs. Observed Values
data.frame(
  pred.medv = pred.rf.fit,
  true.medv = test_data$medv
)

plot(pred.rf.fit, test_data$medv,
     xlim = c(0, 50),
     ylim = c(0, 50),
     xlab = "Predicted Values",
     ylab = "Observed Values",
     main = "Random Forest Regression",
     pch = 19)

abline(a = 0, b = 1, col = "red", lty = 2)


########################################################################
## Model Comparison
## Multiple Linear Regression vs. Random Forest
########################################################################

## MSE and RMSE Comparison
results <- data.frame(
  Model = c("Linear Regression", "Random Forest"),
  MSE = c(mse.lm.fit, mse.rf.fit),
  RMSE = c(rmse.lm.fit, rmse.rf.fit)
)

print(results)

## Visualization of Prediction Results
p1 <- ggplot(
  data.frame(
    Predicted = pred.lm.fit,
    Observed = test_data$medv
  ),
  aes(x = Predicted, y = Observed)
) +
  geom_point(color = "blue") +
  geom_abline(intercept = 0, slope = 1,
              color = "red", linetype = 2) +
  coord_fixed(xlim = c(0, 50), ylim = c(0, 50)) +
  labs(title = "Linear Regression") +
  theme_bw()

p2 <- ggplot(
  data.frame(
    Predicted = pred.rf.fit,
    Observed = test_data$medv
  ),
  aes(x = Predicted, y = Observed)
) +
  geom_point(color = "darkgreen") +
  geom_abline(intercept = 0, slope = 1,
              color = "red", linetype = 2) +
  coord_fixed(xlim = c(0, 50), ylim = c(0, 50)) +
  labs(title = "Random Forest") +
  theme_bw()

grid.arrange(p1, p2, ncol = 2)

########################################################################
## Effect of mtry on Random Forest
########################################################################

mtry.values <- 1:13
oob.mse <- numeric(length(mtry.values))

for (i in seq_along(mtry.values)) {

  set.seed(123)

  fit <- randomForest(
    medv ~ .,
    data = train_data,
    ntree = 500,
    mtry = mtry.values[i]
  )

  oob.mse[i] <- tail(fit$mse, 1)
}

plot(mtry.values, oob.mse,
     type = "b",
     pch = 19,
     xlab = "Number of Variables (mtry)",
     ylab = "OOB Mean Squared Error",
     main = "Effect of mtry on Random Forest")

########################################################################
## 隨機森林的重要參數
# ntree = 500         建立 500 棵決策樹
# mtry = 2            每次節點分裂隨機選取 2 個候選變數
# nodesize = 5        終端節點的最小樣本數設定
# importance = TRUE   計算變數重要性
# rf.fit$mse          各棵樹累積後的 OOB MSE
# varImpPlot()        繪製變數重要性圖
# predict()           預測測試資料
# mtry 預設值         分類 為 sqrt(p)，迴歸為 max{1,floor(p/3)}。
########################################################################

dev.off()
