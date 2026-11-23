## 原始碼來源：課堂範例 XGBoost_Test.r（已改為 macOS / Rscript 可直接執行：移除 Windows 路徑與 windows()，圖形輸出至 output/）
dir.create("output", showWarnings = FALSE)
pdf("output/XGBoost_Test_plots.pdf", width = 9, height = 7)
##################### Load Packages (For Analysis) ####################
## Libraries Path

## Install Packages (Run Once)

## Load Packages
library(xgboost)      # For XGBoost
library(caTools)      # For Data Splitting
library(ggplot2)      # For Data Visualization
library(GGally)       # For Scatter Plot Matrix
library(gridExtra)    # For Plot Arrangement
library(mlbench)      # For Boston Housing Data


#############################################################################
## XGBoost
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
## XGBoost Classification
########################################################################

## Training Data and Testing Data
set.seed(123)

split.iris <- sample.split(
  iris$Species,
  SplitRatio = 0.8
)

train.iris <- subset(iris, split.iris == TRUE)
test.iris <- subset(iris, split.iris == FALSE)

dim(train.iris)
dim(test.iris)

## Convert Data to Matrix
x.train.iris <- as.matrix(train.iris[, 1:4])
x.test.iris <- as.matrix(test.iris[, 1:4])

## Convert Species to Numeric Labels (0, 1, 2)
y.train.iris <- as.numeric(train.iris$Species) - 1
y.test.iris <- as.numeric(test.iris$Species) - 1

## Create DMatrix
dtrain.iris <- xgb.DMatrix(
  data = x.train.iris,
  label = y.train.iris
)

dtest.iris <- xgb.DMatrix(
  data = x.test.iris,
  label = y.test.iris
)

## XGBoost Parameters
params.iris <- list(
  objective = "multi:softprob",
  num_class = 3,
  eval_metric = "mlogloss",
  max_depth = 3,
  eta = 0.1,
  subsample = 1,
  colsample_bytree = 1,
  tree_method = "hist"
)

## XGBoost Classification Model
set.seed(123)

xgb.iris <- xgb.train(
  params = params.iris,
  data = dtrain.iris,
  nrounds = 100,
  evals = list(
    train = dtrain.iris,
    test = dtest.iris
  ),
  verbose = 0
)

## Model Summary
print(xgb.iris)

## Variable Importance
importance.iris <- xgb.importance(
  model = xgb.iris
)

print(importance.iris)

xgb.plot.importance(
  importance_matrix = importance.iris,
  main = "Variable Importance: Iris"
)

## Predicted Class Probabilities
prob.raw.iris <- predict(
  xgb.iris,
  newdata = dtest.iris
)

## Convert Probabilities to Matrix
## Compatible with vector or matrix predictions
if (is.matrix(prob.raw.iris)) {
  prob.rf.iris <- prob.raw.iris
} else {
  prob.rf.iris <- matrix(
    prob.raw.iris,
    ncol = 3,
    byrow = TRUE
  )
}

colnames(prob.rf.iris) <- levels(iris$Species)

head(prob.rf.iris)

## Predicted Classes
pred.class.iris <- max.col(prob.rf.iris) - 1

pred.xgb.iris <- factor(
  pred.class.iris,
  levels = 0:2,
  labels = levels(iris$Species)
)

## Confusion Matrix
table(
  Predicted = pred.xgb.iris,
  Actual = test.iris$Species
)

## Classification Accuracy
accuracy.xgb.iris <- mean(
  pred.xgb.iris == test.iris$Species
)
accuracy.xgb.iris

## Classification Error Rate
error.xgb.iris <- 1 - accuracy.xgb.iris
error.xgb.iris

## Prediction Results
data.frame(
  Actual = test.iris$Species,
  Predicted = pred.xgb.iris,
  prob.rf.iris
)


#############################################################################
## XGBoost
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

plot(
  pred.lm.fit,
  test_data$medv,
  xlim = c(0, 50),
  ylim = c(0, 50),
  xlab = "Predicted Values",
  ylab = "Observed Values",
  main = "Multiple Linear Regression",
  pch = 19
)

abline(a = 0, b = 1, col = "red", lty = 2)


########################################################################
## XGBoost Regression
########################################################################

## Convert Data to Numeric Matrix
## model.matrix handles factor predictors
x.train <- model.matrix(
  medv ~ . - 1,
  data = train_data
)

x.test <- model.matrix(
  medv ~ . - 1,
  data = test_data
)

## Create DMatrix
dtrain <- xgb.DMatrix(
  data = x.train,
  label = train_data$medv
)

dtest <- xgb.DMatrix(
  data = x.test,
  label = test_data$medv
)

## XGBoost Parameters
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 3,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  tree_method = "hist"
)

## XGBoost Regression Model
set.seed(123)

xgb.fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  evals = list(
    train = dtrain,
    test = dtest
  ),
  verbose = 0
)

## Model Summary
print(xgb.fit)


## Variable Importance
importance.reg <- xgb.importance(
  model = xgb.fit
)

print(importance.reg)

xgb.plot.importance(
  importance_matrix = importance.reg,
  main = "Variable Importance: Boston Housing"
)

## Prediction
pred.xgb.fit <- predict(
  xgb.fit,
  newdata = dtest
)

## Mean Squared Error
mse.xgb.fit <- mean(
  (pred.xgb.fit - test_data$medv)^2
)
mse.xgb.fit

## Root Mean Squared Error
rmse.xgb.fit <- sqrt(mse.xgb.fit)
rmse.xgb.fit

## Predicted vs. Observed Values
data.frame(
  pred.medv = pred.xgb.fit,
  true.medv = test_data$medv
)

plot(
  pred.xgb.fit,
  test_data$medv,
  xlim = c(0, 50),
  ylim = c(0, 50),
  xlab = "Predicted Values",
  ylab = "Observed Values",
  main = "XGBoost Regression",
  pch = 19
)

abline(a = 0, b = 1, col = "red", lty = 2)


########################################################################
## Model Comparison
## Multiple Linear Regression vs. XGBoost
########################################################################

## MSE and RMSE Comparison
results <- data.frame(
  Model = c("Linear Regression", "XGBoost"),
  MSE = c(mse.lm.fit, mse.xgb.fit),
  RMSE = c(rmse.lm.fit, rmse.xgb.fit)
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
  geom_abline(
    intercept = 0,
    slope = 1,
    color = "red",
    linetype = 2
  ) +
  coord_fixed(xlim = c(0, 50), ylim = c(0, 50)) +
  labs(title = "Linear Regression") +
  theme_bw()

p2 <- ggplot(
  data.frame(
    Predicted = pred.xgb.fit,
    Observed = test_data$medv
  ),
  aes(x = Predicted, y = Observed)
) +
  geom_point(color = "darkgreen") +
  geom_abline(
    intercept = 0,
    slope = 1,
    color = "red",
    linetype = 2
  ) +
  coord_fixed(xlim = c(0, 50), ylim = c(0, 50)) +
  labs(title = "XGBoost") +
  theme_bw()

grid.arrange(p1, p2, ncol = 2)


########################################################################
## Effect of max_depth on XGBoost
########################################################################

depth.values <- 1:10
validation.rmse <- numeric(length(depth.values))

## Create validation data from training set
set.seed(123)
val.index <- sample(
  seq_len(nrow(x.train)),
  size = floor(0.2 * nrow(x.train))
)

dtrain.inner <- xgb.DMatrix(
  data = x.train[-val.index, , drop = FALSE],
  label = train_data$medv[-val.index]
)

dvalid <- xgb.DMatrix(
  data = x.train[val.index, , drop = FALSE],
  label = train_data$medv[val.index]
)

for (i in seq_along(depth.values)) {

  params.temp <- params
  params.temp$max_depth <- depth.values[i]

  set.seed(123)

  fit.temp <- xgb.train(
    params = params.temp,
    data = dtrain.inner,
    nrounds = 200,
    verbose = 0
  )

  pred.temp <- predict(
    fit.temp,
    newdata = dvalid
  )

  validation.rmse[i] <- sqrt(mean(
    (pred.temp - train_data$medv[val.index])^2
  ))
}

plot(
  depth.values,
  validation.rmse,
  type = "b",
  pch = 19,
  xlab = "Maximum Tree Depth",
  ylab = "Validation RMSE",
  main = "Effect of max_depth on XGBoost"
)


########################################################################
## Important Parameters in XGBoost
########################################################################

# nrounds = 200         Number of boosting iterations
# max_depth = 3         Maximum depth of each tree
# eta = 0.1             Learning rate
# subsample = 0.8       Row sampling ratio
# colsample_bytree = 0.8 Column sampling ratio
# objective             Loss function / learning objective
# eval_metric           Evaluation metric
# tree_method = "hist"  Histogram-based tree algorithm
#
# xgb.importance()      Variable importance
# xgb.plot.importance() Variable importance plot
# predict()             Prediction
########################################################################

dev.off()
