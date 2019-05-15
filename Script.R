library(tidyverse)
library(jpeg)
library(e1071)
library(caret)

# Data import -------------------------------------------------------------

skies = dir("bda-18-19-sem-2-scene-recognition/images_split/cloudy_sky/", full.names = TRUE)
rivers = dir("bda-18-19-sem-2-scene-recognition/images_split/rivers/", full.names = TRUE)
sunsets = dir("bda-18-19-sem-2-scene-recognition/images_split/sunsets/", full.names = TRUE)
trees = dir("bda-18-19-sem-2-scene-recognition/images_split/trees_and_forest/", full.names = TRUE)
test_set = dir("bda-18-19-sem-2-scene-recognition/images_split/test_set/", full.names = TRUE)


# Demonstration -----------------------------------------------------------

img = readJPEG(rivers[4])
glimpse(img)

paintImage = function(img,..., colors=1:3){img[,,-colors]=0; rasterImage(img,...)}

# set up the canvas
plot.new()
plot.window(c(0,2), c(0,2))

# paint the images in separate color channels
paintImage(img, 0, 1, 1, 2) # original image
paintImage(img, 1, 1, 2, 2, colors=1) # red channel
paintImage(img, 0, 0, 1, 1, colors=2) # green channel
paintImage(img, 1, 0, 2, 1, colors=3) # blue channel

# set up the canvas
plot.new()
plot.window(c(0,2), c(0,2))

# paint the images in separate color channels
paintImage(img, 0, 1, 1, 2) # original image
paintImage(img, 1, 1, 2, 2, colors=1:2) # red + green channel = yellow
paintImage(img, 0, 0, 1, 1, colors=2:3) # green + blue channel = cyan
paintImage(img, 1, 0, 2, 1, colors=c(1,3)) # red + blue channel = magenta


# Setting up color data ---------------------------------------------------

# function
readJPEG_as_df <- function(path, featureExtractor = I) {
  img = readJPEG(path)
  # image dimensions
  d = dim(img) 
  # add names to the array dimensions
  dimnames(img) = list(x = 1:d[1], y = 1:d[2], color = c('r','g','b')) 
  # turn array into a data frame 
  df  <- 
    as.table(img) %>% 
    as.data.frame(stringsAsFactors = FALSE) %>% 
    # make the final format handier and add file name to the data frame
    mutate(file = basename(path), x = as.numeric(x)-1, y = as.numeric(y)-1) %>%
    mutate(pixel_id = x + 28 * y) %>% 
    rename(pixel_value = Freq) %>%
    select(file, pixel_id, x, y, color, pixel_value)
  # extract features 
  df %>%
    featureExtractor
}

# check function
readJPEG_as_df(sunsets[1]) %>% head()

# histogram function
peekImage = . %>% spread(color, pixel_value) %>%  mutate(x=rev(x), color = rgb(r,g,b)) %>%  
  {ggplot(., aes(y, x, fill = color)) + geom_tile(show.legend = FALSE) + theme_light() + 
    scale_fill_manual(values=levels(as.factor(.$color))) + facet_wrap(~ file)}

readJPEG_as_df(trees[22]) %>% peekImage
readJPEG_as_df(sunsets[18]) %>% peekImage

readJPEG_as_df(sunsets[18]) %>% ggplot(aes(pixel_value, fill=color)) + geom_density(alpha=0.5)

# compare over categories

bind_rows(
  readJPEG_as_df(rivers[2]) %>% mutate(category = "rivers"), 
  readJPEG_as_df(sunsets[10]) %>% mutate(category = "sunsets"), 
  readJPEG_as_df(trees[122]) %>% mutate(category = "trees_and_forest"), 
  readJPEG_as_df(skies[20]) %>% mutate(category = "cloudy_sky")
) %>% 
  ggplot(aes(pixel_value, fill=color)) + geom_density(alpha=0.5, col=NA) + facet_wrap(~ category)


# Compare histograms ------------------------------------------------------

# load first 10 images from each category
Rivers  = map_df(rivers[1:10], readJPEG_as_df) %>% mutate(category = "rivers")
Sunsets = map_df(sunsets[1:10], readJPEG_as_df) %>% mutate(category = "sunsets")
Skies   = map_df(skies[1:10], readJPEG_as_df) %>% mutate(category = "cloudy_sky")
Trees   = map_df(trees[1:10], readJPEG_as_df) %>% mutate(category = "trees_and_forest")

# histograms over 10 images from each class
bind_rows(Rivers, Sunsets, Skies, Trees) %>% 
  ggplot(aes(pixel_value, fill=color)) + geom_histogram(bins=30, alpha=0.5) + facet_wrap(~category)


# Simple statistical features ---------------------------------------------

# exploiting the rule of 3rds

nr = nc = 6
myFeatures  <- . %>% # starting with '.' defines the pipe to be a function 
  group_by(file, X=cut(x, nr, labels = FALSE)-1, Y=cut(y, nc, labels=FALSE)-1, color) %>%
  summarise(
    m = mean(pixel_value),
    s = sd(pixel_value),
    min = min(pixel_value),
    max = max(pixel_value),
    q25 = quantile(pixel_value, .25),
    q75 = quantile(pixel_value, .75),
    skew = e1071::skewness(pixel_value)
  ) 


# Complete data import ----------------------------------------------------

# wide format reshape function
myImgDFReshape = . %>%
  gather(feature, value, -file, -X, -Y, -color) %>% 
  unite(feature, color, X, Y, feature) %>% 
  spread(feature, value)

# import
Sunsets = map_df(sunsets, readJPEG_as_df, featureExtractor = myFeatures) %>% 
  myImgDFReshape %>%
  mutate(category = "sunsets")
Trees = map_df(trees, readJPEG_as_df, featureExtractor = myFeatures) %>% 
  myImgDFReshape %>%
  mutate(category = "trees_and_forest")
Rivers = map_df(rivers, readJPEG_as_df, featureExtractor = myFeatures) %>% 
  myImgDFReshape %>%
  mutate(category = "rivers")
Skies = map_df(skies, readJPEG_as_df, featureExtractor = myFeatures) %>% 
  myImgDFReshape %>%
  mutate(category = "cloudy_sky")

Train = bind_rows(Sunsets, Trees, Rivers, Skies) 
head(Train)


# Modelling ---------------------------------------------------------------

# install newest version of xgBoost
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
require(xgboost)

### Preprocessing 
## prepare outcome variable
# store labels
labels <- tibble(category = levels(factor(Train$category)), 
                 label = 0:3)

# make category numeric and subtract so they start with 0
train_data <- Train %>% ungroup
train_data$category <- as.numeric(factor(train_data$category))-1

## set up training and test set
#Create training set
train <- train_data %>% sample_frac(.75)
#Create test set
test  <- anti_join(train_data, train, by = 'file')

# extract outcome variable
train_label <- train$category
test_label <- test$category

# make a numeric matrix 
train <- train %>% select(-file, -category) %>% as.matrix()
test <- test %>% select(-file, -category) %>% as.matrix()

# set up xgb matrices
train_matrix <- xgb.DMatrix(data = train, label = train_label)
test_matrix <- xgb.DMatrix(data = test, label = test_label)

### build model
# set up parameters
numberOfClasses <- length(unique(train_data$category))

xgb_params <- list("objective" = "multi:softmax",
                   "num_class" = numberOfClasses)

nround <- 100 # number of XGBoost rounds
cv.nfold <- 5


# Default model -----------------------------------------------------------

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE,
                   showsd = T, stratified = T) #, print_every_n = 10, early_stopping_rounds = 10, maximize = F)

# best iteration: 60
nround <- 40

# make prediction table
cv_pred <- tibble(pred = cv_model$pred[,1], cat = train_label)

# confusion matrix
confusionMatrix(factor(cv_pred$pred),
                factor(cv_pred$cat),
                mode = "everything")

### fit full model
full_label <- train_data$category
train_data <- train_data %>% select(-file, -category) %>% as.matrix()
full_matrix <- xgb.DMatrix(data = train_data, label = full_label)

best_model <- xgb.train(params = xgb_params,
                        data = full_matrix,
                        nrounds = nround)

# model prediction on hold-back test set
best_pred <- predict(best_model, test_matrix)

confusionMatrix(factor(best_pred), factor(test_label))

# variable importance plot
mat <- xgb.importance(feature_names = colnames(train), model = best_model)
xgb.plot.importance(importance_matrix = mat[1:20]) 


# Test set predictions submission -----------------------------------------
# import test set
Test <- map_df(test_set, readJPEG_as_df, featureExtractor = myFeatures) %>% myImgDFReshape

# make matrix for predictions
Test_mat <- Test %>% ungroup() %>% select(-file) %>% as.matrix()

# make predictions
pred_fin <- predict(best_model, Test_mat)

# recode predictions with labels
library(qdap)
predictions <- lookup(pred_fin, labels$label, labels$category)

# make final data frame
Test %>% ungroup %>% transmute(file=file, category = predictions) %>% write_csv("my_submission3.csv")

