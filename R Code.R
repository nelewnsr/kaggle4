rm(list = ls())

library(dplyr)
library(tidyverse)
library(jpeg)
library(gridExtra)
library(randomForest)
library(glmnet)
require(rpart)
require(FNN)
library(OpenImageR)
library(gbm)

setwd("/Users/Nele/Desktop/Big Data Analytics/Competition 4")


skies = dir("/Users/Nele/Desktop/Big Data Analytics/Competition 4/images_split/cloudy_sky/", full.names = TRUE)
rivers = dir("/Users/Nele/Desktop/Big Data Analytics/Competition 4/images_split/rivers/", full.names = TRUE)
sunsets = dir("/Users/Nele/Desktop/Big Data Analytics/Competition 4/images_split/sunsets/", full.names = TRUE)
trees = dir("/Users/Nele/Desktop/Big Data Analytics/Competition 4/images_split/trees_and_forest/", full.names = TRUE)
test_set = dir("/Users/Nele/Desktop/Big Data Analytics/Competition 4/images_split/test_set/", full.names = TRUE)


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


Rivers  = map_df(rivers, readJPEG_as_df) %>% mutate(category = "rivers")
Sunsets = map_df(sunsets, readJPEG_as_df) %>% mutate(category = "sunsets")
Skies   = map_df(skies, readJPEG_as_df) %>% mutate(category = "cloudy_sky")
Trees   = map_df(trees, readJPEG_as_df) %>% mutate(category = "trees_and_forest")

# histograms over 10 images from each class
try <- bind_rows(Rivers, Sunsets, Skies, Trees) %>% 
  ggplot(aes(pixel_value, fill=color)) + geom_histogram(bins=30, alpha=0.5) + facet_wrap(~category)

############## Feature Extraction ##############

nr = nc = 3 #verhogen 7
myFeatures  <- . %>% # starting with '.' defines the pipe to be a function 
  group_by(file, X=cut(x, nr, labels = FALSE)-1, Y=cut(y, nc, labels=FALSE)-1, color) %>%
  summarise(
    m = mean(pixel_value),
    s = sd(pixel_value),
    min = min(pixel_value),
    max = max(pixel_value),
    q25 = quantile(pixel_value, .25),
    q75 = quantile(pixel_value, .75),
    skew = e1071::skewness(pixel_value),
    lag1.1 = cor(pixel_value, lag(pixel_value), use = "pairwise"),
    lag1.2 = cor(pixel_value, lag(pixel_value, n = 2), use = "pairwise")
  )

  
  
# mutate(cluster = kmeans(pixel_value, 5)$cluster) 


example <- readJPEG_as_df(skies[4], myFeatures) %>% head(10)
names(example)


# because we need to reshape from long to wide format multiple times lets define a function:
myImgDFReshape = . %>%
  gather(feature, value, -file, -X, -Y, -color) %>% 
  unite(feature, color, X, Y, feature) %>% 
  spread(feature, value)

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


############## Modelling ##############


# Classification Tree -----------------------------------------------------

fittree = rpart(factor(category) ~ . , Train[, !colnames(Train) %in% c("file")])
opttree = prune(fittree, cp = 0.033333)


# Random Forest -----------------------------------------------------------

ranfor = randomForest(factor(category) ~ . - file, Train, importance =TRUE)


# Boosting ----------------------------------------------------------------

boost.boston=gbm(factor(category) ~ ., TrainTrain[, !colnames(Train) %in% c("file")], distribution = "multinomial", 
                 n.trees = 5000, interaction.depth = 4)


############## Preparing for Submission ##############

Test = map_df(test_set, readJPEG_as_df, featureExtractor = myFeatures) %>% myImgDFReshape
Test %>% ungroup %>% transmute(file=file, category = predict(ranfor, ., type = "class")) %>% write_csv("my_submission.csv")
file.show("my_submission.csv")

Test = map_df(test_set, readJPEG_as_df, featureExtractor = myFeatures) %>% myImgDFReshape
Test %>% ungroup %>% transmute(file=file, category = predict(boost.boston, ., n.trees=5000, response = "class")) %>% write_csv("my_submission.csv")
file.show("my_submission.csv")

predict(boost.boston,newdata=Test, n.trees=5000)

Test = map_df(test_set, readJPEG_as_df, featureExtractor = myFeatures) %>% myImgDFReshape
random = Test %>% ungroup %>% transmute(file=file, category = predict(ranfor, ., type = "class"))
count_random_60 = count(random, var = category)

Test = map_df(test_set, readJPEG_as_df, featureExtractor = myFeatures) %>% myImgDFReshape
tree = Test %>% ungroup %>% transmute(file=file, category = predict(fittree, ., type = "class"))
count(tree, var = category)

count_random_60
count_random_100
count_random_150
count_random_1050

#neural net
#(extreme) boosting
#xgboost