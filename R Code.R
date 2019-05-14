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

readJPEG_as_df(sunsets[1]) %>% head()

#peekImage = . %>% spread(color, pixel_value) %>%  mutate(x=rev(x), color = rgb(r,g,b)) %>%  {ggplot(., aes(y, x, fill = color)) + geom_tile(show.legend = FALSE) + theme_light() + 
#    scale_fill_manual(values=levels(as.factor(.$color))) + facet_wrap(~ file)}

#readJPEG_as_df(rivers[2]) %>% peekImage
#readJPEG_as_df(sunsets[18]) %>% peekImage

#readJPEG_as_df(sunsets[18]) %>% ggplot(aes(pixel_value, fill=color)) + geom_density(alpha=0.5)

#bind_rows(
#  readJPEG_as_df(rivers[2]) %>% mutate(category = "rivers"), 
#  readJPEG_as_df(sunsets[10]) %>% mutate(category = "sunsets"), 
#  readJPEG_as_df(trees[122]) %>% mutate(category = "trees_and_forest"), 
#  readJPEG_as_df(skies[20]) %>% mutate(category = "cloudy_sky")
#) %>% 
#  ggplot(aes(pixel_value, fill=color)) + geom_density(alpha=0.5, col=NA) + facet_wrap(~ category)

Rivers  = map_df(rivers, readJPEG_as_df) %>% mutate(category = "rivers")
Sunsets = map_df(sunsets, readJPEG_as_df) %>% mutate(category = "sunsets")
Skies   = map_df(skies, readJPEG_as_df) %>% mutate(category = "cloudy_sky")
Trees   = map_df(trees, readJPEG_as_df) %>% mutate(category = "trees_and_forest")

# histograms over 10 images from each class
try <- bind_rows(Rivers, Sunsets, Skies, Trees) %>% 
  ggplot(aes(pixel_value, fill=color)) + geom_histogram(bins=30, alpha=0.5) + facet_wrap(~category)

############## Feature Extraction ##############

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")

#cl1 <- kmeans(Rivers$pixel_value, 5)
#plot(Rivers$pixel_value, col = cl1$cluster)

#cl2 <- kmeans(Sunsets$pixel_value, 5)$cluster
#plot(Sunsets$pixel_value, col = cl2$cluster)

#min(Sunsets$pixel_value)
#min(Rivers$pixel_value)
#min(Skies$pixel_value)

#count(Rivers, vars = pixel_value)
#as.data.frame(table(Rivers$pixel_value))

#count = as.data.frame(table(pixel_value))$Freq)

mean(Sunsets$pixel_value)
k <- kmeans(Sunsets$pixel_value, 5)
k$centers


FOLDER_path = paste0(system.file("tmp_images", "same_type", package = "OpenImageR"), '/')

res = HOG_apply(Sunsets)
res

HOG_apply(Sunsets[1:10,], 3, 3)
Sunsets[2]
Sunsets[1:10,]
#count = sum(pixel_value),
nr = nc = 7 #verhogen 7
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

  
  
mutate(cluster = kmeans(pixel_value, 5)$cluster) 


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

### Classification Tree -------

fittree = rpart(factor(category) ~ . - file, Train)
fittree = rpart(factor(category) ~ . , Train[, !colnames(Train) %in% c("file")])

# various representations: 
## graphical decision tree, 
#plot(fittree, compress=T, uniform=T, margin=0.05, branch=.75); text(fittree, cex=0.6, all=T, use.n=T)
## textual decision tree, 
#fittree 
## data frame (for some reason the 'split' that contains the split value is missing)
#fittree$frame %>% select(-yval2)

# information about the CV of complexity penalty optimum
#printcp(fittree)

opttree = prune(fittree, cp = 0.033333)
#plot(opttree, compress=T, uniform=T, margin=0.05, branch=.75); text(opttree, cex=0.6, all=T, use.n=T)

#predtree = predict(fittree, Train, type='class') # also try opttree in stead of fittree
#table(truth=Train$category, pred = predtree)
#mean(Train$category == predtree) # 80% correct... can human brains do better?

### Random Forest -------

ranfor = randomForest(factor(category) ~ . - file, Train)
#ranfor

#layout(matrix(1:4,2,2,byrow=T))
#plot(ranfor)
#varImpPlot(ranfor,cex=0.5)

#getTree(ranfor, 2, labelVar = T) # unfortunately not easy to plot this with plot.rpart

#predrf = predict(ranfor, Train, type='class')
#table(truth = Train$category, pred = predrf)
#mean(Train$category != predrf) # prediction is essentially perfect... with OOB error this indicates overfitting

############## Preparing for Submission ##############

Test = map_df(test_set, readJPEG_as_df, featureExtractor = myFeatures) %>% myImgDFReshape
Test %>% ungroup %>% transmute(file=file, category = predict(ranfor, ., type = "class")) %>% write_csv("my_submission.csv")
file.show("my_submission.csv")


Test = map_df(test_set, readJPEG_as_df, featureExtractor = myFeatures) %>% myImgDFReshape
random = Test %>% ungroup %>% transmute(file=file, category = predict(ranfor, ., type = "class"))
count_random = count(random, var = category)

Test = map_df(test_set, readJPEG_as_df, featureExtractor = myFeatures) %>% myImgDFReshape
tree = Test %>% ungroup %>% transmute(file=file, category = predict(fittree, ., type = "class"))
count(tree, var = category)

#neural net
#(extreme) boosting