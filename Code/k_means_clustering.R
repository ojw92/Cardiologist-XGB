library(scales)

tr_df <- read.csv('./Data/train_df0.1.csv')
te_df <- read.csv('./Data/test_df0.1.csv')
# Concatenate train and test data
df <- rbind(tr_df, te_df)
colnames(df)

# Scale all columns to 0-1
df_scaled <- as.data.frame(lapply(df, rescale))

#k-means for clustering
n = 20 #max. amount of clusters to include
tot.withinss <- numeric(n) #vector for total within-cluster sum of squares

# applying k-means for each number of clusters
for (c in 1:n) {
  km_result <- kmeans(df_scaled, centers = c)
  tot.withinss[c] <- sum(km_result$tot.withinss)
}

# create elbow plot
plot(1:n, tot.withinss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
text(1:n, tot.withinss, labels = 1:n, pos = 3, cex = 0.5)


# k-means with 6 clusters
k = 6 #based on the elbow plot
kmeans_result <- kmeans(df_scaled, centers = k)

# add cluster column to dataframe
df$cluster <- kmeans_result$cluster
# save dataframe with cluster column
write.csv(df, file = "./Data/clustered_df.csv")