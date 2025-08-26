library(tidyverse)
library(car)
library(dplyr)
library(vcd)
library(corrplot)

#loading the dataset to a dataframe
df <- read.csv('./Data/heart_2022_clean.csv')

#exploring the data
str(df)
head(df)
summary(df)

#looking for NA and NaN values
which(is.na(df))

#unique values in dataframe
ulst <- lapply(df, unique)
ulst


# continuous var columns
cnts_cols <- c('PhysicalHealthDays',
             'MentalHealthDays',
             'SleepHours',
             'HeightInMeters',
             'BMI',
             'GeneralHealth',
             'RemovedTeeth',
             'AgeCategory',
             'SmokerStatus_ord',
             'ECigaretteUsage_ord')
  
# categorical var columns
cat_cols <- c('AlcoholDrinkers',
             'BlindOrVisionDifficulty',
             'ChestScan',
             'CovidPos',
             'DeafOrHardOfHearing',
             'DifficultyConcentrating',
             'DifficultyDressingBathing',
             'DifficultyErrands',
             'DifficultyWalking',
             'FluVaxLast12',
             'HIVTesting',
             'HadAngina',
             'HadArthritis',
             'HadAsthma',
             'HadCOPD',
             'HadDepressiveDisorder',
             'HadDiabetes',
             'HadHeartAttack',
             'HadKidneyDisease',
             'HadSkinCancer',
             'HadStroke',
             'HighRiskLastYear',
             'LastCheckupTime',
             'PhysicalActivities',
             'PneumoVaxEver',
             'RaceEthnicityCategory',
             'Region',
             'Sex',
             'TetanusLast10Tdap')

{
#library(ggcorrplot)
## correlation matrix of categorical variables (automatically one-hot encodes)
#model.matrix(~0+., data=df[cat_cols]) %>% 
#  cor(use="pairwise.complete.obs") %>% 
#  ggcorrplot(show.diag=FALSE, type="lower", lab=TRUE, lab_size=2)
}


# Instantiate dataframe of categorical variables 
df_cat <-  df[cat_cols]

# Initialize empty matrix to store coefficients
empty_m <- matrix(ncol = length(df_cat),
                  nrow = length(df_cat),
                  dimnames = list(names(df_cat), 
                                  names(df_cat)))

# Function that accepts matrix for coefficients and data and returns a correlation matrix
calculate_cramer <- function(m, df_cat) {
  for (r in seq(nrow(m))){
    for (c in seq(ncol(m))){
      m[[r, c]] <- assocstats(table(df_cat[[r]], df_cat[[c]]))$cramer
    }
  }
  return(m)
}

# Create Cramer's V correlation matrix of categorical variables
cor_matrix <- calculate_cramer(empty_m ,df_cat)
corrplot(cor_matrix)

# top values in matrix excluding 1.0 correlation values
head(sort(cor_matrix,decreasing=TRUE),n=35)     # 0.4459, 0.4157, 0.3838, ...

# save as .csv file
write.csv(cor_matrix, 'C:/Users/user/Downloads/cor_matrix.csv')



# VIF for categorical & continuous variables
cnts_model <- lm(BMI ~., df[c(cnts_cols, cat_cols)])
vif(cnts_model)
cat_model <- lm(Sex ~., df[c(cnts_cols, cat_cols)])
vif(cat_model)

