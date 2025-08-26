library(tidyverse)
library(MASS)
library(caret)

options(width = 200)

set.seed(123)

df <- read.csv("./Data/heart_2022_clean.csv")
colnames(df)
dim(df)
# current working dir

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

# In RaceEthnicityCategory column replace spaces with underscores
df$RaceEthnicityCategory <- gsub(" ", "_", df$RaceEthnicityCategory)
# String columns that need converting to factors
df$RaceEthnicityCategory <- as.factor(df$RaceEthnicityCategory)
df$Region <- as.factor(df$Region)

# for LastCheckupTime convert "Within past year" to 2, "Over 2 years ago" to 0, and "Within past 2 years" to 1
df$LastCheckupTime <- ifelse(df$LastCheckupTime == "Within past year", 2, ifelse(df$LastCheckupTime == "Over 2 years ago", 0, 1))
# TetanusLast10Tdap convert Yes-Tdap to 2, No to 0, and the rest to 1
df$TetanusLast10Tdap <- ifelse(df$TetanusLast10Tdap == "Yes-Tdap", 2, ifelse(df$TetanusLast10Tdap == "No", 0, 1))

# stepwise logistic regression model
all_cols <- c(cnts_cols, cat_cols)
df <- df[all_cols]
names(df)

full_glm <- glm(HadHeartAttack ~ ., data = df, family = binomial)
summary(full_glm)

step_model <- step(full_glm, direction = "both", trace = .5, steps = 500)
summary(step_model)

# save model summary output to txt
summary_full_glm <- summary(full_glm)
summary_step_model <- summary(step_model)

sink("./Code/model_summaries.txt")
cat("Summary of Full Model:\n")
print(summary(full_glm))
cat("\n")
cat("Summary of Stepwise Model:\n")
print(summary(step_model))
sink()

# # generate and save all diagnostic plots
# png("./Visualizations/stepwise_diagnostic_plots_%d.png")
# for (i in 1:length(attributes(plot(step_model))$makePlot)) {
#   if (attributes(plot(step_model))$makePlot[i]) {
#     plot(step_model, which = i)
#   }
# }
# dev.off() # close graphs

# create df with significant variables (0.05)
p_values <- summary(step_model)$coefficients[-1, 4]
p_values

significant_variables <- names(p_values[p_values <= 0.1])
significant_variables

# convert factor variables in df to dummy variables
dmy <- dummyVars(" ~ .", data = df, sep=NULL)
df_dummy <- data.frame(predict(dmy, newdata = df))
# replace Non.Hispanic in column names with Non-Hispanic
colnames(df_dummy) <- gsub("Non.Hispanic", "Non-Hispanic", colnames(df_dummy))
# replace dot in column names with comma
colnames(df_dummy) <- gsub("\\.", ",", colnames(df_dummy))

# print what is in significant_variables that is not in colnames(df_dummy)
significant_variables[!(significant_variables %in% colnames(df_dummy))]

# add HadHeartAttack to significant_variables
significant_variables <- c(significant_variables, "HadHeartAttack")

df_significant <- df_dummy[, significant_variables]

# save df with significant variables
write.csv(df_significant, "./Data/heart_2022_clean_stepwise0.1.csv", row.names = FALSE)
