#install.packages("rattle")
#install.packages("rpart.plot")
#install.packages("ROCR")
#install.packages("ggplot2")

library(rpart)

seed <- 42 

dataset <- read.csv("telcochurn.csv",header=TRUE)
head(dataset,5)

set.seed(seed) 
nobs <- nrow(dataset)  
train <- sample(nrow(dataset), 0.5*nobs) 

test <- setdiff(seq_len(nrow(dataset)), train) 

# Build the Decision Tree model.

input <- 2:35
target<- 36


rpart <- rpart(churn ~ .,
               data=dataset[train,c(input, target)],
               method="class",
               parms=list(split="information"),
               control=rpart.control(minsplit=20,
                                     minbucket=7))


print(rpart)
printcp(rpart)

# Obtain the response from the Decision Tree model.

pr <- predict(rpart, newdata=dataset[test, c(input, target)], type="class")



# Generate the confusion matrix showing counts.

table(dataset[test, c(input, target)]$churn, pr,
      useNA="ifany",
      dnn=c("Actual", "Predicted"))

# Generate the confusion matrix showing proportions.

pcme <- function(actual, cl)
{
  x <- table(actual, cl)
  nc <- nrow(x) # Number of classes.
  nv <- length(actual) - sum(is.na(actual) | is.na(cl)) # Number of values.
  tbl <- cbind(x/nv,
               Error=sapply(1:nc,
                            function(r) round(sum(x[r,-r])/sum(x[r,]), 2)))
  names(attr(tbl, "dimnames")) <- c("Actual", "Predicted")
  return(tbl)
}
per <- pcme(dataset[test, c(input, target)]$churn, pr)
round(per, 2) 

# Calculate the overall error percentage.

cat(100*round(1-sum(diag(per), na.rm=TRUE), 2)) #p=23 for (20,7) p=24 for(40,7)


# Plot the resulting Decision Tree. 

# We use the rpart.plot package.

library(rpart.plot)
rpart.plot(rpart)


# List the rules from the tree 

list.rules.rpart <- function(model)
{
  if (!inherits(model, "rpart")) stop("Not a legitimate rpart tree")
  #
  # Get some information.
  #
  frm     <- model$frame
  names   <- row.names(frm)
  ylevels <- attr(model, "ylevels")
  ds.size <- model$frame[1,]$n
  #
  # Print each leaf node as a rule.
  #
  for (i in 1:nrow(frm))
  {
    if (frm[i,1] == "<leaf>")
    {
      cat("\n")
      cat(sprintf(" Rule number: %s ", names[i]))
      cat(sprintf("[yval=%s cover=%d (%.0f%%) prob=%0.2f]\n",
                  ylevels[frm[i,]$yval], frm[i,]$n,
                  round(100*frm[i,]$n/ds.size), frm[i,]$yval2[,5]))
      pth <- path.rpart(model, nodes=as.numeric(names[i]), print.it=FALSE)
      cat(sprintf("   %s\n", unlist(pth)[-1]), sep="")
    }
  }
}

list.rules.rpart(rpart)



# Evaluate model performance. 

# ROC Curve: requires the ROCR package.

library(ROCR)

# ROC Curve: requires the ggplot2 package.

library(ggplot2, quietly=TRUE)

# Generate an ROC Curve for the rpart model

pr <- predict(rpart, newdata=dataset[test, c(input, target)])[,2]

# Remove observations with missing target.

no.miss   <- na.omit(dataset[test, c(input, target)]$churn)
miss.list <- attr(no.miss, "na.action")
attributes(no.miss) <- NULL

if (length(miss.list))
{
  pred <- prediction(pr[-miss.list], no.miss)
} else
{
  pred <- prediction(pr, no.miss)
}

pe <- performance(pred, "tpr", "fpr")
au <- performance(pred, "auc")@y.values[[1]]
pd <- data.frame(fpr=unlist(pe@x.values), tpr=unlist(pe@y.values))
p <- ggplot(pd, aes(x=fpr, y=tpr))
p <- p + geom_line(colour="red")
p <- p + xlab("False Positive Rate") + ylab("True Positive Rate")
p <- p + ggtitle("ROC Curve Decision Tree telcochurn.csv [test] churn")
p <- p + theme(plot.title=element_text(size=10))
p <- p + geom_line(data=data.frame(), aes(x=c(0,1), y=c(0,1)), colour="grey")
p <- p + annotate("text", x=0.50, y=0.00, hjust=0, vjust=0, size=5,
                  label=paste("AUC =", round(au, 2)))
print(p) # AUC =0.74 when (20,7), drop to 0.73 for (15,5),(40,7) no change for (30,10)


# Plot the lift chart.


# # Convert rate of positive predictions to percentage.
 
per <- performance(pred, "lift", "rpp")
per@x.values[[1]] <- per@x.values[[1]]*100

# Plot the lift chart.
ROCR::plot(per, col="#CC0000FF", lty=1, xlab="Caseload (%)", add=FALSE)


legend("topright", c("Test"), col=rainbow(2, 1, .8), lty=1:2, title="Decision Tree", inset=c(0.05, 0.05))

title(main="Lift Chart")
grid() # lift value lower for (40,7)


# Score a dataset. 


prcls <- predict(rpart, newdata=dataset[test, c(input)], type="class")
pr <- predict(rpart, newdata=dataset[test, c(input)])

# Extract the relevant variables from the dataset.

sdata <- subset(dataset[test,], select=c(1, 36))

# Output the combined data.

write.csv(cbind(sdata, prcls, pr), file="/Users/josephine/Desktop/Workshop/BAP/03 PA/Day4/telcochurn_test_score_idents_DT.csv", row.names=FALSE)


# Build a Random Forest model using the traditional approach.

set.seed(seed)

rf <- randomForest::randomForest(as.factor(churn) ~ .,
                                     data=dataset[train, c(input, target)], 
                                     ntree=150,
                                     mtry=5,
                                     importance=TRUE,
                                     na.action=randomForest::na.roughfix,
                                     replace=FALSE)

# Generate textual output of the 'Random Forest' model.

rf

# The `pROC' package implements various AUC functions.

# Calculate the Area Under the Curve (AUC).

pROC::roc(rf$y, as.numeric(rf$predicted)) #0.6951


# List the importance of the variables.

rn <- round(randomForest::importance(rf), 2)
rn[order(rn[,3], decreasing=TRUE),]



# The `ada' package implements the boost algorithm.

# Build the Ada Boost model.

set.seed(seed)
ada <- ada::ada(churn ~ .,
                    data=dataset[train,c(input, target)],
                    control=rpart::rpart.control(maxdepth=6,
                                                 cp=0.010000,
                                                 minsplit=20,
                                                 xval=10),
                    iter=50)

# Print the results of the modelling.
library(rattle)
print(ada) #0.114
round(ada$model$errs[ada$iter,], 2)
cat('Variables actually used in tree construction:\n')
print(sort(names(listAdaVarsUsed(ada))))
cat('\nFrequency of variables actually used:\n')
print(listAdaVarsUsed(ada))
