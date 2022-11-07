# NYU Greene

.libPaths("/home/yt2170/R/x86_64-centos-linux-gnu-library/4.0/")


library(dplyr)
library(mclust)
library(doParallel)
library(MASS)
library(caret)
library(mtlgmm)


Sys.setenv(LANG = "en_US.UTF-8")
seed <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
cat("seed=", seed, "\n")

filename <- paste("/home/yt2170/work/mtlgmm/experiments/real-data/har/result/", seed, ".RData", sep = "")


if (file.exists(filename)) {
  stop("Done!")
}

set.seed(seed, kind = "L'Ecuyer-CMRG")
if(Sys.getenv("SLURM_CPUS_PER_TASK") != "") {
  cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK"))
} else {
  cores <- detectCores()
}




# -------------------------------------------------------
D_train <- read.table("/home/yt2170/work/mtlgmm/datasets/har/X_train.txt")
D_y <- read.table("/home/yt2170/work/mtlgmm/datasets/har/y_train.txt")
D_id <- read.table("/home/yt2170/work/mtlgmm/datasets/har/subject_train.txt")
D_test <- read.table("/home/yt2170/work/mtlgmm/datasets/har/X_test.txt")
D_y_test <- read.table("/home/yt2170/work/mtlgmm/datasets/har/y_test.txt")
D_id_test <- read.table("/home/yt2170/work/mtlgmm/datasets/har/subject_test.txt")

D_train <- rbind(D_train, D_test) %>% mutate(id = c(as.numeric(unlist(D_id)), as.numeric(unlist(D_id_test))), y = c(as.numeric(unlist(D_y)), as.numeric(unlist(D_y_test))))


x <- sapply(1:length(unique(D_train$id)), function(k){
  D_train %>% filter(id == unique(D_train$id)[k], y %in% c(5, 6)) %>% dplyr::select(-id, -y)
}, simplify = FALSE)
y <- sapply(1:length(unique(D_train$id)), function(k){
  a <- D_train %>% filter(id == unique(D_train$id)[k], y %in% c(5, 6)) %>% dplyr::select(y) %>% unlist(.) %>% as.numeric(.)
  a[a == 5] <- 1
  a[a == 6] <- 2
  a
}, simplify = FALSE)



train_id <- sapply(1:length(x), function(k){
  sample(1:nrow(x[[k]]), floor(nrow(x[[k]])*0.9))
}, simplify = FALSE)

pca_train <- sapply(1:length(unique(D_train$id)), function(k){
  prcomp(x[[k]][train_id[[k]], ])
}, simplify = FALSE)

x_train <- sapply(1:length(x), function(k){
  pca_train[[k]]$x[, 1:5]
}, simplify = FALSE)

x_test <- sapply(1:length(x), function(k){
  predict(pca_train[[k]], newdata = x[[k]][-train_id[[k]], ])[, 1:5]
}, simplify = FALSE)

y_test <- sapply(1:length(x), function(k){
  y[[k]][-train_id[[k]]]
}, simplify = FALSE)


x_train_std <- x_train
x_test_std <- x_test

# Single-task GMM
fitted_values <- initialize(x_train_std, "EM")
L <- alignment(fitted_values$mu1, fitted_values$mu2, method = "greedy")
fitted_values <- alignment_swap(L$L1, L$L2, initial_value_list = fitted_values)
y_pred_single <- sapply(1:length(x), function(k){
  predict_gmm(w = fitted_values$w[k], mu1 = fitted_values$mu1[, k], mu2 = fitted_values$mu2[, k], beta = fitted_values$beta[, k], newx = x_test_std[[k]])
}, simplify = FALSE)


# Pooling
K <- length(x)
x.comb <- Reduce("rbind", x_train_std)
fit_pooled <- quiet(Mclust(x.comb, G = 2, modelNames = "EEE"))
fitted_values_pooled <- list(w = NULL, mu1 = NULL, mu2 = NULL, beta = NULL, Sigma = NULL)
fitted_values_pooled$w <- rep(fit_pooled$parameters$pro[1], K)
fitted_values_pooled$mu1 <- matrix(rep(fit_pooled$parameters$mean[,1], K), ncol = K)
fitted_values_pooled$mu2 <- matrix(rep(fit_pooled$parameters$mean[,2], K), ncol = K)
fitted_values_pooled$Sigma <- sapply(1:K, function(k){
  fit_pooled$parameters$variance$Sigma
}, simplify = FALSE)
fitted_values_pooled$beta <- sapply(1:K, function(k){
  solve(fit_pooled$parameters$variance$Sigma)%*% (fit_pooled$parameters$mean[,1] - fit_pooled$parameters$mean[,2])
})


y_pred_pooling <- sapply(1:length(x), function(k){
  predict_gmm(w = fitted_values_pooled$w[k], mu1 = fitted_values_pooled$mu1[, k], mu2 = fitted_values_pooled$mu2[, k], beta = fitted_values_pooled$beta[, k], newx = x_test_std[[k]])
}, simplify = FALSE)

c(classification_error(y_pred_single, y_test),
  classification_error(y_pred_pooling, y_test))

# MTL-GMM

fit <- mtlgmm(x_train_std, eta_w = 0.1, eta_mu = 0.1, eta_beta = 0.01, C1_w = 0.01, C1_mu = 1, C1_beta = 1,
              C2_w = 0.01, C2_mu = 1, C2_beta = 1, kappa0 = 1/3, initial_method = "EM", ncores = cores, cv_length = 5,
              trim = 0.1, cv_upper = 2, cv_lower = 0.01, lambda = "cv", step_size = "lipschitz")

y_pred_mtlgmm <- sapply(1:length(x), function(k){
  predict_gmm(w = fit$w[k], mu1 = fit$mu1[, k], mu2 = fit$mu2[, k], beta = fit$beta[, k], newx = x_test_std[[k]])
}, simplify = FALSE)



error <- c(classification_error(y_pred_single, y_test),
           classification_error(y_pred_pooling, y_test),
           classification_error(y_pred_mtlgmm, y_test))

print(error)
# -------------------------------------------------------
save(error, file = filename)



