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
if (seed %% 30 == 0) {
  job.id <- 30
} else {
  job.id <- seed %% 30
}

cat("seed=", seed, "\n")
cat("job.id=", job.id, "\n")


filename = paste("/home/yt2170/work/mtlgmm/experiments/real-data/har_tl/result/", seed, ".RData", sep = "")
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

id_set <- unique(D_train$id)

x <- sapply(1:length(id_set), function(k){
  D_train %>% filter(id == id_set[k]) %>% dplyr::select(-id, -y)
}, simplify = FALSE)

y <- sapply(1:length(id_set), function(k){
  a <- D_train %>% filter(id == id_set[k]) %>% dplyr::select(y) %>% unlist(.) %>% as.numeric(.)
  a[a %in% 1:3] <- 1
  a[a %in% 4:6] <- 2
  a
}, simplify = FALSE)



# ---------------
j <- job.id

x_source <- sapply(setdiff(1:length(id_set), j), function(k){
  train_id_source <- sample(1:nrow(x[[k]]), floor(nrow(x[[k]])*1))
  pca_source <- prcomp(x[[k]][train_id_source, ])
  pca_source$x[, 1:30]
}, simplify = FALSE)

train_id <- sample(1:nrow(x[[j]]), floor(nrow(x[[j]])*0.95))

pca_train <- prcomp(x[[j]][train_id, ])
x_train <- pca_train$x[, 1:30]

x_test <- predict(pca_train, newdata = x[[j]][-train_id, ])[, 1:30]

y_test <- y[[j]][-train_id]

x_all <- x_source
x_all[[length(id_set)]] <- x_train

# Target GMM
fitted_values <- initialize(list(x_train), "EM")
y_pred_single <- predict_gmm(w = fitted_values$w, mu1 = fitted_values$mu1, mu2 = fitted_values$mu2, beta = fitted_values$beta, newx = x_test)

# Pooling
K <- length(x_all)
x.comb <- Reduce("rbind", x_all)
fit_pooled <- quiet(Mclust(x.comb, G = 2, modelNames = "EEE"))
fitted_values_pooled <- list(w = NULL, mu1 = NULL, mu2 = NULL, beta = NULL, Sigma = NULL)
fitted_values_pooled$w <- fit_pooled$parameters$pro[1]
fitted_values_pooled$mu1 <- fit_pooled$parameters$mean[,1]
fitted_values_pooled$mu2 <- fit_pooled$parameters$mean[,2]
fitted_values_pooled$Sigma <- fit_pooled$parameters$variance$Sigma
fitted_values_pooled$beta <- solve(fit_pooled$parameters$variance$Sigma)%*% (fit_pooled$parameters$mean[,1] - fit_pooled$parameters$mean[,2])
y_pred_pooling <- predict_gmm(w = fitted_values_pooled$w, mu1 = fitted_values_pooled$mu1, mu2 = fitted_values_pooled$mu2, beta = fitted_values_pooled$beta, newx = x_test)


# MTL-GMM-center
fit <- mtlgmm(x_source, eta_w = 0.1, eta_mu = 0.1, eta_beta = 0.01, C1_w = 0.01, C1_mu = 1, C1_beta = 1,
              C2_w = 0.01, C2_mu = 1, C2_beta = 1, kappa0 = 1/3, initial_method = "EM", ncores = cores, cv_length = 5,
              trim = 0.1, cv_upper = 2, cv_lower = 0.01, lambda = "cv", step_size = "lipschitz")
y_pred_mtlgmm_center <- predict_gmm(w = fit$w_bar, mu1 = fit$mu1_bar, mu2 = fit$mu2_bar, beta = fit$beta_bar, newx = x_test)

# TL-GMM
fit.tl <- tlgmm(x = x_train, fitted_bar = fit, eta_w = 0.2, eta_mu = 0.2, eta_beta = 0.01, C1_w = 0.01, C1_mu = 0.2, C1_beta = 0.2,
                C2_w = 0.01, C2_mu = 1, C2_beta = 1, kappa0 = 1/3, initial_method = "EM", ncores = cores, cv_length = 5,
                cv_upper = 2, cv_lower = 0.01, lambda = "cv", step_size = "lipschitz",
                mtl_initial_mu1 = fit$initial_mu1, mtl_initial_mu2 = fit$initial_mu2)
y_pred_tlgmm <- predict_gmm(w = fit.tl$w, mu1 = fit.tl$mu1, mu2 = fit.tl$mu2, beta = fit.tl$beta, newx = x_test)


# MTL-GMM
fit_mtlgmm <- mtlgmm(x_all, eta_w = 0.1, eta_mu = 0.1, eta_beta = 0.01, C1_w = 0.01, C1_mu = 1, C1_beta = 1,
              C2_w = 0.01, C2_mu = 1, C2_beta = 1, kappa0 = 1/3, initial_method = "EM", ncores = cores, cv_length = 5,
              trim = 0.1, cv_upper = 2, cv_lower = 0.01, lambda = "cv", step_size = "lipschitz")
y_pred_mtlgmm <- predict_gmm(w = fit_mtlgmm$w[length(id_set)], mu1 = fit_mtlgmm$mu1[, length(id_set)], mu2 = fit_mtlgmm$mu2[, length(id_set)], beta = fit_mtlgmm$beta[, length(id_set)], newx = x_test)


error <- c(classification_error(y_pred_single, y_test),
           classification_error(y_pred_pooling, y_test),
           classification_error(y_pred_mtlgmm_center, y_test),
           classification_error(y_pred_mtlgmm, y_test),
           classification_error(y_pred_tlgmm, y_test))

names(error) <- c("Target-GMM", "Pooling-GMM", "MTL-GMM-center", "MTL-GMM", "TL-GMM")



print(error)



# -------------------------------------------------------
save(error, file = filename)



