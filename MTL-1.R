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

filename <- paste("/home/yt2170/work/mtlgmm/experiments/simulation_test/result/", seed, ".RData", sep = "")


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
h_w <- c(0.05, 0.15)
h_mu <- seq(0, 2, 0.2)
outlier_num <- 0:2
C_matrix <- as.matrix(expand.grid(h_w, h_mu, outlier_num))
colnames(C_matrix) <- c("h_w", "h_mu", "outlier_num")

error_single <- matrix(nrow = nrow(C_matrix), ncol = 5, dimnames = list(NULL, c("w", "mu", "beta", "Sigma", "prediction")))
error_mtlgmm <- matrix(nrow = nrow(C_matrix), ncol = 5, dimnames = list(NULL, c("w", "mu", "beta", "Sigma", "prediction")))
error_pooling <- matrix(nrow = nrow(C_matrix), ncol = 5, dimnames = list(NULL, c("w", "mu", "beta", "Sigma", "prediction")))

for (i in 1:nrow(C_matrix)) {
  flag <- 1
  while(flag) {
    print(i)
    data_bundle <- data_generation(K = 10, outlier_K = C_matrix[i, "outlier_num"], outlier_type = "parameter", generate_type = "mu", h_w = C_matrix[i, "h_w"], h_mu = C_matrix[i, "h_mu"], n = 600)

    # data_bundle <- data_generation(K = 10, outlier_K = 0, outlier_type = "parameter", h_w = 0.05, h_mu = 1.4, n = 600, generate_type = "beta")
    x_train <- sapply(1:10, function(k){
      data_bundle$data$x[[k]][1:100,]
    }, simplify = FALSE)
    x_test <- sapply(1:10, function(k){
      data_bundle$data$x[[k]][-(1:100),]
    }, simplify = FALSE)
    y_test <- sapply(1:10, function(k){
      data_bundle$data$y[[k]][-(1:100)]
    }, simplify = FALSE)


    # Single-task GMM
    # x <- data_bundle$data$x
    x <- x_train
    fitted_values <- initialize(x, "EM")
    L <- alignment(fitted_values$mu1, fitted_values$mu2, method = "greedy")
    fitted_values <- alignment_swap(L$L1, L$L2, initial_value_list = fitted_values)

    # fit <- mtlgmm(x, eta_w = 0.1, eta_mu = 0.1, eta_beta = 0.01, C1_w = 0.01, C1_mu = 0.5, C1_beta = 0.5,
    #               C2_w = 0.01, C2_mu = 0.5, C2_beta = 0.5, kappa0 = 1/3, initial_method = "EM", ncores = cores, cv_length = 5,
    #               trim = 0.1, cv_upper = 2, cv_lower = 0.01, lambda = "fixed", step_size = "lipschitz")

    # MTL-GMM
    fit <- try(mtlgmm(x, eta_w = 0.2, eta_mu = 0.2, eta_beta = 0.01, C1_w = 0.01, C1_mu = 1, C1_beta = 1,
                  C2_w = 0.01, C2_mu = 1, C2_beta = 1, kappa0 = 1/3, initial_method = "EM", ncores = cores, cv_length = 5,
                  trim = 0.1, cv_upper = 2, cv_lower = 0.01, lambda = "cv", step_size = "lipschitz"))

    if (class(fit) != "try-error") {
      flag <- 0
    } else {
      next
    }
    #
    #   fit <- mtlgmm2(x, eta_w = 0.1, eta_mu = 0.1, eta_beta = 0.01, C1_w = 0.01, C1_mu = 0.5, C1_beta = 0.5,
    #                  C2_w = 0.01, C2_mu = 0.5, C2_beta = 0.5, kappa0 = 1/3,  cv_length = 5,
    #                  cv_upper = 2, cv_lower = 0.01, lambda = "fixed", step_size = "lipschitz")

    # Pooling
    K <- 10
    x.comb <- Reduce("rbind", x)
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


    if (C_matrix[i, "outlier_num"] == 0) {
      data_bundle$data$outlier_index <- 11
    }

    error_single[i, "w"] <- estimation_error(fitted_values$w[-data_bundle$data$outlier_index], data_bundle$parameter$w[-data_bundle$data$outlier_index], "w")
    error_pooling[i, "w"] <- estimation_error(fitted_values_pooled$w[-data_bundle$data$outlier_index], data_bundle$parameter$w[-data_bundle$data$outlier_index], "w")
    error_mtlgmm[i, "w"] <- estimation_error(fit$w[-data_bundle$data$outlier_index], data_bundle$parameter$w[-data_bundle$data$outlier_index], "w")

    error_single[i, "mu"] <- estimation_error(list(fitted_values$mu1[, -data_bundle$data$outlier_index], fitted_values$mu2[, -data_bundle$data$outlier_index]), list(data_bundle$parameter$mu1[, -data_bundle$data$outlier_index], data_bundle$parameter$mu2[, -data_bundle$data$outlier_index]), "mu")
    error_pooling[i, "mu"] <- estimation_error(list(fitted_values_pooled$mu1[, -data_bundle$data$outlier_index], fitted_values_pooled$mu2[, -data_bundle$data$outlier_index]), list(data_bundle$parameter$mu1[, -data_bundle$data$outlier_index], data_bundle$parameter$mu2[, -data_bundle$data$outlier_index]), "mu")
    error_mtlgmm[i, "mu"] <- estimation_error(list(fit$mu1[, -data_bundle$data$outlier_index], fit$mu2[, -data_bundle$data$outlier_index]), list(data_bundle$parameter$mu1[, -data_bundle$data$outlier_index], data_bundle$parameter$mu2[, -data_bundle$data$outlier_index]), "mu")

    error_single[i, "beta"] <- estimation_error(fitted_values$beta[, -data_bundle$data$outlier_index], data_bundle$parameter$beta[, -data_bundle$data$outlier_index], "beta")
    error_pooling[i, "beta"] <- estimation_error(fitted_values_pooled$beta[, -data_bundle$data$outlier_index], data_bundle$parameter$beta[, -data_bundle$data$outlier_index], "beta")
    error_mtlgmm[i, "beta"] <- estimation_error(fit$beta[, -data_bundle$data$outlier_index], data_bundle$parameter$beta[, -data_bundle$data$outlier_index], "beta")

    error_single[i, "Sigma"] <- estimation_error(fitted_values$Sigma[-data_bundle$data$outlier_index], data_bundle$parameter$Sigma[-data_bundle$data$outlier_index], "Sigma")
    error_pooling[i, "Sigma"] <- estimation_error(fitted_values_pooled$Sigma[-data_bundle$data$outlier_index], data_bundle$parameter$Sigma[-data_bundle$data$outlier_index], "Sigma")
    error_mtlgmm[i, "Sigma"] <- estimation_error(fit$Sigma[-data_bundle$data$outlier_index], data_bundle$parameter$Sigma[-data_bundle$data$outlier_index], "Sigma")


    y_pred_single <- sapply(1:10, function(k){
      predict_gmm(w = fitted_values$w[k], mu1 = fitted_values$mu1[, k], mu2 = fitted_values$mu2[, k], beta = fitted_values$beta[, k], newx = x_test[[k]])
    }, simplify = FALSE)
    y_pred_pooling <- sapply(1:10, function(k){
      predict_gmm(w = fitted_values_pooled$w[k], mu1 = fitted_values_pooled$mu1[, k], mu2 = fitted_values_pooled$mu2[, k], beta = fitted_values_pooled$beta[, k], newx = x_test[[k]])
    }, simplify = FALSE)
    y_pred_mtlgmm <- sapply(1:10, function(k){
      predict_gmm(w = fit$w[k], mu1 = fit$mu1[, k], mu2 = fit$mu2[, k], beta = fit$beta[, k], newx = x_test[[k]])
    }, simplify = FALSE)

    error_single[i, "prediction"] <- classification_error(y_pred_single[-data_bundle$data$outlier_index], y_test[-data_bundle$data$outlier_index])
    error_pooling[i, "prediction"] <- classification_error(y_pred_pooling[-data_bundle$data$outlier_index], y_test[-data_bundle$data$outlier_index])
    error_mtlgmm[i, "prediction"] <- classification_error(y_pred_mtlgmm[-data_bundle$data$outlier_index], y_test[-data_bundle$data$outlier_index])

    print(error_single[i, ])
    print(error_pooling[i, ])
    print(error_mtlgmm[i, ])
  }

}


save(error_single, error_pooling, error_mtlgmm, file = filename)




