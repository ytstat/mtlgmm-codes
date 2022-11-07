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

filename <- paste("/home/yt2170/work/mtlgmm/experiments/simulation_test_tl_new/result/", seed, ".RData", sep = "")


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
error_mtlgmm_center <- matrix(nrow = nrow(C_matrix), ncol = 5, dimnames = list(NULL, c("w", "mu", "beta", "Sigma", "prediction")))
error_mtlgmm <- matrix(nrow = nrow(C_matrix), ncol = 5, dimnames = list(NULL, c("w", "mu", "beta", "Sigma", "prediction")))
error_pooling <- matrix(nrow = nrow(C_matrix), ncol = 5, dimnames = list(NULL, c("w", "mu", "beta", "Sigma", "prediction")))
error_tlgmm <- matrix(nrow = nrow(C_matrix), ncol = 5, dimnames = list(NULL, c("w", "mu", "beta", "Sigma", "prediction")))

for (i in 1:nrow(C_matrix)) {
  flag <- 1
  while(flag) {
    print(i)
    n.list <- sample(c(100, 100), size = 11, replace = TRUE)
    data_bundle <- data_generation(K = 10, outlier_K = C_matrix[i, "outlier_num"], outlier_type = "parameter", generate_type = "mu", h_w = 0, h_mu = 0, n = n.list[1:10])
    data_bundle_target <- data_generation(K = 1, outlier_K = 0, outlier_type = "parameter", generate_type = "mu", h_w = C_matrix[i, "h_w"], h_mu = C_matrix[i, "h_mu"], n = n.list[11]+500)

    x_train_mtl <- sapply(1:10, function(k){
      data_bundle$data$x[[k]]
    }, simplify = FALSE)
    x_train <- data_bundle_target$data$x[[1]][1:n.list[11],]
    x_test <- data_bundle_target$data$x[[1]][-(1:n.list[11]),]
    y_test <- data_bundle_target$data$y[[1]][-(1:n.list[11])]


    # Target-GMM
    fitted_values <- initialize(list(x_train), "EM")

    # MTL-GMM-center and TL-GMM
    fit <- try(mtlgmm(x_train_mtl, eta_w = 0.2, eta_mu = 0.2, eta_beta = 0.01, C1_w = 0.01, C1_mu = 1, C1_beta = 1,
                  C2_w = 0.01, C2_mu = 1, C2_beta = 1, kappa0 = 1/3, initial_method = "EM", ncores = cores, cv_length = 5,
                  trim = 0.1, cv_upper = 2, cv_lower = 0.01, lambda = "cv", step_size = "lipschitz"))

    if (class(fit) != "try-error") {
      flag <- 0
      fit.tl <- tlgmm(x = x_train, fitted_bar = fit, eta_w = 0.2, eta_mu = 0.2, eta_beta = 0.01, C1_w = 0.01, C1_mu = 0.2, C1_beta = 0.2,
                      C2_w = 0.01, C2_mu = 1, C2_beta = 1, kappa0 = 1/3, initial_method = "EM", ncores = cores, cv_length = 5,
                      cv_upper = 2, cv_lower = 0.01, lambda = "cv", step_size = "lipschitz",
                      mtl_initial_mu1 = fit$initial_mu1, mtl_initial_mu2 = fit$initial_mu2)
    } else {
      next
    }

    # MTL-GMM
    x_train_mtl[[11]] <- x_train
    fit_mtl <- try(mtlgmm(x_train_mtl, eta_w = 0.2, eta_mu = 0.2, eta_beta = 0.01, C1_w = 0.01, C1_mu = 1, C1_beta = 1,
                      C2_w = 0.01, C2_mu = 1, C2_beta = 1, kappa0 = 1/3, initial_method = "EM", ncores = cores, cv_length = 5,
                      trim = 0.1, cv_upper = 2, cv_lower = 0.01, lambda = "cv", step_size = "lipschitz"))
    if (class(fit_mtl) == "try-error") {
      flag <- 1
      next
    }

    # Pooling
    K <- length(x_train_mtl)
    x.comb <- Reduce("rbind", x_train_mtl)
    fit_pooled <- quiet(Mclust(x.comb, G = 2, modelNames = "EEE"))
    fitted_values_pooled <- list(w = NULL, mu1 = NULL, mu2 = NULL, beta = NULL, Sigma = NULL)
    fitted_values_pooled$w <- fit_pooled$parameters$pro[1]
    fitted_values_pooled$mu1 <- fit_pooled$parameters$mean[,1]
    fitted_values_pooled$mu2 <- fit_pooled$parameters$mean[,2]
    fitted_values_pooled$Sigma <- fit_pooled$parameters$variance$Sigma
    fitted_values_pooled$beta <- solve(fit_pooled$parameters$variance$Sigma)%*% (fit_pooled$parameters$mean[,1] - fit_pooled$parameters$mean[,2])



    error_single[i, "w"] <- estimation_error(fitted_values$w, data_bundle_target$parameter$w, "w")
    error_mtlgmm_center[i, "w"] <- estimation_error(fit$w_bar, data_bundle_target$parameter$w, "w")
    error_mtlgmm[i, "w"] <- estimation_error(fit_mtl$w[[11]], data_bundle_target$parameter$w, "w")
    error_pooling[i, "w"] <- estimation_error(fitted_values_pooled$w, data_bundle_target$parameter$w, "w")
    error_tlgmm[i, "w"] <- estimation_error(fit.tl$w, data_bundle_target$parameter$w, "w")

    error_single[i, "mu"] <- estimation_error(list(fitted_values$mu1, fitted_values$mu2), list(data_bundle_target$parameter$mu1, data_bundle_target$parameter$mu2), "mu")
    error_mtlgmm_center[i, "mu"] <- estimation_error(list(fit$mu1_bar, fit$mu2_bar), list(data_bundle_target$parameter$mu1, data_bundle_target$parameter$mu2), "mu")
    error_mtlgmm[i, "mu"] <- estimation_error(list(fit_mtl$mu1[, 11], fit_mtl$mu2[, 11]), list(data_bundle_target$parameter$mu1, data_bundle_target$parameter$mu2), "mu")
    error_pooling[i, "mu"] <- estimation_error(list(fitted_values_pooled$mu1, fitted_values_pooled$mu2), list(data_bundle_target$parameter$mu1, data_bundle_target$parameter$mu2), "mu")
    error_tlgmm[i, "mu"] <- estimation_error(list(fit.tl$mu1, fit.tl$mu2), list(data_bundle_target$parameter$mu1, data_bundle_target$parameter$mu2), "mu")

    error_single[i, "beta"] <- estimation_error(fitted_values$beta, data_bundle_target$parameter$beta, "beta")
    error_mtlgmm_center[i, "beta"] <- estimation_error(fit$beta_bar, data_bundle_target$parameter$beta, "beta")
    error_mtlgmm[i, "beta"] <- estimation_error(fit_mtl$beta[, 11], data_bundle_target$parameter$beta, "beta")
    error_pooling[i, "beta"] <- estimation_error(fitted_values_pooled$beta, data_bundle_target$parameter$beta, "beta")
    error_tlgmm[i, "beta"] <- estimation_error(fit.tl$beta, data_bundle_target$parameter$beta, "beta")

    error_single[i, "Sigma"] <- estimation_error(fitted_values$Sigma, data_bundle_target$parameter$Sigma, "Sigma")
    error_mtlgmm_center[i, "Sigma"] <- NA
    error_mtlgmm[i, "Sigma"] <- estimation_error(fit_mtl$Sigma[11], data_bundle_target$parameter$Sigma, "Sigma")
    error_pooling[i, "Sigma"] <- estimation_error(list(fitted_values_pooled$Sigma), data_bundle_target$parameter$Sigma, "Sigma")
    error_tlgmm[i, "Sigma"] <- estimation_error(list(fit.tl$Sigma), data_bundle_target$parameter$Sigma, "Sigma")


    y_pred_single <- predict_gmm(w = fitted_values$w, mu1 = fitted_values$mu1, mu2 = fitted_values$mu2, beta = fitted_values$beta, newx = x_test)
    y_pred_mtlgmm_center <- predict_gmm(w = fit$w_bar, mu1 = fit$mu1_bar, mu2 = fit$mu2_bar, beta = fit$beta_bar, newx = x_test)
    y_pred_mtlgmm <- predict_gmm(w = fit_mtl$w[11], mu1 = fit_mtl$mu1[, 11], mu2 = fit_mtl$mu2[, 11], beta = fit_mtl$beta[, 11], newx = x_test)
    y_pred_pooling <- predict_gmm(w = fitted_values_pooled$w, mu1 = fitted_values_pooled$mu1, mu2 = fitted_values_pooled$mu2, beta = fitted_values_pooled$beta, newx = x_test)
    y_pred_tlgmm <- predict_gmm(w = fit.tl$w, mu1 = fit.tl$mu1, mu2 = fit.tl$mu2, beta = fit.tl$beta, newx = x_test)


    error_single[i, "prediction"] <- classification_error(y_pred_single, y_test)
    error_mtlgmm_center[i, "prediction"] <- classification_error(y_pred_mtlgmm_center, y_test)
    error_mtlgmm[i, "prediction"] <- classification_error(y_pred_mtlgmm, y_test)
    error_pooling[i, "prediction"] <- classification_error(y_pred_pooling, y_test)
    error_tlgmm[i, "prediction"] <- classification_error(y_pred_tlgmm, y_test)

    print(error_single[i, ])
    print(error_mtlgmm_center[i, ])
    print(error_mtlgmm[i, ])
    print(error_pooling[i, ])
    print(error_tlgmm[i, ])
  }

}


save(error_single, error_mtlgmm_center, error_mtlgmm, error_pooling, error_tlgmm, file = filename)




