# Time Series Modeling of Financial Data with R
#### Prof. Daniel P. Palomar
#### 
MAFS5310 - Portfolio Optimization with R  
MSc in Financial Mathematics  
The Hong Kong University of Science and Technology (HKUST)  
Fall 2020-21
* * *
This R session will illustrate the different models for univariate and multivariate financial time series, in particular, for the conditional mean and conditional covariance matrix or volatility.
# Mean models
This section explores conditional mean models.
## Packages
The following R packages are widely used by the R community for mean modeling:
-   [**rugarch**](https://cran.r-project.org/package=rugarch): \[see [intro to rugarch](https://cran.r-project.org/package=rugarch/vignettes/Introduction_to_the_rugarch_package.pdf)\] probably the most comprehensive package with a neverending list of different ARMA and GARCH models, including different distributions for the residual, as well as utilities for simulating, forecasting, and plotting;
-   [**forecast**](https://cran.r-project.org/package=forecast): also a popular package for ARMA modeling (see the book [https://otexts.com/fpp2](https://otexts.com/fpp2));
-   [**MTS**](https://cran.r-project.org/package=MTS): multivariate time series modeling based on Tsay’s book.
There are many other packages worth exploring such as [**prophet**](https://cran.r-project.org/package=prophet), [**tsDyn**](https://cran.r-project.org/package=tsDyn), [**vars**](https://cran.r-project.org/package=vars), [**rucm**](https://cran.r-project.org/package=rucm), etc. For a full list of packages for time series modeling, check the [CRAN Task View: Time Series Analysis](https://cran.r-project.org/view=TimeSeries).
## i.i.d. model
We will start with the simple i.i.d. model as a warm-up. The i.i.d. model assumes an NNN\-dimensional Gaussian time series for the log-returns xtxt\\mathbf{x}\_t:
xt∼N(μ,Σ).xt∼N(μ,Σ).
\\mathbf{x}\_t \\sim \\mathcal{N}(\\boldsymbol{\\mu}, \\boldsymbol{\\Sigma}). The sample estimators for the mean and covariance matrix are, respectively, the sample meanμ^\=1T∑t\=1Txtμ^\=1T∑t\=1Txt
\\hat{\\boldsymbol{\\mu}} = \\frac{1}{T}\\sum\_{t=1}^T \\mathbf{x}\_t and the sample covariance matrix
Σ^\=1T−1∑t\=1T(xt−μ^)T(xt−μ^).Σ^\=1T−1∑t\=1T(xt−μ^)T(xt−μ^).
\\hat{\\boldsymbol{\\Sigma}} = \\frac{1}{T-1}\\sum\_{t=1}^T (\\mathbf{x}\_t - \\hat{\\boldsymbol{\\mu}})^T(\\mathbf{x}\_t - \\hat{\\boldsymbol{\\mu}}).
We start by generating synthetic data to become familiar with the procedure and to make sure the estimation process gives the correct result (i.e., sanity check). Only once everything is under control we will move to real market data and fit the different models.
Let’s generate synthetic i.i.d. data and estimate the mean and covariance matrix:
    # generate Gaussian synthetic return data
    library(mvtnorm)
    set.seed(357)
    N <- 100
    T <- 120
    mu <- runif(N)
    U <- t(rmvnorm(n = round(0.7*N), sigma = 0.1*diag(N)))
    Sigma <- U %*% t(U) + diag(N)
    X <- rmvnorm(n = T, mean = mu, sigma = Sigma)
    
    # sample estimates (sample mean and sample covariance matrix)
    mu_sm <- colMeans(X)
    Sigma_scm <- cov(X)
    
    # errors
    norm(mu_sm     - mu, "2")
    #> [1] 2.443633
    norm(Sigma_scm - Sigma, "F")
    #> [1] 70.79889
Now, let’s do it again for different number of observations TTT:
    # first generate all the data
    set.seed(357)
    N <- 100
    T_max <- 1000
    mu <- runif(N)
    U <- t(rmvnorm(n = round(0.7*N), sigma = 0.1*diag(N)))
    Sigma <- U %*% t(U) + diag(N)
    X <- rmvnorm(n = T_max, mean = mu, sigma = Sigma)
    
    # now loop over subsets of the samples
    error_mu_vs_T <- error_Sigma_vs_T <- NULL
    T_sweep <- ceiling(seq(1.01*N, T_max, length.out = 20))
    for (T_ in T_sweep) {
      X_ <- X[1:T_, ]
      # sample estimates
      mu_sm <- colMeans(X_)
      Sigma_scm <- cov(X_)
      # compute errors
      error_mu_vs_T    <- c(error_mu_vs_T,    norm(mu_sm     - mu, "2"))
      error_Sigma_vs_T <- c(error_Sigma_vs_T, norm(Sigma_scm - Sigma, "F"))
    }
    names(error_mu_vs_T) <- names(error_Sigma_vs_T) <- paste("T =", T_sweep)
    
    # plots
    plot(T_sweep, error_mu_vs_T, type = "b", pch = 20, col = "blue",
         main = "Error in estimation of mu", xlab = "T", ylab = "error")
    plot(T_sweep, error_Sigma_vs_T, type = "b", pch = 20, col = "blue",
         main = "Error in estimation of Sigma", xlab = "T", ylab = "error")
## Univariate ARMA model
An ARMA(p,q) model on the log-returns xtxtx\_t is
xt\=ϕ0+∑i\=1pϕixt−i+wt−∑j\=1qθjwt−ixt\=ϕ0+∑i\=1pϕixt−i+wt−∑j\=1qθjwt−i
x\_t = \\phi\_0 + \\sum\_{i=1}^p \\phi\_i x\_{t-i} + w\_t - \\sum\_{j=1}^q \\theta\_j w\_{t-i} where wtwtw\_t is a white noise series with zero mean and variance σ2σ2\\sigma^2. The parameters of the model are the coefficients ϕiϕi\\phi\_i, θiθi\\theta\_i, and the noise variance σ2σ2\\sigma^2.
Note that an ARIMA(p,d,q) model is simply an ARMA(p,q) model on a time series that has been differenced ddd times. So if we xtxtx\_t denotes instead the log-prices, then the previous model on the log-returns is actually an ARIMA(p,1,q) model, because differencing once the log-prices we obtain the log-returns.
### Synthetic data generation with package **rugarch**
We will use the package **rugarch** to generate synthetic univariate ARMA data, estimate the parameters, and forecast.
First, we need to define the model order:
    library(rugarch)
    
    # specify an AR(1) model with given coefficients and parameters
    arma_fixed_spec <- arfimaspec(mean.model = list(armaOrder = c(1,0), include.mean = TRUE), 
                                  fixed.pars = list(mu = 0.01, ar1 = -0.9, sigma = 0.2))
    arma_fixed_spec
    #> 
    #> *----------------------------------*
    #> *       ARFIMA Model Spec          *
    #> *----------------------------------*
    #> Conditional Mean Dynamics
    #> ------------------------------------
    #> Mean Model           : ARFIMA(1,0,0)
    #> Include Mean     : TRUE 
    #> 
    #> Conditional Distribution
    #> ------------------------------------
    #> Distribution :  norm 
    #> Includes Skew    :  FALSE 
    #> Includes Shape   :  FALSE 
    #> Includes Lambda  :  FALSE
    
    arma_fixed_spec@model$pars
    #>          Level Fixed Include Estimate LB UB
    #> mu        0.01     1       1        0 NA NA
    #> ar1      -0.90     1       1        0 NA NA
    #> ma        0.00     0       0        0 NA NA
    #> arfima    0.00     0       0        0 NA NA
    #> archm     0.00     0       0        0 NA NA
    #> mxreg     0.00     0       0        0 NA NA
    #> sigma     0.20     1       1        0 NA NA
    #> alpha     0.00     0       0        0 NA NA
    #> beta      0.00     0       0        0 NA NA
    #> gamma     0.00     0       0        0 NA NA
    #> eta1      0.00     0       0        0 NA NA
    #> eta2      0.00     0       0        0 NA NA
    #> delta     0.00     0       0        0 NA NA
    #> lambda    0.00     0       0        0 NA NA
    #> vxreg     0.00     0       0        0 NA NA
    #> skew      0.00     0       0        0 NA NA
    #> shape     0.00     0       0        0 NA NA
    #> ghlambda  0.00     0       0        0 NA NA
    #> xi        0.00     0       0        0 NA NA
    
    arma_fixed_spec@model$fixed.pars
    #> $mu
    #> [1] 0.01
    #> 
    #> $ar1
    #> [1] -0.9
    #> 
    #> $sigma
    #> [1] 0.2
    
    true_params <- unlist(arma_fixed_spec@model$fixed.pars)
    true_params
    #>    mu   ar1 sigma 
    #>  0.01 -0.90  0.20
Then, we can generate one realization of the time series path:
    library(xts)
    # simulate one path
    T <- 2000
    set.seed(42)
    path_arma <- arfimapath(arma_fixed_spec, n.sim = T)
    
    str(path_arma@path$seriesSim)
    #>  num [1:2000, 1] 0.284 -0.35 0.406 -0.22 0.298 ...
    
    # convert to xts and plot
    synth_log_returns <- xts(path_arma@path$seriesSim, order.by = as.Date("2010-01-01") + 0:(T-1))
    plot(synth_log_returns, main = "Synthetic log-returns from ARMA model", lwd = 1.5)
    synth_log_prices <- xts(diffinv(synth_log_returns)[-1], order.by = index(synth_log_returns))
    plot(synth_log_prices, main = "Synthetic log-prices from ARMA model", lwd = 1.5)
### ARMA fitting with package **rugarch**
Now, we can estimate the parameters (which we already know):
    # specify an AR(1) model
    arma_spec = arfimaspec(mean.model = list(armaOrder = c(1,0), include.mean = TRUE))
    
    # estimate model
    arma_fit <- arfimafit(spec = arma_spec, data = synth_log_returns)
    #show(arma_fit)  # to get a huge amount of statistical details
    
    coef(arma_fit)
    #>           mu          ar1        sigma 
    #>  0.008327694 -0.888701453  0.198713996
    
    true_params
    #>    mu   ar1 sigma 
    #>  0.01 -0.90  0.20
    
    abs(coef(arma_fit) - true_params)
    #>          mu         ar1       sigma 
    #> 0.001672306 0.011298547 0.001286004
We can also study the effect of the number of samples TTT in the error of the estimation of parameters:
    # loop
    estim_coeffs_vs_T <- error_coeffs_vs_T <- NULL
    T_sweep <- ceiling(seq(100, T, length.out = 20))
    for (T_ in T_sweep) {
      arma_fit <- arfimafit(spec = arma_spec, data = synth_log_returns[1:T_])
      estim_coeffs_vs_T <- rbind(estim_coeffs_vs_T, coef(arma_fit))
      error_coeffs_vs_T <- rbind(error_coeffs_vs_T, abs(coef(arma_fit) - true_params)/true_params)
    }
    rownames(error_coeffs_vs_T) <- rownames(estim_coeffs_vs_T) <- paste("T =", T_sweep)
    
    # plots
    matplot(T_sweep, estim_coeffs_vs_T, 
            main = "Estimated ARMA coefficients", xlab = "T", ylab = "value",
            type = "b", pch = 20, col = rainbow(3))
    legend("topright", inset = 0.01, legend = colnames(estim_coeffs_vs_T), pch = 20, col = rainbow(3))
    matplot(T_sweep, 100*error_coeffs_vs_T, 
            main = "Relative error in estimated ARMA coefficients", xlab = "T", ylab = "error (%)",
            type = "b", pch = 20, col = rainbow(3))
    legend("topright", inset = 0.01, legend = colnames(error_coeffs_vs_T), pch = 20, col = rainbow(3))
### ARMA fitting with package **forecast**
As a sanity check, we will now compare the results of the two packages **forecast** and **rugarch**:
    library(rugarch)
    library(forecast)  #install.packages("forecast")
    
    # specify an AR(1) model with given coefficients and parameters
    arma_fixed_spec = arfimaspec(mean.model = list(armaOrder = c(1,0), include.mean = TRUE), 
                                 fixed.pars = list(mu = 0.005, ar1 = -0.9, sigma = 0.1))
    
    # generate one realization of length 1000
    set.seed(42)
    x <- arfimapath(arma_fixed_spec, n.sim = 1000)@path$seriesSim
    
    # specify and fit the model using package "rugarch"
    arma_spec = arfimaspec(mean.model = list(armaOrder = c(1,0), include.mean = TRUE))
    rugarch_fit <- arfimafit(spec = arma_spec, data = x)
    
    # fit the model using package "forecast"
    forecast_fit <- Arima(x, order = c(1,0,0))
    print(forecast_fit)
    #> Series: x 
    #> ARIMA(1,0,0) with non-zero mean 
    #> 
    #> Coefficients:
    #>           ar1    mean
    #>       -0.8982  0.0036
    #> s.e.   0.0139  0.0017
    #> 
    #> sigma^2 estimated as 0.01004:  log likelihood=881.6
    #> AIC=-1757.2   AICc=-1757.17   BIC=-1742.47
    
    # compare model coefficients
    print(c(coef(forecast_fit), "sigma" = sqrt(forecast_fit$sigma2)))
    #>          ar1    intercept        sigma 
    #> -0.898181148  0.003574781  0.100222964
    print(coef(rugarch_fit))
    #>           mu          ar1        sigma 
    #>  0.003605805 -0.898750138  0.100199956
Indeed, both packages give the same results.
### ARMA model selection with package **rugarch**
In the previous experiments, we implicitly assumed we knew the order of the ARMA model, i.e., p\=1p\=1p=1 and q\=0q\=0q=0. In practice, the order is unknown and one has to try different combinations of orders. The higher the order, the better the fit, but this will inevitable produce overfitting. Many methods have been developed to penalize the increase of the order complexity to avoid overfitting, e.g., AIC, BIC, SIC, HQIC, etc.
    library(rugarch)
    
    # try different combinations
    arma_fit <- autoarfima(data = synth_log_returns, 
                           ar.max = 3, ma.max = 3, include.mean = TRUE, 
                           criterion = "BIC", method = "partial")  # "AIC","BIC","SIC","HQIC"
    # see the ranking of the combinations
    arma_fit$rank.matrix
    #>    AR MA Mean ARFIMA         BIC converged
    #> 1   1  0    1      0 -0.38249098         1
    #> 2   1  1    1      0 -0.37883157         1
    #> 3   2  0    1      0 -0.37736340         1
    #> 4   1  2    1      0 -0.37503980         1
    #> 5   2  1    1      0 -0.37459177         1
    #> 6   3  0    1      0 -0.37164609         1
    #> 7   1  3    1      0 -0.37143480         1
    #> 8   2  2    1      0 -0.37107841         1
    #> 9   3  1    1      0 -0.36795491         1
    #> 10  2  3    1      0 -0.36732669         1
    #> 11  3  2    1      0 -0.36379209         1
    #> 12  3  3    1      0 -0.36058264         1
    #> 13  0  3    1      0 -0.11875575         1
    #> 14  0  2    1      0  0.02957266         1
    #> 15  0  1    1      0  0.39326050         1
    #> 16  0  0    1      0  1.17294875         1
    
    #choose the best
    armaOrder <- arma_fit$rank.matrix[1, c("AR","MA")]
    armaOrder
    #> AR MA 
    #>  1  0
In this case, the order was properly detected because the number of observations T\=1000T\=1000T=1000 is large enough. If instead, one tries with T\=200T\=200T=200, then the detected order is p\=1,q\=3p\=1,q\=3p=1, q=3.
### ARMA forecasting with package **rugarch**
Once the ARMA model parameters have been estimated, ϕ^iϕ^i\\hat{\\phi}\_i and θ^jθ^j\\hat{\\theta}\_j, one can use the model to forecast the values ahead. For example, the forecast of xtxtx\_t based on the past information is
x^t\=ϕ^0+∑i\=1pϕ^ixt−i−∑j\=1qθ^jwt−ix^t\=ϕ^0+∑i\=1pϕ^ixt−i−∑j\=1qθ^jwt−i
\\hat{x}\_t = \\hat{\\phi}\_0 + \\sum\_{i=1}^p \\hat{\\phi}\_i x\_{t-i} - \\sum\_{j=1}^q \\hat{\\theta}\_j w\_{t-i} and the forecast error will be xt−x^t\=wtxt−x^t\=wtx\_t - \\hat{x}\_t = w\_t (assuming the parameters have been perfectly estimated), which has a variance of σ2σ2\\sigma^2. The package **rugarch** makes the forecast of the out-of-sample data straightforward:
    # estimate model excluding the out of sample
    out_of_sample <- round(T/2)
    dates_out_of_sample <- tail(index(synth_log_returns), out_of_sample)
    arma_spec = arfimaspec(mean.model = list(armaOrder = c(1,0), include.mean = TRUE))
    arma_fit <- arfimafit(spec = arma_spec, data = synth_log_returns, out.sample = out_of_sample)
    coef(arma_fit)
    #>           mu          ar1        sigma 
    #>  0.007212069 -0.898745183  0.200400119
    
    # forecast log-returns along the whole out-of-sample
    arma_fore <- arfimaforecast(arma_fit, n.ahead = 1, n.roll = out_of_sample-1)
    forecast_log_returns <- xts(arma_fore@forecast$seriesFor[1, ], dates_out_of_sample)
    
    # recover log-prices
    prev_log_price <- head(tail(synth_log_prices, out_of_sample+1), out_of_sample)
    forecast_log_prices <- xts(prev_log_price + arma_fore@forecast$seriesFor[1, ], dates_out_of_sample)
    
    # plot of log-returns
    plot(cbind("fitted"   = fitted(arma_fit),
               "forecast" = forecast_log_returns,
               "original" = synth_log_returns), 
         col = c("blue", "red", "black"), lwd = c(0.5, 0.5, 2),
         main = "Forecast of synthetic log-returns", legend.loc = "topleft")
    # plot of log-prices
    plot(cbind("forecast" = forecast_log_prices,
               "original" = synth_log_prices), 
         col = c("red", "black"), lwd = c(0.5, 0.5, 2),
         main = "Forecast of synthetic log-prices", legend.loc = "topleft")
## Multivariate VARMA model
A VARMA(p,q) model on the log-returns xtxt\\mathbf{x}\_t is
xt\=ϕ0+∑i\=1pΦixt−i+wt−∑j\=1qΘjwt−ixt\=ϕ0+∑i\=1pΦixt−i+wt−∑j\=1qΘjwt−i
\\mathbf{x}\_t = \\boldsymbol{\\phi}\_0 + \\sum\_{i=1}^p \\boldsymbol{\\Phi}\_i \\mathbf{x}\_{t-i} + \\mathbf{w}\_t - \\sum\_{j=1}^q \\boldsymbol{\\Theta}\_j \\mathbf{w}\_{t-i} where wtwt\\mathbf{w}\_t is a white noise series with zero mean and covariance matrix ΣwΣw\\mathbf{\\Sigma}\_w. The parameters of the model are the vector/matrix coefficients ϕ0ϕ0\\boldsymbol{\\phi}\_0, ΦiΦi\\boldsymbol{\\Phi}\_i, ΘjΘj\\boldsymbol{\\Theta}\_j, and the noise covariance matrix ΣwΣw\\mathbf{\\Sigma}\_w.
The package **MTS** is probably a good one for VARMA modeling.
## Static comparison
Let’s start by loading the S&P500:
    library(xts)
    library(quantmod)
    
    # load S&P 500 data
    SP500_index_prices <- Ad(getSymbols("^GSPC", from = "2012-01-01", to = "2015-12-31", auto.assign = FALSE))
    colnames(SP500_index_prices) <- "SP500"
    head(SP500_index_prices)
    #>              SP500
    #> 2012-01-03 1277.06
    #> 2012-01-04 1277.30
    #> 2012-01-05 1281.06
    #> 2012-01-06 1277.81
    #> 2012-01-09 1280.70
    #> 2012-01-10 1292.08
    
    # prepare training and test data
    logprices <- log(SP500_index_prices)
    logreturns <- diff(logprices)[-1]
    T <- nrow(logreturns)
    T_trn <- round(0.7*T)
    T_tst <- T - T_trn
    logreturns_trn <- logreturns[1:T_trn]
    logreturns_tst <- logreturns[-c(1:T_trn)]
    
    # plot
    { plot(logreturns, main = "Returns", lwd = 1.5)
      addEventLines(xts("training", index(logreturns[T_trn])), srt=90, pos=2, lwd = 2, col = "blue") }
Now, we use the training data (i.e., for t\=1,…,Ttrnt\=1,…,Ttrnt=1,\\dots,T\_\\textsf{trn}) to fit different models (note that out-of-sample data is excluded by indicating `out.sample = T_tst`). In particular, we will consider the i.i.d. model, AR model, ARMA model, and also some ARCH and GARCH models (to be studied later in more detail for the variance modeling).
    library(rugarch)
    
    # fit i.i.d. model
    iid_spec <- arfimaspec(mean.model = list(armaOrder = c(0,0), include.mean = TRUE))
    iid_fit <- arfimafit(spec = iid_spec, data = logreturns, out.sample = T_tst)
    coef(iid_fit)
    #>           mu        sigma 
    #> 0.0005712982 0.0073516993
    mean(logreturns_trn)
    #> [1] 0.0005681388
    sd(logreturns_trn)
    #> [1] 0.007360208
    
    # fit AR(1) model
    ar_spec <- arfimaspec(mean.model = list(armaOrder = c(1,0), include.mean = TRUE))
    ar_fit <- arfimafit(spec = ar_spec, data = logreturns, out.sample = T_tst)
    coef(ar_fit)
    #>            mu           ar1         sigma 
    #>  0.0005678014 -0.0220185181  0.0073532716
    
    # fit ARMA(2,2) model
    arma_spec <- arfimaspec(mean.model = list(armaOrder = c(2,2), include.mean = TRUE))
    arma_fit <- arfimafit(spec = arma_spec, data = logreturns, out.sample = T_tst)
    coef(arma_fit)
    #>            mu           ar1           ar2           ma1           ma2         sigma 
    #>  0.0007223304  0.0268612636  0.9095552008 -0.0832923604 -0.9328475211  0.0072573570
    
    # fit ARMA(1,1) + ARCH(1) model
    arch_spec <- ugarchspec(mean.model = list(armaOrder = c(1,1), include.mean = TRUE), 
                            variance.model = list(model = "sGARCH", garchOrder = c(1,0)))
    arch_fit <- ugarchfit(spec = arch_spec, data = logreturns, out.sample = T_tst)
    coef(arch_fit)
    #>            mu           ar1           ma1         omega        alpha1 
    #>  6.321441e-04  8.720929e-02 -9.391019e-02  4.898885e-05  9.986975e-02
    
    # fit ARMA(0,0) + ARCH(10) model
    long_arch_spec <- ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean = TRUE), 
                                 variance.model = list(model = "sGARCH", garchOrder = c(10,0)))
    long_arch_fit <- ugarchfit(spec = long_arch_spec, data = logreturns, out.sample = T_tst)
    coef(long_arch_fit)
    #>           mu        omega       alpha1       alpha2       alpha3       alpha4       alpha5 
    #> 7.490786e-04 2.452099e-05 6.888561e-02 7.207551e-02 1.419938e-01 1.909541e-02 3.082806e-02 
    #>       alpha6       alpha7       alpha8       alpha9      alpha10 
    #> 4.026539e-02 3.050040e-07 9.260183e-02 1.150128e-01 1.068426e-06
    
    # fit ARMA(1,1) + GARCH(1,1) model
    garch_spec <- ugarchspec(mean.model = list(armaOrder = c(1,1), include.mean = TRUE), 
                             variance.model = list(model = "sGARCH", garchOrder = c(1,1)))
    garch_fit <- ugarchfit(spec = garch_spec, data = logreturns, out.sample = T_tst)
    coef(garch_fit)
    #>            mu           ar1           ma1         omega        alpha1         beta1 
    #>  6.660346e-04  9.664597e-01 -1.000000e+00  7.066506e-06  1.257786e-01  7.470725e-01
We are ready to use the different models to forecast the log-returns:
    # prepare to forecast log-returns along the out-of-sample period
    dates_out_of_sample <- tail(index(logreturns), T_tst)
    
    # forecast with i.i.d. model
    iid_fore_logreturns <- xts(arfimaforecast(iid_fit, n.ahead = 1, n.roll = T_tst - 1)@forecast$seriesFor[1, ], 
                              dates_out_of_sample)
    
    # forecast with AR(1) model
    ar_fore_logreturns <- xts(arfimaforecast(ar_fit, n.ahead = 1, n.roll = T_tst - 1)@forecast$seriesFor[1, ], 
                             dates_out_of_sample)
    
    # forecast with ARMA(2,2) model
    arma_fore_logreturns <- xts(arfimaforecast(arma_fit, n.ahead = 1, n.roll = T_tst - 1)@forecast$seriesFor[1, ], 
                               dates_out_of_sample)
    
    # forecast with ARMA(1,1) + ARCH(1) model
    arch_fore_logreturns <- xts(ugarchforecast(arch_fit, n.ahead = 1, n.roll = T_tst - 1)@forecast$seriesFor[1, ], 
                               dates_out_of_sample)
    
    # forecast with ARMA(0,0) + ARCH(10) model
    long_arch_fore_logreturns <- xts(ugarchforecast(long_arch_fit, n.ahead = 1, n.roll = T_tst - 1)@forecast$seriesFor[1, ], 
                                    dates_out_of_sample)
    
    # forecast with ARMA(1,1) + GARCH(1,1) model
    garch_fore_logreturns <- xts(ugarchforecast(garch_fit, n.ahead = 1, n.roll = T_tst - 1)@forecast$seriesFor[1, ], 
                                dates_out_of_sample)
We can compute the forecast errors (in-sample and out-of-sample) of the different models:
    error_var <- rbind("iid"                    = c(var(logreturns - fitted(iid_fit)),
                                                    var(logreturns - iid_fore_logreturns)),
                       "AR(1)"                  = c(var(logreturns - fitted(ar_fit)),
                                                    var(logreturns - ar_fore_logreturns)),
                       "ARMA(2,2)"              = c(var(logreturns - fitted(arma_fit)),
                                                    var(logreturns - arma_fore_logreturns)),
                       "ARMA(1,1) + ARCH(1)"    = c(var(logreturns - fitted(arch_fit)),
                                                    var(logreturns - arch_fore_logreturns)),
                       "ARCH(10)"               = c(var(logreturns - fitted(long_arch_fit)),
                                                    var(logreturns - long_arch_fore_logreturns)),
                       "ARMA(1,1) + GARCH(1,1)" = c(var(logreturns - fitted(garch_fit)),
                                                    var(logreturns - garch_fore_logreturns)))
    colnames(error_var) <- c("in-sample", "out-of-sample")
    print(error_var)
    #>                           in-sample out-of-sample
    #> iid                    5.417266e-05  8.975710e-05
    #> AR(1)                  5.414645e-05  9.006139e-05
    #> ARMA(2,2)              5.265204e-05  1.353213e-04
    #> ARMA(1,1) + ARCH(1)    5.415836e-05  8.983266e-05
    #> ARCH(10)               5.417266e-05  8.975710e-05
    #> ARMA(1,1) + GARCH(1,1) 5.339071e-05  9.244012e-05
We can observe that as the complexity of the model increases, the in-sample error tends to become smaller as expected (due the more degrees of freedom to fit the data), albeit the difference is neglibible. The important quantity is really the out-of-sample error: we can see that increasing the model complexity may give disappointing results. It seems that the simplest iid model is good enough in terms of error of forecast returns.
Finally, let’s show some plots of the out-of-sample error:
    error_logreturns <- cbind(logreturns - garch_fore_logreturns,
                              logreturns - long_arch_fore_logreturns,
                              logreturns - arch_fore_logreturns,
                              logreturns - arma_fore_logreturns,
                              logreturns - ar_fore_logreturns,
                              logreturns - iid_fore_logreturns)
    names(error_logreturns) <- c("GARCH", "long-ARCH", "ARCH", "ARMA", "AR", "i.i.d.")
    plot(error_logreturns, col = c("red", "green", "magenta", "purple", "blue", "black"), lwd = c(1, 1, 1, 1, 1, 2),
         main = "Out-of-sample error of static return forecast for different models", legend.loc = "bottomleft")
## Rolling-window comparison
Let’s first compare the concept of static forecast vs rolling forecast with a simple example:
    # model specification of an ARMA(2,2)
    spec <- arfimaspec(mean.model = list(armaOrder = c(2,2), include.mean = TRUE))
    
    # static fit and forecast
    ar_static_fit <- arfimafit(spec = spec, data = logreturns, out.sample = T_tst)
    ar_static_fore_logreturns <- xts(arfimaforecast(ar_static_fit, n.ahead = 1, n.roll = T_tst - 1)@forecast$seriesFor[1, ],
                                     dates_out_of_sample)
    
    # rolling fit and forecast
    modelroll <- arfimaroll(spec = spec, data = logreturns, n.ahead = 1, 
                            forecast.length = T_tst, refit.every = 50, refit.window = "moving")
    ar_rolling_fore_logreturns <- xts(modelroll@forecast$density$Mu, dates_out_of_sample)
    
    # plot of forecast
    plot(cbind("static forecast"  = ar_static_fore_logreturns,
               "rolling forecast" = ar_rolling_fore_logreturns),
         col = c("black", "red"), lwd = 2,
         main = "Forecast with ARMA(2,2) model", legend.loc = "topleft")
    
    # plot of forecast error
    error_logreturns <- cbind(logreturns - ar_static_fore_logreturns,
                              logreturns - ar_rolling_fore_logreturns)
    names(error_logreturns) <- c("rolling forecast", "static forecast")
    plot(error_logreturns, col = c("black", "red"), lwd = 2,
         main = "Forecast error with ARMA(2,2) model", legend.loc = "topleft")
Now we can redo all the forecast for all the models on a rolling-window basis:
    # rolling forecast with i.i.d. model
    iid_rolling_fore_logreturns <- xts(arfimaroll(iid_spec, data = logreturns, n.ahead = 1, forecast.length = T_tst, 
                                                  refit.every = 50, refit.window = "moving")@forecast$density$Mu, 
                                       dates_out_of_sample)
    
    # rolling forecast with AR(1) model
    ar_rolling_fore_logreturns <- xts(arfimaroll(ar_spec, data = logreturns, n.ahead = 1, forecast.length = T_tst, 
                                                 refit.every = 50, refit.window = "moving")@forecast$density$Mu, 
                                      dates_out_of_sample)
    
    # rolling forecast with ARMA(2,2) model
    arma_rolling_fore_logreturns <- xts(arfimaroll(arma_spec, data = logreturns, n.ahead = 1, forecast.length = T_tst, 
                                                   refit.every = 50, refit.window = "moving")@forecast$density$Mu, 
                                        dates_out_of_sample)
    
    # rolling forecast with ARMA(1,1) + ARCH(1) model
    arch_rolling_fore_logreturns <- xts(ugarchroll(arch_spec, data = logreturns, n.ahead = 1, forecast.length = T_tst, 
                                                   refit.every = 50, refit.window = "moving")@forecast$density$Mu, 
                                        dates_out_of_sample)
    
    # rolling forecast with ARMA(0,0) + ARCH(10) model
    long_arch_rolling_fore_logreturns <- xts(ugarchroll(long_arch_spec, data = logreturns, n.ahead = 1, forecast.length = T_tst, 
                                                        refit.every = 50, refit.window = "moving")@forecast$density$Mu, 
                                             dates_out_of_sample)
    
    # rolling forecast with ARMA(1,1) + GARCH(1,1) model
    garch_rolling_fore_logreturns <- xts(ugarchroll(garch_spec, data = logreturns, n.ahead = 1, forecast.length = T_tst, 
                                                    refit.every = 50, refit.window = "moving")@forecast$density$Mu, 
                                         dates_out_of_sample)
Let’s see the forecast errors in the rolling-basis case:
    rolling_error_var <- rbind(
      "iid"                    = c(var(logreturns - fitted(iid_fit)),
                                   var(logreturns - iid_rolling_fore_logreturns)),
      "AR(1)"                  = c(var(logreturns - fitted(ar_fit)),
                                   var(logreturns - ar_rolling_fore_logreturns)),
      "ARMA(2,2)"              = c(var(logreturns - fitted(arma_fit)),
                                   var(logreturns - arma_rolling_fore_logreturns)),
      "ARMA(1,1) + ARCH(1)"    = c(var(logreturns - fitted(arch_fit)),
                                   var(logreturns - arch_rolling_fore_logreturns)),
      "ARCH(10)"               = c(var(logreturns - fitted(long_arch_fit)),
                                   var(logreturns - long_arch_rolling_fore_logreturns)),
      "ARMA(1,1) + GARCH(1,1)" = c(var(logreturns - fitted(garch_fit)),
                                   var(logreturns - garch_rolling_fore_logreturns)))
    colnames(rolling_error_var) <- c("in-sample", "out-of-sample")
    print(rolling_error_var)
    #>                           in-sample out-of-sample
    #> iid                    5.417266e-05  8.974166e-05
    #> AR(1)                  5.414645e-05  9.038057e-05
    #> ARMA(2,2)              5.265204e-05  8.924223e-05
    #> ARMA(1,1) + ARCH(1)    5.415836e-05  8.997048e-05
    #> ARCH(10)               5.417266e-05  8.976736e-05
    #> ARMA(1,1) + GARCH(1,1) 5.339071e-05  8.895682e-05
and some plots:
    error_logreturns <- cbind(logreturns - garch_rolling_fore_logreturns,
                              logreturns - long_arch_rolling_fore_logreturns,
                              logreturns - arch_rolling_fore_logreturns,
                              logreturns - arma_rolling_fore_logreturns,
                              logreturns - ar_rolling_fore_logreturns,
                              logreturns - iid_rolling_fore_logreturns)
    names(error_logreturns) <- c("GARCH", "long-ARCH", "ARCH", "ARMA", "AR", "i.i.d.")
    plot(error_logreturns, col = c("red", "green", "magenta", "purple", "blue", "black"), lwd = c(1, 1, 1, 1, 1, 2),
         main = "Error of rolling forecast for different models", legend.loc = "topleft")
We can finally compare the static vs rolling basis errors:
    barplot(rbind(error_var[, "out-of-sample"], rolling_error_var[, "out-of-sample"]), 
            col = c("darkblue", "darkgoldenrod"), 
            legend = c("static forecast", "rolling forecast"), 
            main = "Out-of-sample forecast error for different models", 
            xlab = "method", ylab = "variance", beside = TRUE)
# Variance models
## Packages
The following R packages are widely used by the R community for volatility clustering modeling:
-   [**fGarch**](https://cran.r-project.org/package=fGarch): this is a popular package for GARCH models;
-   [**rugarch**](https://cran.r-project.org/package=rugarch): \[see [intro to rugarch](https://cran.r-project.org/package=rugarch/vignettes/Introduction_to_the_rugarch_package.pdf)\] probably the most comprehensive package with a neverending list of different ARMA and GARCH models, including different distributions for the residual, namely, Gaussian/normal distribution, Student distribution, generalized error distributions, skewed distributions, generalized hyperbolic distributions, generalized hyperbolic skewed Student distribution, Johnson’s reparameterized SU distribution, as well as utilities for simulating, forecasting, and plotting. It includes the following GARCH models:
    -   standard GARCH model (’sGARCH’)
    -   integrated GARCH model (’iGARCH’)
    -   exponential GARCH model
    -   GJR-GARCH model (’gjrGARCH’)
    -   asymmetric power ARCH model (’apARCH’)
    -   Exponential GARCH model
    -   family GARCH model (’fGARCH’)
        -   Absolute Value GARCH (AVGARCH) model (submodel = ’AVGARCH’)
        -   GJR GARCH (GJRGARCH) model (submodel = ’GJRGARCH’)
        -   Threshold GARCH (TGARCH) model (submodel = ’TGARCH’)
        -   Nonlinear ARCH model (submodel = ’NGARCH’)
        -   Nonlinear Asymmetric GARCH model (submodel = ’NAGARCH’)
        -   Asymmetric Power ARCH model (submodel = ’APARCH’)
        -   Full fGARCH model (submodel = ’ALLGARCH’)
    -   Component sGARCH model (’csGARCH’)
    -   Multiplicative Component sGARCH model (’mcsGARCH’)
    -   realized GARCH model (’realGARCH’)
    -   fractionally integrated GARCH model (’fiGARCH’);
-   [**rmgarch**](https://cran.r-project.org/package=rmgarch): extension of **rugarch** to the multivariate case. It includes the models: CCC, DCC, GARCH-Copula, GO-GARCH ([vignette](https://cran.r-project.org/package=rmgarch/vignettes/The_rmgarch_models.pdf));
-   [**MTS**](https://cran.r-project.org/package=MTS): multivariate time series modeling based on Tsay’s book;
-   [**stochvol**](https://cran.r-project.org/package=stochvol): stochastic volatility modeling based on MCMC (computationally intensive).
There are many other packages worth exploring. For a full list of packages for time series modeling, check the [CRAN Task View: Time Series Analysis](https://cran.r-project.org/view=TimeSeries).
## ARCH and GARCH models
An ARCH(mmm) model on the log-returns residuals wtwtw\_t is
wt\=σtztwt\=σtzt
w\_t = \\sigma\_t z\_t where ztztz\_t is a white noise series with zero mean and constant unit variance, and the conditional variance σ2tσt2\\sigma\_t^2 is modeled asσ2t\=ω+∑i\=1mαiw2t−iσt2\=ω+∑i\=1mαiwt−i2
\\sigma\_t^2 = \\omega + \\sum\_{i=1}^m \\alpha\_i w\_{t-i}^2 where mmm is the model order, and ω\>0,αi≥0ω\>0,αi≥0\\omega>0,\\alpha\_i\\ge0 are the parameters.
A GARCH(mmm,sss) model extends the ARCH model with a recursive term on σ2tσt2\\sigma\_t^2:
σ2t\=ω+∑i\=1mαiw2t−i+∑j\=1sβjσ2t−jσt2\=ω+∑i\=1mαiwt−i2+∑j\=1sβjσt−j2
\\sigma\_t^2 = \\omega + \\sum\_{i=1}^m \\alpha\_i w\_{t-i}^2 + \\sum\_{j=1}^s \\beta\_j \\sigma\_{t-j}^2 where the parameters ω\>0,αi≥0,βj≥0ω\>0,αi≥0,βj≥0\\omega>0,\\alpha\_i\\ge0,\\beta\_j\\ge0 need to satisfy ∑mi\=1αi+∑sj\=1βj≤1∑i\=1mαi+∑j\=1sβj≤1\\sum\_{i=1}^m \\alpha\_i + \\sum\_{j=1}^s \\beta\_j \\le 1 for stability.
### Synthetic data generation with package **rugarch**
First, we need to define the model order:
    library(rugarch)
    
    # specify an GARCH model with given coefficients and parameters
    garch_fixed_spec <- ugarchspec(mean.model = list(armaOrder = c(1,0), include.mean = TRUE), 
                                   variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                                   fixed.pars = list(mu = 0.005, ar1 = -0.9, 
                                                     omega = 0.001, alpha1 = 0.3, beta1 = 0.65))
    
    garch_fixed_spec
    #> 
    #> *---------------------------------*
    #> *       GARCH Model Spec          *
    #> *---------------------------------*
    #> 
    #> Conditional Variance Dynamics    
    #> ------------------------------------
    #> GARCH Model      : sGARCH(1,1)
    #> Variance Targeting   : FALSE 
    #> 
    #> Conditional Mean Dynamics
    #> ------------------------------------
    #> Mean Model       : ARFIMA(1,0,0)
    #> Include Mean     : TRUE 
    #> GARCH-in-Mean        : FALSE 
    #> 
    #> Conditional Distribution
    #> ------------------------------------
    #> Distribution :  norm 
    #> Includes Skew    :  FALSE 
    #> Includes Shape   :  FALSE 
    #> Includes Lambda  :  FALSE
    
    garch_fixed_spec@model$pars
    #>           Level Fixed Include Estimate LB UB
    #> mu        0.005     1       1        0 NA NA
    #> ar1      -0.900     1       1        0 NA NA
    #> ma        0.000     0       0        0 NA NA
    #> arfima    0.000     0       0        0 NA NA
    #> archm     0.000     0       0        0 NA NA
    #> mxreg     0.000     0       0        0 NA NA
    #> omega     0.001     1       1        0 NA NA
    #> alpha1    0.300     1       1        0 NA NA
    #> beta1     0.650     1       1        0 NA NA
    #> gamma     0.000     0       0        0 NA NA
    #> eta1      0.000     0       0        0 NA NA
    #> eta2      0.000     0       0        0 NA NA
    #> delta     0.000     0       0        0 NA NA
    #> lambda    0.000     0       0        0 NA NA
    #> vxreg     0.000     0       0        0 NA NA
    #> skew      0.000     0       0        0 NA NA
    #> shape     0.000     0       0        0 NA NA
    #> ghlambda  0.000     0       0        0 NA NA
    #> xi        0.000     0       0        0 NA NA
    
    garch_fixed_spec@model$fixed.pars
    #> $mu
    #> [1] 0.005
    #> 
    #> $ar1
    #> [1] -0.9
    #> 
    #> $omega
    #> [1] 0.001
    #> 
    #> $alpha1
    #> [1] 0.3
    #> 
    #> $beta1
    #> [1] 0.65
    
    true_params <- unlist(garch_fixed_spec@model$fixed.pars)
    true_params
    #>     mu    ar1  omega alpha1  beta1 
    #>  0.005 -0.900  0.001  0.300  0.650
Then, we can generate one realization of the returns time series path:
    # simulate one path
    T <- 2000
    set.seed(42)
    path_garch <- ugarchpath(garch_fixed_spec, n.sim = T)
    
    str(path_garch@path$seriesSim)
    #>  num [1:2000, 1] 0.167 -0.217 0.248 -0.148 0.182 ...
    
    # plot log-returns
    synth_log_returns <- xts(path_garch@path$seriesSim, order.by = as.Date("2010-01-01") + 0:(T-1))
    synth_volatility <- xts(path_garch@path$sigmaSim, order.by = as.Date("2010-01-01") + 0:(T-1))
    { plot(synth_log_returns, main = "Synthetic log-returns from GARCH model", lwd = 1.5)
      lines(synth_volatility, col = "red", lwd = 2) }
### GARCH fitting with package **rugarch**
Now, we can estimate the parameters (which we already know):
    # specify a GARCH model
    garch_spec <- ugarchspec(mean.model = list(armaOrder = c(1,0), include.mean = TRUE), 
                             variance.model = list(model = "sGARCH", garchOrder = c(1,1)))
    
    # estimate model
    garch_fit <- ugarchfit(spec = garch_spec, data = synth_log_returns)
    coef(garch_fit)
    #>            mu           ar1         omega        alpha1         beta1 
    #>  0.0036510100 -0.8902333595  0.0008811434  0.2810460728  0.6717486402
    
    true_params
    #>     mu    ar1  omega alpha1  beta1 
    #>  0.005 -0.900  0.001  0.300  0.650
    
    # error in coefficients
    abs(coef(garch_fit) - true_params)
    #>           mu          ar1        omega       alpha1        beta1 
    #> 0.0013489900 0.0097666405 0.0001188566 0.0189539272 0.0217486402
We can also study the effect of the number of samples TTT in the error of the estimation of parameters:
    # loop
    estim_coeffs_vs_T <- error_coeffs_vs_T <- NULL
    T_sweep <- ceiling(seq(100, T, length.out = 20))
    for (T_ in T_sweep) {
      garch_fit <- ugarchfit(spec = garch_spec, data = synth_log_returns[1:T_])
      error_coeffs_vs_T <- rbind(error_coeffs_vs_T, abs((coef(garch_fit) - true_params)/true_params))
      estim_coeffs_vs_T <- rbind(estim_coeffs_vs_T, coef(garch_fit))
    }
    rownames(error_coeffs_vs_T) <- rownames(estim_coeffs_vs_T) <- paste("T =", T_sweep)
    
    # plots
    matplot(T_sweep, 100*error_coeffs_vs_T, 
            main = "Relative error in estimated GARCH coefficients", xlab = "T", ylab = "error (%)",
            type = "b", pch = 20, col = rainbow(5), ylim=c(-10,250))
    legend("topright", inset = 0.01, legend = colnames(error_coeffs_vs_T), pch = 20, col = rainbow(5))
### GARCH fitting with package **fGarch**
As a sanity check, we will now compare the results of the two packages **fGarch** and **rugarch**:
    library(rugarch)
    library(fGarch)
    
    # specify ARMA(0,0)-GARCH(1,1) with particular parameter values as the data generating process
    garch_fixed_spec <- ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean = TRUE),
                                   variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                                   fixed.pars = list(mu = 0.1, 
                                                     omega = 0.01, alpha1 = 0.1, beta1 = 0.8))
    
    # generate one realization of length 1000
    set.seed(237)
    x <- ugarchpath(garch_fixed_spec, n.sim = 1000)@path$seriesSim
    
    # specify and fit the model using package "rugarch"
    garch_spec = ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean = TRUE),
                            variance.model = list(garchOrder = c(1,1)))
    rugarch_fit <- ugarchfit(spec = garch_spec, data = x)
    
    # fit the model using package "fGarch"
    fGarch_fit <- fGarch::garchFit(formula = ~ garch(1, 1), data = x, trace = FALSE)
    #> Warning: Using formula(x) is deprecated when x is a character vector of length > 1.
    #>   Consider formula(paste(x, collapse = " ")) instead.
    
    # compare model coefficients
    print(coef(fGarch_fit))
    #>         mu      omega     alpha1      beta1 
    #> 0.09749904 0.01395109 0.13510445 0.73938595
    print(coef(rugarch_fit))
    #>         mu      omega     alpha1      beta1 
    #> 0.09750394 0.01392648 0.13527024 0.73971658
    
    # compare fitted standard deviations
    print(head(fGarch_fit@sigma.t))
    #> [1] 0.3513549 0.3254788 0.3037747 0.2869034 0.2735266 0.2708994
    print(head(rugarch_fit@fit$sigma))
    #> [1] 0.3538569 0.3275037 0.3053974 0.2881853 0.2745264 0.2716555
Indeed, both packages give the same results.
### GARCH forecasting with package **rugarch**
Once the parameters of the GARCH model have been estimated, one can use the model to forecast the values ahead. For example, the one-step forecast of the conditional variace σ2tσt2\\sigma\_t^2 based on the past information is
σ^2t\=ω^+∑i\=1mα^iw2t−i+∑j\=1sβ^jσ^2t−jσ^t2\=ω^+∑i\=1mα^iwt−i2+∑j\=1sβ^jσ^t−j2
\\hat{\\sigma}\_t^2 = \\hat{\\omega} + \\sum\_{i=1}^m \\hat{\\alpha}\_i w\_{t-i}^2 + \\sum\_{j=1}^s \\hat{\\beta}\_j \\hat{\\sigma}\_{t-j}^2 with unconditional variance given my ω^/(1−∑mi\=1α^i−∑sj\=1β^j)ω^/(1−∑i\=1mα^i−∑j\=1sβ^j)\\hat{\\omega}/(1 - \\sum\_{i=1}^m \\hat{\\alpha}\_i - \\sum\_{j=1}^s \\hat{\\beta}\_j). The package **rugarch** makes the forecast of the out-of-sample data straightforward:
    # estimate model excluding the out-of-sample
    out_of_sample <- round(T/2)
    dates_out_of_sample <- tail(index(synth_log_returns), out_of_sample)
    garch_spec <- ugarchspec(mean.model = list(armaOrder = c(1,0), include.mean = TRUE), 
                             variance.model = list(model = "sGARCH", garchOrder = c(1,1)))
    garch_fit <- ugarchfit(spec = garch_spec, data = synth_log_returns, out.sample = out_of_sample)
    coef(garch_fit)
    #>            mu           ar1         omega        alpha1         beta1 
    #>  0.0034964331 -0.8996287630  0.0006531088  0.3058756796  0.6815452241
    
    # forecast log-returns along the whole out-of-sample
    garch_fore <- ugarchforecast(garch_fit, n.ahead = 1, n.roll = out_of_sample-1)
    forecast_log_returns <- xts(garch_fore@forecast$seriesFor[1, ], dates_out_of_sample)
    forecast_volatility <- xts(garch_fore@forecast$sigmaFor[1, ], dates_out_of_sample)
    
    # plot of log-returns
    plot(cbind("fitted"   = fitted(garch_fit),
               "forecast" = forecast_log_returns,
               "original" = synth_log_returns), 
         col = c("blue", "red", "black"), lwd = c(0.5, 0.5, 2),
         main = "Forecast of synthetic log-returns", legend.loc = "topleft")
    # plot of volatility log-returns
    plot(cbind("fitted volatility"   = sigma(garch_fit),
               "forecast volatility" = forecast_volatility,
               "log-returns"         = synth_log_returns), 
         col = c("blue", "red", "black"), lwd = c(2, 2, 1),
         main = "Forecast of volatility of synthetic log-returns", legend.loc = "topleft")
## Envelope from different methods
Let’s start by loading the S&P500:
    library(xts)
    library(quantmod)
    
    # load S&P 500 data
    SP500_index_prices <- Ad(getSymbols("^GSPC", from = "2008-01-01", to = "2013-12-31", auto.assign = FALSE))
    colnames(SP500_index_prices) <- "SP500"
    head(SP500_index_prices)
    #>              SP500
    #> 2008-01-02 1447.16
    #> 2008-01-03 1447.16
    #> 2008-01-04 1411.63
    #> 2008-01-07 1416.18
    #> 2008-01-08 1390.19
    #> 2008-01-09 1409.13
    
    # prepare training and test data
    logprices <- log(SP500_index_prices)
    x <- diff(logprices)[-1]
    T <- nrow(x)
    T_trn <- round(0.7*T)
    T_tst <- T - T_trn
    x_trn <- x[1:T_trn]
    x_tst <- x[-c(1:T_trn)]
    
    # plot
    { plot(x, main = "Returns", lwd = 1.5)
      addEventLines(xts("training", index(x[T_trn])), srt=90, pos=2, lwd = 2, col = "blue") }
### Constant
Let’s start with the constant envelope:
    var_constant <- var(x_trn)  # or: mean(x_trn^2)
    plot(cbind(sqrt(var_constant), x_trn), col = c("red", "black"), lwd = c(2.5, 1.5),
         main = "Constant envelope")
### MA
Now, let’s use a simple rolling means (aka moving average) of the squared returns:
y^t\=1m∑i\=1myt−iy^t\=1m∑i\=1myt−i
\\hat{y}\_{t}=\\frac{1}{m}\\sum\_{i=1}^{m}y\_{t-i} with yt\=x2tyt\=xt2y\_t=x\_t^2 (we could have used yt\=(xt−μt)2yt\=(xt−μt)2y\_t=(x\_t - \\mu\_t)^2 but the difference is negligible).
    library(RcppRoll)  # fast rolling means
    lookback_var <- 20
    var_t <- roll_meanr(x_trn^2, n = lookback_var, fill = NA)
    plot(cbind(sqrt(var_t), x_trn), col = c("red", "black"), lwd = c(2.5, 1.5),
         main = "Envelope based on simple rolling means of squares (lookback=20)")
    x_trn_std <- x_trn/sqrt(var_t)
    var_ma <- var(x_trn_std, na.rm = TRUE) * tail(var_t, 1)
### EWMA
A more adaptive version is the exponentially weighted moving average (EWMA):
y^t\=αyt−1+(1−α)y^t−1y^t\=αyt−1+(1−α)y^t−1
\\hat{y}\_{t}=\\alpha y\_{t-1}+(1-\\alpha)\\hat{y}\_{t-1} with yt\=x2tyt\=xt2y\_t=x\_t^2. Note that this can also be modeled in component form as an ETS(A,N,N) innovations state space model:
ytℓt\=ℓt−1+εt\=ℓt−1+αεt.yt\=ℓt−1+εtℓt\=ℓt−1+αεt.
\\begin{aligned} y\_{t} &= \\ell\_{t-1} + \\varepsilon\_t\\\\ \\ell\_{t} &= \\ell\_{t-1} + \\alpha\\varepsilon\_t. \\end{aligned}
    library(forecast)
    fit_ets <- ets(x_trn^2, model = "ANN")
    std_t <- as.numeric(sqrt(fit_ets$fitted))
    plot(cbind(std_t, x_trn), col = c("red", "black"), lwd = c(2.5, 1.5),
         main = "Envelope based on EWMA of squares")
    x_trn_std <- x_trn/std_t
    var_ewma <- var(x_trn_std, na.rm = TRUE) * tail(std_t, 1)^2
### Multiplicative ETS
We can also try different variations of the ETS models. For example, the multiplicative noise version ETS(M,N,N), with innovations state space model:
ytℓt\=ℓt−1(1+εt)\=ℓt−1(1+αεt).yt\=ℓt−1(1+εt)ℓt\=ℓt−1(1+αεt).
\\begin{aligned} y\_{t} &= \\ell\_{t-1}(1 + \\varepsilon\_t)\\\\ \\ell\_{t} &= \\ell\_{t-1}(1 + \\alpha\\varepsilon\_t). \\end{aligned}
    fit_ets <- ets(1e-6 + x_trn^2, model = "MNN")
    std_t <- as.numeric(sqrt(fit_ets$fitted))
    plot(cbind(std_t, x_trn), col = c("red", "black"), lwd = c(2.5, 1.5),
         main = "Envelope based on ETS(M,N,N) of squares")
    x_trn_std <- x_trn/std_t
    var_ets_mnn <- var(x_trn_std, na.rm = TRUE) * tail(std_t, 1)^2
### ARCH
We can now use the more sophisticated ARCH modeling:
wtσ2t\=σtzt\=ω+∑i\=1mαiw2t−i.wt\=σtztσt2\=ω+∑i\=1mαiwt−i2.
\\begin{aligned} w\_t &= \\sigma\_t z\_t\\\\ \\sigma\_t^2 &= \\omega + \\sum\_{i=1}^m \\alpha\_i w\_{t-i}^2. \\end{aligned}
    library(fGarch)
    arch_fit <- fGarch::garchFit(formula = ~ garch(5,0), x_trn, trace = FALSE)
    #> Warning: Using formula(x) is deprecated when x is a character vector of length > 1.
    #>   Consider formula(paste(x, collapse = " ")) instead.
    std_t <- arch_fit@sigma.t
    plot(cbind(std_t, x_trn), col = c("red", "black"), lwd = c(2.5, 1.5),
         main = "Envelope based on ARCH(5)")
    var_arch <- tail(std_t, 1)^2
### GARCH
We can step up our model to a GARCH:
wtσ2t\=σtzt\=ω+∑i\=1mαiw2t−i+∑j\=1sβjσ2t−j.wt\=σtztσt2\=ω+∑i\=1mαiwt−i2+∑j\=1sβjσt−j2.
\\begin{aligned} w\_t &= \\sigma\_t z\_t\\\\ \\sigma\_t^2 &= \\omega + \\sum\_{i=1}^m \\alpha\_i w\_{t-i}^2 + \\sum\_{j=1}^s \\beta\_j \\sigma\_{t-j}^2. \\end{aligned}
    garch_fit <- fGarch::garchFit(formula = ~ garch(1,1), x_trn, trace = FALSE)
    #> Warning: Using formula(x) is deprecated when x is a character vector of length > 1.
    #>   Consider formula(paste(x, collapse = " ")) instead.
    std_t <- garch_fit@sigma.t
    plot(cbind(std_t, x_trn), col = c("red", "black"), lwd = c(2.5, 1.5),
         main = "Envelope based on GARCH(1,1)")
    var_garch <- tail(std_t, 1)^2
### SV
Finally, we can use the stochastic volatility modeling:
wtht−h¯\=exp(ht/2)zt\=ϕ(ht−1−h¯)+ut.wt\=exp⁡(ht/2)ztht−h¯\=ϕ(ht−1−h¯)+ut.
\\begin{aligned} w\_{t} &= \\exp\\left(h\_{t}/2\\right)z\_{t}\\\\ h\_{t}-\\bar{h} &= \\phi\\left(h\_{t-1}-\\bar{h}\\right)+u\_{t}. \\end{aligned} or, equivalently,
wtlog(σ2t)\=σtzt\=h¯+ϕ(log(σ2t−1)−h¯)+ut.wt\=σtztlog⁡(σt2)\=h¯+ϕ(log⁡(σt−12)−h¯)+ut.
\\begin{aligned} w\_{t} &= \\sigma\_{t}z\_{t}\\\\ \\log\\left(\\sigma\_{t}^{2}\\right) &= \\bar{h}+\\phi\\left(\\log\\left(\\sigma\_{t-1}^{2}\\right)-\\bar{h}\\right)+u\_{t}. \\end{aligned}
    library(stochvol)
    res <- svsample(x_trn - mean(x_trn), priormu = c(0, 100), priornu = c(4, 100))
    #summary(res, showlatent = FALSE)
    std_t <- res$summary$sd[, 1]
    plot(cbind(std_t, x_trn), col = c("red", "black"), lwd = c(2.5, 1.5),
         main = "Envelope based on stochastic volatility")
    var_sv <- tail(std_t, 1)^2
### Comparison
We can now compare the error in the estimation of the variance by each method for the out-of-sample period (however, this may not be a representative comparison, just anecdotic):
    error_all <- c("MA"         = abs(var_ma      - var(x_tst)),
                   "EWMA"       = abs(var_ewma    - var(x_tst)),
                   "ETS(M,N,N)" = abs(var_ets_mnn - var(x_tst)),
                   "ARCH(5)"    = abs(var_arch    - var(x_tst)),
                   "GARCH(1,1)" = abs(var_garch   - var(x_tst)),
                   "SV"         = abs(var_sv      - var(x_tst)))
    print(error_all)
    #>           MA         EWMA   ETS(M,N,N)      ARCH(5)   GARCH(1,1)           SV 
    #> 2.204965e-05 7.226188e-06 3.284057e-06 7.879039e-05 6.496545e-06 6.705059e-06
    barplot(error_all, main = "Error in estimation of out-of-sample variance", col = rainbow(6))
## Rolling-window comparison
Rolling-window comparison of six methods: MA, EWMA, ETS(MNN), ARCH(5), GARCH(1,1), and SV.
    error_sv <- error_garch <- error_arch <- error_ets_mnn <- error_ewma <- error_ma <- NULL
      
    # rolling window
    lookback <- 200
    len_tst <- 40
    for (i in seq(lookback, T-len_tst, by = len_tst)) {
      x_trn <- x[(i-lookback+1):i]
      x_tst <- x[(i+1):(i+len_tst)]
      var_tst <- var(x_tst)
      
      # MA
      var_t <- roll_meanr(x_trn^2, n = 20, fill = NA)
      var_fore <- var(x_trn/sqrt(var_t), na.rm = TRUE) * tail(var_t, 1)
      error_ma <- c(error_ma, abs(var_fore - var_tst))
      
      # EWMA
      fit_ets <- ets(x_trn^2, model = "ANN")
      std_t <- as.numeric(sqrt(fit_ets$fitted))
      var_fore <- var(x_trn/std_t, na.rm = TRUE) * tail(std_t, 1)^2
      error_ewma <- c(error_ewma, abs(var_fore - var_tst))
      
      # ETS(M,N,N)
      fit_ets <- ets(1e-6 + x_trn^2, model = "MNN")
      std_t <- as.numeric(sqrt(fit_ets$fitted))
      var_fore <- var(x_trn/std_t, na.rm = TRUE) * tail(std_t, 1)^2
      error_ets_mnn <- c(error_ets_mnn, abs(var_fore - var_tst))
      
      # ARCH
      arch_fit <- fGarch::garchFit(formula = ~ garch(5,0), x_trn, trace = FALSE)
      std_t <- as.numeric(arch_fit@sigma.t)
      var_fore <- var(x_trn/std_t, na.rm = TRUE) * tail(std_t, 1)^2
      error_arch <- c(error_arch, abs(var_fore - var_tst))
      
      # GARCH
      garch_fit <- fGarch::garchFit(formula = ~ garch(1,1), x_trn, trace = FALSE)
      std_t <- as.numeric(garch_fit@sigma.t)
      var_fore <- var(x_trn/std_t, na.rm = TRUE) * tail(std_t, 1)^2
      error_garch <- c(error_garch, abs(var_fore - var_tst))
      
      # SV
      res <- svsample(x_trn - mean(x_trn), priormu = c(0, 100), priornu = c(4, 100))
      std_t <- res$summary$sd[, 1]
      var_fore <- var(x_trn/std_t, na.rm = TRUE) * tail(std_t, 1)^2
      error_sv <- c(error_sv, abs(var_fore - var_tst))
    }
    
    error_all <- c("MA"         = mean(error_ma),
                   "EWMA"       = mean(error_ewma),
                   "ETS(M,N,N)" = mean(error_ets_mnn),
                   "ARCH(5)"    = mean(error_arch),
                   "GARCH(1,1)" = mean(error_garch),
                   "SV"         = mean(error_sv))
    print(error_all)
    #>           MA         EWMA   ETS(M,N,N)      ARCH(5)   GARCH(1,1)           SV 
    #> 1.786851e-04 2.218130e-04 2.081554e-04 1.690483e-04 1.642066e-04 9.588908e-05
    barplot(error_all, main = "Error in estimation of variance", col = rainbow(6))
## Multivariate GARCH models
We will consider only the constant conditional correlation (CCC) and dynamic conditional correlation (DCC) models for illustration purposes since they are the most popular ones. The log-returns residuals wtwtw\_t are modeled as
wt\=Σ1/2tztwt\=Σt1/2zt
\\mathbf{w}\_t = \\boldsymbol{\\Sigma}\_t^{1/2} \\mathbf{z}\_t where ztzt\\mathbf{z}\_t is an i.i.d. white noise series with zero mean and constant covariance matrix II\\mathbf{I}. The conditional covariance matrix ΣtΣt\\boldsymbol{\\Sigma}\_t is modeled asΣt\=DtCDtΣt\=DtCDt
\\boldsymbol{\\Sigma}\_t = \\mathbf{D}\_t\\mathbf{C}\\mathbf{D}\_t where Dt\=Diag(σ1,t,…,σN,t)Dt\=Diag(σ1,t,…,σN,t)\\mathbf{D}\_t = \\textsf{Diag}(\\sigma\_{1,t},\\dots,\\sigma\_{N,t}) and mathbfCmathbfCmathbf{C} is the CCC covariance matrix of the standardized noise vector ηt\=C−1wtηt\=C−1wt\\boldsymbol{\\eta}\_t = \\mathbf{C}^{-1}\\mathbf{w}\_t (i.e., it contains diagonal elements equal to 1).
Basically, with this model, the diagonal matrix DtDt\\mathbf{D}\_t contains a set of univariate GARCH models and then the matrix CC\\mathbf{C} incorporates some correlation among the assets. The main drawback of this model is that the matrix CC\\mathbf{C} is constant. To overcome this, the DCC was proposed as
Σt\=DtCtDtΣt\=DtCtDt
\\boldsymbol{\\Sigma}\_t = \\mathbf{D}\_t\\mathbf{C}\_t\\mathbf{D}\_t where CtCt\\mathbf{C}\_t contains diagonal elements equal to 1. To enforce diagonal elements equal to 1, Engle modeled it asCt\=Diag−1/2(Qt)QtDiag−1/2(Qt)Ct\=Diag−1/2(Qt)QtDiag−1/2(Qt)
\\mathbf{C}\_t = \\textsf{Diag}^{-1/2}(\\mathbf{Q}\_t) \\mathbf{Q}\_t \\textsf{Diag}^{-1/2}(\\mathbf{Q}\_t) where QtQt\\mathbf{Q}\_t has arbitrary diagonal elements and follows the model
Qt\=αηtηTt+(1−α)Qt−1.Qt\=αηtηtT+(1−α)Qt−1.
\\mathbf{Q}\_t = \\alpha \\boldsymbol{\\eta}\_t\\boldsymbol{\\eta}\_t^T + (1-\\alpha)\\mathbf{Q}\_{t-1}.
We will use the package **rmgarch**, which is authored by the same author as **rugarch**, and is used in the same way, to generate synthetic data, estimate the parameters, and forecast.
Let’s start by loading some multivariate ETF data:
-   SPY: SPDR S&P 500 ETF Trust
-   TLE: 20+ Year Treasury Bond ETF
-   IEF: 7-10 Year Treasury Bond ETF
    library(quantmod)
    stock_namelist <- c("SPY", "TLT", "IEF")
    
    # download data from YahooFinance
    prices <- xts()
    for (stock_index in 1:length(stock_namelist))
      prices <- cbind(prices, Ad(getSymbols(stock_namelist[stock_index], 
                                            from = "2013-01-01", to = "2016-12-31", auto.assign = FALSE)))
    colnames(prices) <- stock_namelist
    indexClass(prices) <- "Date"
    #> Warning: 'indexClass<-' is deprecated.
    #> Use 'tclass<-' instead.
    #> See help("Deprecated") and help("xts-deprecated").
    logreturns <- diff(log(prices))[-1]
    head(prices)
    #>                 SPY      TLT      IEF
    #> 2013-01-02 125.4519 98.26711 92.50271
    #> 2013-01-03 125.1685 96.93565 92.02721
    #> 2013-01-04 125.7181 97.31373 92.07043
    #> 2013-01-07 125.3746 97.35482 92.12228
    #> 2013-01-08 125.0139 97.99590 92.34708
    #> 2013-01-09 125.3316 97.90547 92.39889
    
    # plot the three series of log-prices
    plot(log(prices), col = c("black", "blue", "magenta"),
         main = "Log-prices of the three ETFs", legend.loc = "topleft")
First, we define the model:
    library(rmgarch)  #install.packages("rmgarch")
    
    # specify i.i.d. model for the univariate time series
    ugarch_spec <- ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean = FALSE), 
                              variance.model = list(model = "sGARCH", garchOrder = c(1,1)))
    
    # specify DCC model
    dcc_spec <- dccspec(uspec = multispec(replicate(ugarch_spec, n = 3)),
                        VAR = TRUE, lag = 3,
                        model = "DCC", dccOrder = c(1,1))
Next, we fit the model:
    # estimate model
    garchdcc_fit <- dccfit(dcc_spec, data = logreturns, solver = "nlminb")
    #> Warning in .sgarchfit(spec = spec, data = data, out.sample = out.sample, : 
    #> ugarchfit-->warning: solver failer to converge.
    garchdcc_fit
    #> 
    #> *---------------------------------*
    #> *          DCC GARCH Fit          *
    #> *---------------------------------*
    #> 
    #> Distribution         :  mvnorm
    #> Model                :  DCC(1,1)
    #> No. Parameters       :  44
    #> [VAR GARCH DCC UncQ] : [30+9+2+3]
    #> No. Series           :  3
    #> No. Obs.             :  1007
    #> Log-Likelihood       :  12219.78
    #> Av.Log-Likelihood    :  12.13 
    #> 
    #> Optimal Parameters
    #> -----------------------------------
    #>               Estimate  Std. Error   t value Pr(>|t|)
    #> [SPY].omega   0.000007    0.000001   9.98320 0.000000
    #> [SPY].alpha1  0.197691    0.023754   8.32256 0.000000
    #> [SPY].beta1   0.690058    0.029803  23.15368 0.000000
    #> [TLT].omega   0.000001    0.000001   0.90905 0.363323
    #> [TLT].alpha1  0.019717    0.010308   1.91283 0.055770
    #> [TLT].beta1   0.963759    0.006468 149.01419 0.000000
    #> [IEF].omega   0.000000    0.000001   0.45508 0.649049
    #> [IEF].alpha1  0.031740    0.023594   1.34526 0.178540
    #> [IEF].beta1   0.937779    0.016669  56.25761 0.000000
    #> [Joint]dcca1  0.039814    0.018571   2.14390 0.032041
    #> [Joint]dccb1  0.854229    0.083315  10.25296 0.000000
    #> 
    #> Information Criteria
    #> ---------------------
    #>                     
    #> Akaike       -24.182
    #> Bayes        -23.968
    #> Shibata      -24.186
    #> Hannan-Quinn -24.101
    #> 
    #> 
    #> Elapsed time : 1.491833
We can plot the time-varying correlations:
    # extract time-varying covariance and correlation matrix
    dcc_cor <- rcor(garchdcc_fit)
    dim(dcc_cor)
    #> [1]    3    3 1007
    
    #plot
    corr_t <- xts(cbind(dcc_cor[1, 2, ], dcc_cor[1, 3, ], dcc_cor[2, 3, ]), order.by = index(logreturns))
    colnames(corr_t) <- c("SPY vs LTL", "SPY vs IEF", "TLT vs IEF")
    plot(corr_t, col = c("black", "red", "blue"),
         main = "Time-varying correlations", legend.loc = "left")
 