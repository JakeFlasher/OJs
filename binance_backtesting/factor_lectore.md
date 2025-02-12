Factor Models with R
Prof. Daniel P. Palomar
MAFS5310 - Portfolio Optimization with R
MSc in Financial Mathematics
The Hong Kong University of Science and Technology (HKUST)
Fall 2020-21
This R session will illustrate the practical implementation and use of factor models.

The R package covFactorModel, available in GitHub, is recommended.

(Useful R links: Cookbook R, Quick-R, R documentation, CRAN, METACRAN.)

Macroeconomic factor model with single market factor
We will start with a simple example consisting of a single known factor (i.e., the market index). The model is
xt=α+βft+ϵt,t=1,…,T
where the explicit factor ft
 is the S&P 500 index. We will do a simple least squares (LS) regression to estimate the intercept α
 and the loading beta β
:
minimizeα,β∑t=1T∥xt−α−βft∥2

Most the code lines go into preparing the data rather than actually performing the factor modeling. Let’s start getting the data ready:

library(xts)
library(quantmod)

# set begin-end date and stock namelist
begin_date <- "2016-01-01"
end_date <- "2017-12-31"
stock_namelist <- c("AAPL", "AMD", "ADI",  "ABBV", "AEZS", "A",  "APD", "AA","CF")
sector_namelist <- c(rep("Information Technology", 3), rep("Health Care", 3), rep("Materials", 3))

# download data from YahooFinance
data_set <- xts()
for (stock_index in 1:length(stock_namelist))
  data_set <- cbind(data_set, Ad(getSymbols(stock_namelist[stock_index], 
                                            from = begin_date, to = end_date, auto.assign = FALSE)))
colnames(data_set) <- stock_namelist
indexClass(data_set) <- "Date"
#> Warning: 'indexClass<-' is deprecated.
#> Use 'tclass<-' instead.
#> See help("Deprecated") and help("xts-deprecated").
str(data_set)
#> An 'xts' object on 2016-01-04/2017-12-29 containing:
#>   Data: num [1:503, 1:9] 24.4 23.8 23.4 22.4 22.5 ...
#>  - attr(*, "dimnames")=List of 2
#>   ..$ : NULL
#>   ..$ : chr [1:9] "AAPL" "AMD" "ADI" "ABBV" ...
#>   Indexed by objects of class: [Date] TZ: UTC
#>   xts Attributes:  
#>  NULL
head(data_set)
#>                AAPL  AMD      ADI     ABBV AEZS        A       APD       AA       CF
#> 2016-01-04 24.44304 2.77 48.96699 46.14122 4.40 39.00735 105.02506 23.00764 34.31850
#> 2016-01-05 23.83051 2.75 48.60722 45.94900 4.21 38.87313 103.14715 21.96506 33.24234
#> 2016-01-06 23.36416 2.51 46.53844 45.95699 3.64 39.04568 100.63515 20.40121 30.36976
#> 2016-01-07 22.37808 2.28 45.35113 45.82084 3.29 37.38723  97.26139 19.59558 28.92924
#> 2016-01-08 22.49641 2.14 44.95538 44.57140 3.29 36.99419  96.75736 19.12169 28.65807
#> 2016-01-11 22.86068 2.34 46.02574 43.15377 3.13 36.37106  97.13944 18.95583 27.49718
tail(data_set)
#>                AAPL   AMD      ADI     ABBV AEZS        A      APD    AA       CF
#> 2017-12-21 42.17631 10.89 83.78490 84.42201 2.70 65.88075 151.9022 48.99 37.97728
#> 2017-12-22 42.17631 10.54 83.97394 84.68070 2.45 65.71491 152.1442 49.99 38.40079
#> 2017-12-26 41.10630 10.46 83.76600 84.28406 2.33 65.61732 151.7532 50.38 39.45034
#> 2017-12-27 41.11353 10.53 84.21021 84.57722 2.36 65.66613 152.3770 51.84 39.65289
#> 2017-12-28 41.22921 10.55 84.47484 84.31854 2.41 65.81248 153.2614 54.14 39.22938
#> 2017-12-29 40.78337 10.28 84.14406 83.38734 2.36 65.48880 153.6454 53.87 39.16494

SP500_index <- Ad(getSymbols("^GSPC", from = begin_date, to = end_date, auto.assign = FALSE))
colnames(SP500_index) <- "index"
head(SP500_index)
#>              index
#> 2016-01-04 2012.66
#> 2016-01-05 2016.71
#> 2016-01-06 1990.26
#> 2016-01-07 1943.09
#> 2016-01-08 1922.03
#> 2016-01-11 1923.67
plot(SP500_index)


# compute the log-returns of the stocks and of the SP500 index as explicit factor
X <- diff(log(data_set), na.pad = FALSE)
N <- ncol(X)  # number of stocks
T <- nrow(X)  # number of days
f <- diff(log(SP500_index), na.pad = FALSE)
str(X)
#> An 'xts' object on 2016-01-05/2017-12-29 containing:
#>   Data: num [1:502, 1:9] -0.02538 -0.01976 -0.04312 0.00527 0.01606 ...
#>  - attr(*, "dimnames")=List of 2
#>   ..$ : NULL
#>   ..$ : chr [1:9] "AAPL" "AMD" "ADI" "ABBV" ...
#>   Indexed by objects of class: [Date] TZ: UTC
#>   xts Attributes:  
#>  NULL
str(f)
#> An 'xts' object on 2016-01-05/2017-12-29 containing:
#>   Data: num [1:502, 1] 0.00201 -0.013202 -0.023986 -0.010898 0.000853 ...
#>  - attr(*, "dimnames")=List of 2
#>   ..$ : NULL
#>   ..$ : chr "index"
#>   Indexed by objects of class: [Date] TZ: UTC
#>   xts Attributes:  
#> List of 2
#>  $ src    : chr "yahoo"
#>  $ updated: POSIXct[1:1], format: "2020-10-04 13:58:40"
Now we are ready to do the factor model fitting. The solution to the previous LS fitting is
β^α^ϵ^iσ^2iΣ^=cov(xt,ft)/var(ft)=x¯−β^f¯=xi−αi1−βif,i=1,…,N=1T−2ϵ^Tiϵ^i,Ψ^=diag(σ^21,…,σ^2N)=var(ft)β^β^T+Ψ^

which can be readily implemented in R as follows:

beta <- cov(X,f)/as.numeric(var(f))
alpha <- colMeans(X) - beta*colMeans(f)
sigma2 <- rep(NA, N)
for (i in 1:N) {
  eps_i <- X[, i] - alpha[i] - beta[i]*f
  sigma2[i] <- (1/(T-2)) * t(eps_i) %*% eps_i
}
Psi <- diag(sigma2)
Sigma <- as.numeric(var(f)) * beta %*% t(beta) + Psi
print(alpha)
#>              index
#> AAPL  0.0003999098
#> AMD   0.0013825599
#> ADI   0.0003609969
#> ABBV  0.0006684644
#> AEZS -0.0022091301
#> A     0.0002810622
#> APD   0.0001786379
#> AA    0.0006429140
#> CF   -0.0006256999
print(beta)
#>          index
#> AAPL 1.0957902
#> AMD  2.1738304
#> ADI  1.2683043
#> ABBV 0.9022734
#> AEZS 1.7115761
#> A    1.3277198
#> APD  1.0239446
#> AA   1.8593524
#> CF   1.5712742
Alternatively, we can do the fitting using a more compact matrix notation (recall time is along the vertical axis in X
)
XT=α1T+βfT+ET
and then write the LS fitting as
minimizeα,β∥XT−α1T−βfT∥2F.
More conveniently, we can define Γ=[α,β]
 and the extended factors F~=[1,f]
. The LS formulation can then be written as
minimizeΓ∥XT−ΓF~T∥2F
with solution
Γ=XTF~(F~TF~)−1

F_ <- cbind(ones = 1, f)
Gamma <- t(X) %*% F_ %*% solve(t(F_) %*% F_)  # better: Gamma <- t(solve(t(F_) %*% F_, t(F_) %*% X))
colnames(Gamma) <- c("alpha", "beta")
alpha <- Gamma[, 1]  # or alpha <- Gamma[, "alpha"]
beta <- Gamma[, 2]   # or beta <- Gamma[, "beta"]
print(Gamma)
#>              alpha      beta
#> AAPL  0.0003999098 1.0957902
#> AMD   0.0013825599 2.1738304
#> ADI   0.0003609969 1.2683043
#> ABBV  0.0006684644 0.9022734
#> AEZS -0.0022091301 1.7115761
#> A     0.0002810622 1.3277198
#> APD   0.0001786379 1.0239446
#> AA    0.0006429140 1.8593524
#> CF   -0.0006256999 1.5712742
E <- xts(t(t(X) - Gamma %*% t(F_)), index(X))  # residuals
Psi <- (1/(T-2)) * t(E) %*% E
Sigma <- as.numeric(var(f)) * beta %o% beta + diag(diag(Psi))
As expected, we get the same result regardless of the notation used. Alternatively, we can simply use the R package covFactorModel to do the work for us:

library(covFactorModel)
factor_model <- factorModel(X, type = "M", econ_fact = f)
cbind(alpha = factor_model$alpha, beta = factor_model$beta)
#>              alpha     index
#> AAPL  0.0003999098 1.0957902
#> AMD   0.0013825599 2.1738304
#> ADI   0.0003609969 1.2683043
#> ABBV  0.0006684644 0.9022734
#> AEZS -0.0022091301 1.7115761
#> A     0.0002810622 1.3277198
#> APD   0.0001786379 1.0239446
#> AA    0.0006429140 1.8593524
#> CF   -0.0006256999 1.5712742
Visualizing covariance matrices
It is interesting to visualize the estimated covariance matrix of the log-returns Σ=var(ft)ββT+Diag(Ψ)
, as well as that of the residuals Ψ
.

Let’s start with the covariance matrix of the log-returns:

library(corrplot)  #install.packages("corrplot")
corrplot(cov2cor(Sigma), mar = c(0,0,1,0), 
         main = "Covariance matrix of log-returns from 1-factor model")


We can observe that all the stocks are highly correlated, which is the effect of the market factor. In order to inspect finer details of the interdependency of the stocks, we should first remove the market or, equivalently, plot Ψ
:

corrplot(cov2cor(Psi), mar = c(0,0,1,0), order = "hclust", addrect = 3, 
         main = "Covariance matrix of residuals")


cbind(stock_namelist, sector_namelist)  # to recall the sectors
#>       stock_namelist sector_namelist         
#>  [1,] "AAPL"         "Information Technology"
#>  [2,] "AMD"          "Information Technology"
#>  [3,] "ADI"          "Information Technology"
#>  [4,] "ABBV"         "Health Care"           
#>  [5,] "AEZS"         "Health Care"           
#>  [6,] "A"            "Health Care"           
#>  [7,] "APD"          "Materials"             
#>  [8,] "AA"           "Materials"             
#>  [9,] "CF"           "Materials"
Interestingly, we can observe that the automatic clustering performed on Ψ
 correctly identifies the sectors of the stocks.

Assessing investment funds
In this example, we will assess the performance of several investment funds based on factor models. We will use the S&P 500 as the explicit market factor and assume a zero risk-free return rf=0
. In particular, we will consider the following six Exchange-Traded Funds (ETFs):

SPY - SPDR S&P 500 ETF (index tracking)
XIVH - Velocity VIX Short Vol Hedged (high alpha)
SPHB - PowerShares S&P 500 High Beta Portfolio (high beta)
SPLV - PowerShares S&P 500 Low Volatility Portfolio (low beta)
USMV - iShares Edge MSCI Min Vol USA ETF
JKD - iShares Morningstar Large-Cap ETF
We start by loading the data:

library(xts)
library(quantmod)

# set begin-end date and stock namelist
begin_date <- "2016-10-01"
end_date <- "2017-06-30"
stock_namelist <- c("SPY","XIVH", "SPHB", "SPLV", "USMV", "JKD")

# download data from YahooFinance
data_set <- xts()
for (stock_index in 1:length(stock_namelist))
  data_set <- cbind(data_set, Ad(getSymbols(stock_namelist[stock_index], 
                                            from = begin_date, to = end_date, auto.assign = FALSE)))
colnames(data_set) <- stock_namelist
indexClass(data_set) <- "Date"
head(data_set)
#>                 SPY   XIVH     SPHB     SPLV     USMV      JKD
#> 2016-10-03 199.7973 29.400 30.58071 37.71050 41.56594 117.8419
#> 2016-10-04 198.7788 30.160 30.49698 37.27040 41.16050 117.3814
#> 2016-10-05 199.6585 30.160 31.08310 37.18787 41.06836 117.9063
#> 2016-10-06 199.7973 30.160 31.01797 37.25206 41.09600 118.0445
#> 2016-10-07 199.1122 30.670 30.77609 37.15120 41.04993 117.7958
#> 2016-10-10 200.1492 31.394 31.06449 37.34375 41.25264 118.5510

SP500_index <- Ad(getSymbols("^GSPC", from = begin_date, to = end_date, auto.assign = FALSE))
colnames(SP500_index) <- "index"
head(SP500_index)
#>              index
#> 2016-10-03 2161.20
#> 2016-10-04 2150.49
#> 2016-10-05 2159.73
#> 2016-10-06 2160.77
#> 2016-10-07 2153.74
#> 2016-10-10 2163.66

# compute the log-returns of the stocks and of the SP500 index as explicit factor
X <- diff(log(data_set), na.pad = FALSE)
N <- ncol(X)  # number of stocks
T <- nrow(X)  # number of days
f <- diff(log(SP500_index), na.pad = FALSE)
Now we can compute the alpha and beta of all the ETFs:

F_ <- cbind(ones = 1, f)
Gamma <- t(solve(t(F_) %*% F_, t(F_) %*% X))
colnames(Gamma) <- c("alpha", "beta")
alpha <- Gamma[, 1]
beta <- Gamma[, 2]
print(Gamma)
#>              alpha      beta
#> SPY   7.142235e-05 1.0071428
#> XIVH  1.810392e-03 2.4971086
#> SPHB -2.422095e-04 1.5613514
#> SPLV  1.070906e-04 0.6777124
#> USMV  1.166182e-04 0.6511647
#> JKD   2.569557e-04 0.8883886
Some observations are now in order:

SPY is an ETF that follows the S&P 500 and, as expected, it has an alpha almost zero and a beta almost one: α=7.142211×10−5
 and β=1.0071423
.
XIVH is an ETF with a high alpha and indeed the alpha computed is the highest among the ETFs (1-2 orders of magnitude higher): α=1.810392×10−3
.
SPHB is an ETF supposedly with a high beta and the computed beta is amongh the hightest but not the highest: β=1.5613531
. Interestingly, the computed alpha is negative! So this ETF seems to be dangerous and caution should be taken.
SPLV is an ETF that aims at having a low volatility and indeed the computed beta is on the low side: β=0.6777072
.
USMV is also an ETF that aims at having a low volatility and indeed the computed beta is the lowest: β=0.6511671
.
JKD seems not to be extreme and shows a good tradeof.
To get a feeling we can use some visualization:

{ par(mfrow = c(1,2))  # two plots side by side
  barplot(rev(alpha), horiz = TRUE, main = "alpha", col = "red", cex.names = 0.75, las = 1)
  barplot(rev(beta), horiz = TRUE, main = "beta", col = "blue", cex.names = 0.75, las = 1)
  par(mfrow = c(1,1)) }  # reset to normal single plot


We can also compare the different ETFs in a more systematic way using, for example, the Sharpe ratio. Recalling the factor model for one asset and one factor
xt=α+βft+ϵt,t=1,…,T,
we obtain
E[xt]var[xt]=α+βE[ft]=β2var[ft]+σ2
and the Sharpe ratio follows:
SR=E[xt]var[xt]−−−−−√=α+βE[ft]β2var[ft]+σ2−−−−−−−−−−−√=α/β+E[ft]var[ft]+σ2/β2−−−−−−−−−−−−√≈α/β+E[ft]var[ft]−−−−−√
where we have assumed σ2≪var[ft]
. Thus, a way to rank the different assets based on the Sharpe ratio is to rank them based on the ratio α/β
:

idx_sorted <- sort(alpha/beta, decreasing = TRUE, index.return = TRUE)$ix
SR <- colMeans(X)/sqrt(diag(var(X)))
ranking <- cbind("alpha/beta" = (alpha/beta)[idx_sorted], 
                 SR = SR[idx_sorted], 
                 alpha = alpha[idx_sorted], 
                 beta = beta[idx_sorted])
print(ranking)
#>         alpha/beta         SR         alpha      beta
#> XIVH  7.249952e-04 0.13919483  1.810392e-03 2.4971086
#> JKD   2.892379e-04 0.17682634  2.569557e-04 0.8883886
#> USMV  1.790917e-04 0.12280015  1.166182e-04 0.6511647
#> SPLV  1.580178e-04 0.10887776  1.070906e-04 0.6777124
#> SPY   7.091581e-05 0.14170591  7.142235e-05 1.0071428
#> SPHB -1.551281e-04 0.07401573 -2.422095e-04 1.5613514
The following observations are in order:

In terms of α/β
, XIVH is the best (with the largest alpha) and SPHB the worst (with a negative alpha!).
In terms of exact Sharpe ratio (more exactly, Information ratio since we are ignoring the risk-free rate), JDK is the best, followed by SPY, which is surprising since it is just the market. This confirms the wisdom that the majority of the investment funds do not outperform the market.
SPHB is clearly the worst according to any measure: negative alpha, negative alpha-beta ratio, and exact Sharpe ratio.
JDK manages to have the best performance because its alpha is good (although not the best) while at the same time has a moderate beta of 0.88, so its exposure to the market if not extreme.
XIVH and SPHB have extreme exposures to the market with large betas.
USMV has the smallest exposure to the market with an acceptable alpha, and it’s Sharpe ratio is close to the top second and third.
Fundamental factor models: Fama-French
This example will illustrate the popular Fama-French 3-factor model using nine stocks from the S&P 500.

Let’s start by loading the data:

library(xts)
library(quantmod)

# set begin-end date and stock namelist
begin_date <- "2013-01-01"
end_date <- "2017-08-31"
stock_namelist <- c("AAPL", "AMD", "ADI",  "ABBV", "AEZS", "A",  "APD", "AA","CF")

# download data from YahooFinance
data_set <- xts()
for (stock_index in 1:length(stock_namelist))
  data_set <- cbind(data_set, Ad(getSymbols(stock_namelist[stock_index], 
                                            from = begin_date, to = end_date, auto.assign = FALSE)))
colnames(data_set) <- stock_namelist
indexClass(data_set) <- "Date"
#> Warning: 'indexClass<-' is deprecated.
#> Use 'tclass<-' instead.
#> See help("Deprecated") and help("xts-deprecated").

# download Fama-French factors from website
#url <- "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
#temp <- tempfile()
#download.file(url, temp, method = "libcurl", mode = "wb")
#unzip(temp, "F-F_Research_Data_Factors_daily.CSV")
#unlink(temp)
mydata <- read.csv("F-F_Research_Data_Factors_daily.CSV", skip = 4)
mydata <- mydata[-nrow(mydata), ]  # remove last row
fama_lib <- xts(x = mydata[, c(2,3,4)], order.by = as.Date(paste(mydata[, 1]), "%Y%m%d"))
str(fama_lib)
#> An 'xts' object on 1926-07-01/2017-11-30 containing:
#>   Data: num [1:24120, 1:3] 0.1 0.45 0.17 0.09 0.21 -0.71 0.62 0.04 0.48 0.04 ...
#>  - attr(*, "dimnames")=List of 2
#>   ..$ : NULL
#>   ..$ : chr [1:3] "Mkt.RF" "SMB" "HML"
#>   Indexed by objects of class: [Date] TZ: UTC
#>   xts Attributes:  
#>  NULL
head(fama_lib)
#>            Mkt.RF   SMB   HML
#> 1926-07-01   0.10 -0.24 -0.28
#> 1926-07-02   0.45 -0.32 -0.08
#> 1926-07-06   0.17  0.27 -0.35
#> 1926-07-07   0.09 -0.59  0.03
#> 1926-07-08   0.21 -0.36  0.15
#> 1926-07-09  -0.71  0.44  0.56
tail(fama_lib)
#>            Mkt.RF   SMB   HML
#> 2017-11-22  -0.05  0.10 -0.04
#> 2017-11-24   0.21  0.02 -0.44
#> 2017-11-27  -0.06 -0.36  0.03
#> 2017-11-28   1.06  0.38  0.84
#> 2017-11-29   0.02  0.04  1.45
#> 2017-11-30   0.82 -0.56 -0.50

# compute the log-returns of the stocks and the Fama-French factors
X <- diff(log(data_set), na.pad = FALSE)
N <- ncol(X)  # number of stocks
T <- nrow(X)  # number of days
F <- fama_lib[index(X)]/100
Now we have the three explicit factors in matrix F
 and want to fit the model
XT=α1T+BFT+ET
where now the loadings are a matrix of betas: B=[β1,…,βN]T=[b1,b2,b3]
. As before, we can do a LS fitting as
minimizeα,β∥XT−α1T−BFT∥2F.
More conveniently, we define Γ=[α,B]
 and the extended factors F~=[1,F]
. The LS formulation can then be written as
minimizeα,β∥XT−ΓF~T∥2F
with solution
Γ=XTF~(F~TF~)−1

F_ <- cbind(ones = 1, F)
Gamma <- t(solve(t(F_) %*% F_, t(F_) %*% X))
colnames(Gamma) <- c("alpha", "b1", "b2", "b3")
alpha <- Gamma[, 1]
B <- Gamma[, 2:4]
print(Gamma)
#>              alpha        b1          b2          b3
#> AAPL  1.437855e-04 0.9657598 -0.23339181 -0.49806910
#> AMD   6.181760e-04 1.4062105  0.80738336 -0.07240117
#> ADI  -2.285096e-05 1.2124016  0.09025897 -0.20739132
#> ABBV  1.621385e-04 1.0582332  0.02833767 -0.72152570
#> AEZS -4.513235e-03 0.6989534  1.31318108 -0.25160182
#> A     5.725580e-06 1.2179503  0.10360779 -0.20424833
#> APD   6.281556e-05 1.0222936 -0.04394097  0.11061128
#> AA   -4.587722e-05 1.3391852  0.62590136  0.99858692
#> CF   -5.777426e-04 1.0387865  0.48429987  0.82014405
Alternatively, we can simply use the R package covFactorModel to do the work for us:

library(covFactorModel)  # devtools::install_github("dppalomar/covFactorModel")
factor_model <- factorModel(X, type = "M", econ_fact = F)
cbind(alpha = factor_model$alpha, beta = factor_model$beta)
#>              alpha    Mkt.RF         SMB         HML
#> AAPL  1.437855e-04 0.9657598 -0.23339181 -0.49806910
#> AMD   6.181760e-04 1.4062105  0.80738336 -0.07240117
#> ADI  -2.285096e-05 1.2124016  0.09025897 -0.20739132
#> ABBV  1.621385e-04 1.0582332  0.02833767 -0.72152570
#> AEZS -4.513235e-03 0.6989534  1.31318108 -0.25160182
#> A     5.725580e-06 1.2179503  0.10360779 -0.20424833
#> APD   6.281556e-05 1.0222936 -0.04394097  0.11061128
#> AA   -4.587722e-05 1.3391852  0.62590136  0.99858692
#> CF   -5.777426e-04 1.0387865  0.48429987  0.82014405
Statistical factor models
Let’s now consider statistical factor models or implicit factor models where both the factors and the loadings are not available. Recall the principal factor method for the model XT=α1T+BFT+ET
 with K
 factors:

PCA:
sample mean: α^=x¯=1TXT1T
demeaned matrix: X¯=X−1Tx¯T
sample covariance matrix: Σ^=1T−1X¯TX¯
eigen-decomposition: Σ^=Γ^Λ^Γ^T
Estimates:
B^=Γ1^Λ^1/21
Ψ^=Diag(Σ^−B^B^T)
Σ^=B^B^T+Ψ^
Update the eigen-decomposition: Σ^−Ψ^=Γ^Λ^Γ^T
Repeat Steps 2-3 until convergence.
K <- 3
alpha <- colMeans(X)
X_ <- X - matrix(alpha, T, N, byrow = TRUE)
Sigma_prev <- matrix(0, N, N)
Sigma <- (1/(T-1)) * t(X_) %*% X_
eigSigma <- eigen(Sigma)
while (norm(Sigma - Sigma_prev, "F")/norm(Sigma, "F") > 1e-3) {
  B <- eigSigma$vectors[, 1:K, drop = FALSE] %*% diag(sqrt(eigSigma$values[1:K]), K, K)
  Psi <- diag(diag(Sigma - B %*% t(B)))
  Sigma_prev <- Sigma
  Sigma <- B %*% t(B) + Psi
  eigSigma <- eigen(Sigma - Psi)
}
cbind(alpha, B)
#>              alpha                                        
#> AAPL  7.074567e-04 0.0002732012 -0.004631555 -0.0044812833
#> AMD   1.372247e-03 0.0045782050 -0.035202253  0.0114545960
#> ADI   6.533112e-04 0.0004151829 -0.007379021 -0.0053057908
#> ABBV  7.787930e-04 0.0017513351 -0.003967806 -0.0056001220
#> AEZS -4.157636e-03 0.0769496364  0.002935902  0.0006249134
#> A     6.843808e-04 0.0012685628 -0.005680118 -0.0061495896
#> APD   6.236569e-04 0.0005442960 -0.004229341 -0.0057976718
#> AA    6.277163e-04 0.0027404776 -0.009796287 -0.0149172638
#> CF   -5.730289e-05 0.0023108478 -0.007409133 -0.0153434052
Again, we can simply use the R package covFactorModel to do the work for us:

library(covFactorModel)
factor_model <- factorModel(X, type = "S", K = K, max_iter = 10)
cbind(alpha = factor_model$alpha, beta = factor_model$beta)
#>              alpha      factor1      factor2       factor3
#> AAPL  7.074567e-04 0.0002732012 -0.004631555 -0.0044812833
#> AMD   1.372247e-03 0.0045782050 -0.035202253  0.0114545960
#> ADI   6.533112e-04 0.0004151829 -0.007379021 -0.0053057908
#> ABBV  7.787930e-04 0.0017513351 -0.003967806 -0.0056001220
#> AEZS -4.157636e-03 0.0769496364  0.002935902  0.0006249134
#> A     6.843808e-04 0.0012685628 -0.005680118 -0.0061495896
#> APD   6.236569e-04 0.0005442960 -0.004229341 -0.0057976718
#> AA    6.277163e-04 0.0027404776 -0.009796287 -0.0149172638
#> CF   -5.730289e-05 0.0023108478 -0.007409133 -0.0153434052
Final comparison of covariance matrix estimations via different factor models
We will finally compare the following different factor models:

sample covariance matrix
macroeconomic 1-factor model
fundamental 3-factor Fama-French model
statistical factor model
We will estimate the models during a training phase and then we will compare how well the estimated covariance matrices do compared to the sample covariance matrix of the test phase. The estimation error will be evaluated in terms of the Frobenius norm ∥Σ^−Σtrue∥F
 as well as the PRIAL (PeRcentage Improvement in Average Loss):
PRIAL=100×∥Σscm−Σtrue∥2F−∥Σ^−Σtrue∥2F∥Σscm−Σtrue∥2F
which goes to 0 when the estimation Σ^
 tends to the sample covariance matrix Σscm
 and goes to 100 when the estimation Σ^
 tends to the true covariance matrix Σtrue
.

Let’s load the training and test sets:

library(xts)
library(quantmod)

# set begin-end date and stock namelist
begin_date <- "2013-01-01"
end_date <- "2015-12-31"
stock_namelist <- c("AAPL", "AMD", "ADI",  "ABBV", "AEZS", "A",  "APD", "AA","CF")

# prepare stock data
data_set <- xts()
for (stock_index in 1:length(stock_namelist))
  data_set <- cbind(data_set, Ad(getSymbols(stock_namelist[stock_index], 
                                            from = begin_date, to = end_date, auto.assign = FALSE)))
colnames(data_set) <- stock_namelist
indexClass(data_set) <- "Date"
#> Warning: 'indexClass<-' is deprecated.
#> Use 'tclass<-' instead.
#> See help("Deprecated") and help("xts-deprecated").
X <- diff(log(data_set), na.pad = FALSE)
N <- ncol(X)
T <- nrow(X)

# prepare Fama-French factors
mydata <- read.csv("F-F_Research_Data_Factors_daily.CSV", skip = 4)
mydata <- mydata[-nrow(mydata), ]  # remove last row
fama_lib <- xts(x = mydata[, c(2,3,4)], order.by = as.Date(paste(mydata[, 1]), "%Y%m%d"))
F_FamaFrench <- fama_lib[index(X)]/100

# prepare index
SP500_index <- Ad(getSymbols("^GSPC", from = begin_date, to = end_date, auto.assign = FALSE))
colnames(SP500_index) <- "index"
f_SP500 <- diff(log(SP500_index), na.pad = FALSE)

# split data into training and set data
T_trn <- round(0.45*T)
X_trn <- X[1:T_trn, ]
X_tst <- X[(T_trn+1):T, ]
F_FamaFrench_trn <- F_FamaFrench[1:T_trn, ]
F_FamaFrench_tst <- F_FamaFrench[(T_trn+1):T, ]
f_SP500_trn <- f_SP500[1:T_trn, ]
f_SP500_tst <- f_SP500[(T_trn+1):T, ]
Now let’s estimate the different factor models with the training data:

# sample covariance matrix
Sigma_SCM <- cov(X_trn)

# 1-factor model
F_ <- cbind(ones = 1, f_SP500_trn)
Gamma <- t(solve(t(F_) %*% F_, t(F_) %*% X_trn))
colnames(Gamma) <- c("alpha", "beta")
alpha <- Gamma[, 1]
beta <- Gamma[, 2]
E <- xts(t(t(X_trn) - Gamma %*% t(F_)), index(X_trn))
Psi <- (1/(T_trn-2)) * t(E) %*% E
Sigma_SP500 <- as.numeric(var(f_SP500_trn)) * beta %o% beta + diag(diag(Psi))

# Fama-French 3-factor model
F_ <- cbind(ones = 1, F_FamaFrench_trn)
Gamma <- t(solve(t(F_) %*% F_, t(F_) %*% X_trn))
colnames(Gamma) <- c("alpha", "beta1", "beta2", "beta3")
alpha <- Gamma[, 1]
B <- Gamma[, 2:4]
E <- xts(t(t(X_trn) - Gamma %*% t(F_)), index(X_trn))
Psi <- (1/(T_trn-4)) * t(E) %*% E
Sigma_FamaFrench <- B %*% cov(F_FamaFrench_trn) %*% t(B) + diag(diag(Psi))

# Statistical 1-factor model
K <- 1
alpha <- colMeans(X_trn)
X_trn_ <- X_trn - matrix(alpha, T_trn, N, byrow = TRUE)
Sigma_prev <- matrix(0, N, N)
Sigma <- (1/(T_trn-1)) * t(X_trn_) %*% X_trn_
eigSigma <- eigen(Sigma)
while (norm(Sigma - Sigma_prev, "F")/norm(Sigma, "F") > 1e-3) {
  B <- eigSigma$vectors[, 1:K, drop = FALSE] %*% diag(sqrt(eigSigma$values[1:K]), K, K)
  Psi <- diag(diag(Sigma - B %*% t(B)))
  Sigma_prev <- Sigma
  Sigma <- B %*% t(B) + Psi
  eigSigma <- eigen(Sigma - Psi)
}
Sigma_PCA1 <- Sigma

# Statistical 3-factor model
K <- 3
alpha <- colMeans(X_trn)
X_trn_ <- X_trn - matrix(alpha, T_trn, N, byrow = TRUE)
Sigma_prev <- matrix(0, N, N)
Sigma <- (1/(T_trn-1)) * t(X_trn_) %*% X_trn_
eigSigma <- eigen(Sigma)
while (norm(Sigma - Sigma_prev, "F")/norm(Sigma, "F") > 1e-3) {
  B <- eigSigma$vectors[, 1:K] %*% diag(sqrt(eigSigma$values[1:K]), K, K)
  Psi <- diag(diag(Sigma - B %*% t(B)))
  Sigma_prev <- Sigma
  Sigma <- B %*% t(B) + Psi
  eigSigma <- eigen(Sigma - Psi)
}
Sigma_PCA3 <- Sigma

# Statistical 5-factor model
K <- 5
alpha <- colMeans(X_trn)
X_trn_ <- X_trn - matrix(alpha, T_trn, N, byrow = TRUE)
Sigma_prev <- matrix(0, N, N)
Sigma <- (1/(T_trn-1)) * t(X_trn_) %*% X_trn_
eigSigma <- eigen(Sigma)
while (norm(Sigma - Sigma_prev, "F")/norm(Sigma, "F") > 1e-3) {
  B <- eigSigma$vectors[, 1:K] %*% diag(sqrt(eigSigma$values[1:K]), K, K)
  Psi <- diag(diag(Sigma - B %*% t(B)))
  Sigma_prev <- Sigma
  Sigma <- B %*% t(B) + Psi
  eigSigma <- eigen(Sigma - Psi)
}
Sigma_PCA5 <- Sigma
Finally, let’s compare the different estimations in the test data:

Sigma_true <- cov(X_tst)
error <- c(SCM = norm(Sigma_SCM - Sigma_true, "F"),
           SP500 = norm(Sigma_SP500 - Sigma_true, "F"),
           FamaFrench = norm(Sigma_FamaFrench - Sigma_true, "F"),
           PCA1 = norm(Sigma_PCA1 - Sigma_true, "F"),
           PCA3 = norm(Sigma_PCA3 - Sigma_true, "F"),
           PCA5 = norm(Sigma_PCA5 - Sigma_true, "F"))
print(error)
#>         SCM       SP500  FamaFrench        PCA1        PCA3        PCA5 
#> 0.007105451 0.007100626 0.007089245 0.007139007 0.007100196 0.007103370
barplot(error, main = "Error in estimation of covariance matrix", 
        col = "aquamarine3", cex.names = 0.75, las = 1)


ref <- norm(Sigma_SCM - Sigma_true, "F")^2
PRIAL <- 100*(ref - error^2)/ref
print(PRIAL)
#>         SCM       SP500  FamaFrench        PCA1        PCA3        PCA5 
#>  0.00000000  0.13575632  0.45562585 -0.94675034  0.14786420  0.05857685
barplot(PRIAL, main = "PRIAL for estimation of covariance matrix", 
        col = "bisque3", cex.names = 0.75, las = 1)


👉 The final conclusion is that using factor models for covariance matrix estimation may help. In addition, it is not clear that using explicit factors helps as compared to the statistical factor modeling.

