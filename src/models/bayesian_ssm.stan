// Bayesian driver-constructor state-space model (Lindner et al. 2026 core)
// Qualifying: Normal on standardized lap times (lower = better)
// Race: sequential Plackett-Luce (lower mu = better rank)

data {
  int<lower=1> N_qual;
  int<lower=1> N_race_obs;
  int<lower=1> N_races;
  int<lower=1> D;
  int<lower=1> K;
  int<lower=1> T;
  array[N_qual] int<lower=1, upper=D> qual_driver;
  array[N_qual] int<lower=1, upper=K> qual_constructor;
  array[N_qual] int<lower=1, upper=T> qual_gp;
  vector[N_qual] y_qual;
  array[N_races] int<lower=2> race_size;
  array[N_race_obs] int<lower=1, upper=D> race_driver;
  array[N_race_obs] int<lower=1, upper=K> race_constructor;
  array[N_race_obs] int<lower=1, upper=T> race_gp;
  vector[N_race_obs] grid;
  array[N_race_obs] int<lower=1> finish_pos;
}

parameters {
  matrix[D, T] a_raw;
  matrix[K, T] c_raw;
  real<lower=0> sigma_qual;
  real<lower=0> sigma_a;
  real<lower=0> sigma_c;
  real beta_grid;
}

transformed parameters {
  matrix[D, T] a;
  matrix[K, T] c;
  for (t in 1:T) {
    a[, t] = a_raw[, t] - mean(a_raw[, t]);
    c[, t] = c_raw[, t] - mean(c_raw[, t]);
  }
}

model {
  sigma_qual ~ normal(0, 1);
  sigma_a ~ normal(0, 1);
  sigma_c ~ normal(0, 1);
  beta_grid ~ normal(0, 5);

  for (d in 1:D) {
    a_raw[d, 1] ~ normal(0, 1);
    for (t in 2:T) {
      a_raw[d, t] ~ normal(a_raw[d, t - 1], sigma_a);
    }
  }
  for (k in 1:K) {
    c_raw[k, 1] ~ normal(0, 1);
    for (t in 2:T) {
      c_raw[k, t] ~ normal(c_raw[k, t - 1], sigma_c);
    }
  }

  for (n in 1:N_qual) {
    real mu = a[qual_driver[n], qual_gp[n]] + c[qual_constructor[n], qual_gp[n]];
    y_qual[n] ~ normal(mu, sigma_qual);
  }

  {
    int pos = 1;
    for (r in 1:N_races) {
      int n = race_size[r];
      vector[n] mu;
      array[n] int order;
      for (i in 1:n) {
        int idx = pos + i - 1;
        mu[i] = a[race_driver[idx], race_gp[idx]]
              + c[race_constructor[idx], race_gp[idx]]
              + beta_grid * grid[idx];
        order[i] = finish_pos[idx];
      }
      // Sort by finishing position ascending (1 = winner)
      for (k in 1:(n - 1)) {
        int best = 1;
        for (i in 1:n) {
          if (order[i] == k) best = i;
        }
        vector[n - k + 1] remain;
        int m = 0;
        for (i in 1:n) {
          if (order[i] >= k) {
            m += 1;
            remain[m] = mu[i];
          }
        }
        target += remain[1] - log_sum_exp(remain);
      }
      pos += n;
    }
  }
}
