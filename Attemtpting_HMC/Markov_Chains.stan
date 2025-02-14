// Hierarchical Markov Chain Model for Neural Data
data {
  int<lower=1> N;           // Total number of observations
  int<lower=1> P;           // Number of participants
  int<lower=1> E;           // Number of electrodes
  int<lower=1> C;           // Number of conditions
  
  // Indexing
  int<lower=1, upper=P> participant_idx[N];
  int<lower=1, upper=E> electrode_idx[N];
  int<lower=1, upper=C> condition_idx[N];
  
  // Response variable (could be coefficient, complex value, or derived metric)
  vector[N] response;
  
  // Additional covariates
  vector[50] complex_values[N];
  real phase_coherence[N];
  int<lower=1, upper=C> stimulus_category[N];
}
parameters {
  // Participant-level effects
  vector[P] participant_effect;
  real<lower=0> participant_sigma;
  
  // Electrode-level effects
  vector[E] electrode_effect;
  real<lower=0> electrode_sigma;
  
  // Condition-level effects
  vector[C] condition_effect;
  
  // Stimulus category effects
  vector[C] stimulus_category_effect;
  
  // Phase coherence effect
  real phase_coherence_slope;
  
  // Overall model variance
  real<lower=0> sigma;
}
model {
  // Priors
  
  // Participant-level prior
  participant_effect ~ normal(0, participant_sigma);
  participant_sigma ~ cauchy(0, 1);
  
  // Electrode-level prior
  electrode_effect ~ normal(0, electrode_sigma);
  electrode_sigma ~ cauchy(0, 1);
  
  // Condition-level prior
  condition_effect ~ normal(0, 1);
  
  // Stimulus category effects
  stimulus_category_effect ~ normal(0, 1);
  
  // Phase coherence effect
  phase_coherence_slope ~ normal(0, 1);
  
  // Overall variance
  sigma ~ exponential(1);
  
  // Likelihood
  for (i in 1:N) {
    real mu = participant_effect[participant_idx[i]] + 
              electrode_effect[electrode_idx[i]] + 
              condition_effect[condition_idx[i]] + 
              stimulus_category_effect[stimulus_category[i]] + 
              phase_coherence_slope * phase_coherence[i];
    
    response[i] ~ normal(mu, sigma);
  }
}
generated quantities {
  // Log-likelihood for model comparison
  vector[N] log_lik;
  
  // Posterior predictions
  vector[N] y_pred;
  
  for (i in 1:N) {
    real mu = participant_effect[participant_idx[i]] + 
              electrode_effect[electrode_idx[i]] + 
              condition_effect[condition_idx[i]] + 
              stimulus_category_effect[stimulus_category[i]] + 
              phase_coherence_slope * phase_coherence[i];
    
    log_lik[i] = normal_lpdf(response[i] | mu, sigma);
    y_pred[i] = normal_rng(mu, sigma);
  }
}
