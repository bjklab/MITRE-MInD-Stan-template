// demonstration of stan model code for multilevel Bayesian model with random,
// subject-level effects, covarying parameters, and non-centered paramaterization
// parameters drawn from planned Outcome 1A (Figure 2, adjustment set)
// for demonstration purposes, Bernoulli likelihood is shown
data{
    int n; // n observations
    int k; // k covarying parameters
    int j; // j random effects
    array[n] int R; // antibiotic response
     vector[n] A; // antibiotics
     vector[n] C; // microbiome features
    array[n] int P; // pathogen category
    array[n] int M; // member index
}
parameters{
     matrix[k,j] Z;
     vector[k] abar;
     cholesky_factor_corr[k] L_Rho;
     vector<lower=0>[k] sigma;
}
transformed parameters{
     vector[j] a;
     vector[j] bP;
     vector[j] bC;
     vector[j] bA;
     matrix[j,k] v;
    v = (diag_pre_multiply(sigma, L_Rho) * Z)';
    bA = abar[k] + v[, k];
    bC = abar[k-1] + v[, k-1];
    bP = abar[k-2] + v[, k-2];
    a = abar[k-3] + v[, k-3];
}
model{
     vector[n] p;
    sigma ~ exponential( 1 );
    L_Rho ~ lkj_corr_cholesky( k );
    abar ~ normal( 0 , 1 );
    to_vector( Z ) ~ normal( 0 , 1 );
    for ( i in 1:n ) {
        p[i] = a[M[i]] + bP[M[i]] * P[i] + bC[M[i]] * C[i] + bA[M[i]] * A[i];
        p[i] = inv_logit(p[i]);
    }
    R ~ bernoulli( p );
}
generated quantities{
     matrix[k,k] Rho;
    Rho = multiply_lower_tri_self_transpose(L_Rho);
}


