from ..MCDMProblem import MCDMProblem
from ..SWING.StandardSWING import StandardSWING
import numpy as np

class GaussianSWING(StandardSWING, MCDMProblem):
    
    _basic_model = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] vector[CNo] Prf_md;
            array[DmNo] vector[CNo] Prf_sigma; 
            vector<lower=0,upper=1>[CNo] e;
            real <lower=0> gamma_param;
        } 

        parameters {             
            simplex[CNo] wStar;
            real<lower=0> kappaStar;

            array[DmNo] vector<lower=0>[CNo] Prf;
        } 

        transformed parameters {
            array[DmNo] simplex[CNo] W;

            for(i in 1:DmNo)
                W[i] = Prf[i] ./ sum(Prf[i]);       

        }

        model {
            kappaStar ~ gamma(gamma_param,gamma_param);
            wStar ~ dirichlet(0.01*e);

            W ~ dirichlet(kappaStar*wStar);

            for (i in 1:DmNo)
                Prf_md[i] ~ normal(Prf[i], Prf_sigma[i]);
            
        } 
    """

    _basic_model_clustering = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] vector[CNo] Prf_md;
            array[DmNo] vector[CNo] Prf_sigma; 
            vector<lower=0,upper=1>[CNo] e;
            real <lower=0> gamma_param;

            int<lower=2> DmC;
        } 

        parameters { 
            simplex[CNo] wStar;
            real<lower=0> kappaStar;

            array[DmNo] vector<lower=0>[CNo] Prf;

            array[DmC] simplex[CNo] wc;
            array[DmC] real<lower=0> ksi; 
            array[DmNo] simplex[DmC] theta;
        } 

        transformed parameters {
            array[DmNo] simplex[CNo] W;

            for(i in 1:DmNo)
                W[i] = Prf[i] ./ sum(Prf[i]);       

        }

        model {      
            ksi ~ gamma(gamma_param, gamma_param);
            for(i in 1:DmC)
                wc[i] ~ dirichlet(0.01*e);

            for (i in 1:DmNo){
                array[DmC] real contribution;
                for(j in 1:DmC)
                     contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
                target += log_sum_exp(contribution);

                Prf_md[i] ~ normal(Prf[i], Prf_sigma[i]);
        }
    } 
    """
    
    _basicModelSorting = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_md[DmNo]; 
            vector[CNo] AW_md[DmNo];
            vector[CNo] AB_sigma[DmNo]; 
            vector[CNo] AW_sigma[DmNo]; 
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> AltNo;
            int<lower=2> AltC;
            matrix[AltNo, CNo] Alt;
        } 

        parameters { 
            simplex[CNo] W[DmNo];
            real<lower=0> kappa[DmNo]; 

            simplex[CNo] wStar;
            real<lower=0> kappaStar;

            vector[AltC] altMu;

            vector<lower=0>[CNo] AW[DmNo];
            vector<lower=0>[CNo] AB[DmNo];
        } 

        transformed parameters {
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo){
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]); 
                AW_normalized[i] = AW[i] ./ sum(AW[i]);
            }

            vector[AltNo] v = Alt * wStar;
            v ./= (1-v);
            v = log(v);
            array[AltNo, AltC] real<upper=0> soft_z; // log unnormalized clusters
            
            for (n in 1:AltNo)
                for (k in 1:AltC)
                    soft_z[n, k] = -log(AltC) - 0.5 * pow(altMu[k] - v[n],2);        
        }

        model {
            kappaStar ~ gamma(.01,.01);
            wStar ~ dirichlet(0.01*e);

            W ~ dirichlet(kappaStar*wStar);
            kappa ~ gamma(.01,.01);
            
            for (i in 1:DmNo){
                AB_md[i] ~ normal(AB[i], AB_sigma[i]);
                AW_md[i] ~ normal(AW[i], AW_sigma[i]);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }

            for (n in 1:AltNo)
                target += log_sum_exp(soft_z[n]);

        } 
    """
    
    _correlated_model = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] vector[CNo] Prf_md;
            vector[CNo] Prf_sigma[DmNo]; 
            vector<lower=0,upper=1>[CNo] e;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;
            
            real<lower=0> gamma_param; // parameter for gamma distribution
            real<lower=0> sigma_coef; // parameter for sigma
        } 

        parameters { 
            vector[CNo] W_eta[DmNo]; 
            vector[CNo] wStar_eta;
            real<lower=0> kappaStar;  
            real<lower=0> kappa[DmNo]; 

            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];
        } 

        transformed parameters {
            array[DmNo] simplex[CNo] W;

            for(i in 1:DmNo)
                W[i] = Prf[i] ./ sum(Prf[i]);       

            simplex[CNo] wStar;

            wStar = softmax(wStar_eta);
            
            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);
        } 

        model {
            wStar_eta ~ multi_normal(mu, Sigma);
                            
            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, Sigma);

                Prf_md[i] ~ normal(Prf[i], Prf_sigma[i]);
    
            }
        }
    """

    _correlatedModelClustering = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_md[DmNo]; 
            vector[CNo] AW_md[DmNo]; 
            vector[CNo] AB_sigma[DmNo]; 
            vector[CNo] AW_sigma[DmNo];  
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> DmC;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;
        } 

        parameters { 
            vector[CNo] W_eta[DmNo];
            real<lower=0> kappa[DmNo];

            vector[CNo] wc_eta[DmC];
            real<lower=0> ksi[DmC]; 
            simplex[DmC] theta[DmNo];

            real<lower=0> kappaStar;

            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];  
        }

        transformed parameters {
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo){
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]); 
                AW_normalized[i] = AW[i] ./ sum(AW[i]);
            }
            
            simplex[CNo] W[DmNo];
            simplex[CNo] wc[DmC];

            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);
            
            for(i in 1:DmC)
                wc[i] = softmax(wc_eta[i]);  
        } 

        model {
            kappa ~ gamma(.01,.01);
            ksi ~ gamma(.001, .001);
            for (i in 1:DmC)
                wc_eta[i] ~ multi_normal(mu, Sigma);
 
            for (i in 1:DmNo) {     
                real contribution[DmC]; 
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + multi_normal_lpdf( W_eta[i] | wc_eta[j], Sigma);
                target += log_sum_exp(contribution);   

                AB_md[i] ~ normal(AB[i], AB_sigma[i]);
                AW_md[i] ~ normal(AW[i], AW_sigma[i]);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        }
    """

    _correlatedModelSorting = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_md[DmNo]; 
            vector[CNo] AW_md[DmNo]; 
            vector[CNo] AB_sigma[DmNo]; 
            vector[CNo] AW_sigma[DmNo]; 
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> AltNo;
            matrix[AltNo, CNo] Alt;
            int<lower=2> AltC;
            vector<lower=0,upper=1>[AltC] eAlt;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;
        } 

        parameters { 
            vector[CNo] W_eta[DmNo]; 
            vector[CNo] wStar_eta;
            real<lower=0> kappaStar;  
            real<lower=0> kappa[DmNo]; 

            vector[AltC] altMu;

            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];
        } 

        transformed parameters {
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo){
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]); 
                AW_normalized[i] = AW[i] ./ sum(AW[i]);
            }
                   
            simplex[CNo] W[DmNo];
            simplex[CNo] wStar;

            wStar = softmax(wStar_eta);
            
            for(i in 1:DmNo) {
                W[i] = softmax(W_eta[i]);
            }

            vector[AltNo] v = Alt * wStar;
            v ./= (1-v);
            v = log(v);
            array[AltNo, AltC] real<upper=0> soft_z; // log unnormalized clusters
            for (n in 1:AltNo)
                for (k in 1:AltC)
                    soft_z[n, k] = -log(AltC) - 0.5 * pow(altMu[k] - v[n],2);
                
        } 

        model {
            wStar_eta ~ multi_normal(mu, Sigma);
            kappa ~ gamma(.01,.01);
                            
            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, Sigma);

                AB_md[i] ~ normal(AB[i], AB_sigma[i]);
                AW_md[i] ~ normal(AW[i], AW_sigma[i]);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }

            for (n in 1:AltNo)
               target += log_sum_exp(soft_z[n]);
        }
    """

    def __init__(self, Prf_md, Prf_sigma, alternatives = None, dm_cluster_number=-1, alt_sort_number=-1, num_chain=3, num_samples=1000, opt={}):
        self.Prf_MD = np.array(Prf_md)
        self.Prf_Sigma = np.array(Prf_sigma)

        MCDMProblem.__init__(self, alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples,opt)


    @property
    def input_data(self):
        data = self._get_common_data()
        data['Prf_md'] = self.Prf_MD
        data['Prf_sigma'] = self.Prf_Sigma
        data['e'] =  np.ones(self.criteria_no)

        return data
    

    @property
    def dm_no(self):
        return self.Prf_MD.shape[0]
    
    @property
    def criteria_no(self):
        return self.Prf_MD.shape[1]

    def _check_input_data(self):
        assert self.Prf_MD.shape == self.Prf_Sigma.shape, "AB_MD and AB_Sigma must be of the same size!"

        return True


