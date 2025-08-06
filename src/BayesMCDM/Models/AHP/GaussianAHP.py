from .StandardAHP import StandardAHP
from ..MCDMProblem import MCDMProblem
import numpy as np

class GaussianAHP(StandardAHP, MCDMProblem):
    
    _basic_model = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            array[DmNo] matrix[CNo,CNo] PCM_md;
            array[DmNo] matrix[CNo,CNo] PCM_sigma;
            vector<lower=0,upper=1>[CNo] e;
            real<lower=0.0001> gamma_param; // prior for the Gamma distribution
        } 

        parameters {             
            simplex[CNo] wStar;
            real<lower=0> kappaStar;

            array[DmNo] simplex[CNo] W;
            array[DmNo] real<lower=0> kappa;

            array[DmNo] matrix<lower=0>[CNo,CNo] PCM;
        } 

        transformed parameters {
            array[DmNo] matrix[CNo,CNo] PCM_normalized;

            for(i in 1:DmNo) 
                for(j in 1:CNo)
                    PCM_normalized[i][,j] = col(PCM[i],j) ./  sum(col(PCM[i],j));
        }

        model {
            kappaStar ~ gamma(gamma_param,gamma_param);
            wStar ~ dirichlet(0.01*e);
            
            for (i in 1:DmNo){
                kappa[i] ~ gamma(gamma_param,gamma_param);
                W[i] ~ dirichlet(kappaStar*wStar);

                for(j in 1:CNo){
                    for (k in 1:CNo) {
                        PCM_md[i][j,k] ~ normal(PCM[i][j,k], PCM_sigma[i][j,k]);
                    }
                }
            }
        } 
    """

    _basic_model_clustering = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            array[DmNo] matrix[CNo,CNo] PCM_md;
            array[DmNo] matrix[CNo,CNo] PCM_sigma;
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> DmC;
            real<lower=0.0001> gamma_param; // prior for the Gamma distribution
        } 

        parameters { 
            array[DmNo] simplex[CNo] W;
            array[DmNo] real<lower=0> kappa;

            array[DmC] simplex[CNo] wc;
            array[DmC] real<lower=0> ksi; 
            array[DmNo] simplex[DmC] theta;

            array[DmNo] matrix<lower=0>[CNo,CNo] PCM;
        } 

        transformed parameters {
            array[DmNo] matrix[CNo,CNo] PCM_normalized;

            for(i in 1:DmNo) 
                for(j in 1:CNo)
                    PCM_normalized[i][,j] = col(PCM[i],j) ./  sum(col(PCM[i],j));
        }

        model {      
            for(i in 1:DmC){
                wc[i] ~ dirichlet(0.01*e);
                ksi[i] ~ gamma(gamma_param,gamma_param);
            }

            for (i in 1:DmNo){
                array[DmC] real contribution;
                for(j in 1:DmC)
                     contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
                target += log_sum_exp(contribution);

                for(j in 1:CNo)
                    for (k in 1:CNo) 
                        PCM_md[i][j,k] ~ normal(PCM[i][j,k], PCM_sigma[i][j,k]);

                kappa[i] ~ gamma(gamma_param,gamma_param);
                for(j in 1:CNo)
                    PCM_normalized[i][,j] ~ dirichlet(kappa[i]*W[i]);
                
            }
    } 
    """
    
    _basic_model_sorting = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            array[DmNo] vector[CNo] AB_md; 
            array[DmNo] vector[CNo] AW_md; 
            array[DmNo] vector[CNo] AB_sigma; 
            array[DmNo] vector[CNo] AW_sigma; 
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> AltNo;
            int<lower=2> AltC;
            matrix[AltNo, CNo] Alt;
        } 

        parameters { 
            array[DmNo] simplex[CNo] W;
            array[DmNo] real<lower=0> kappa;

            simplex[CNo] wStar;
            real<lower=0> kappaStar;

            vector[AltC] altMu;

            array[DmNo] vector<lower=0>[CNo] AW;
            array[DmNo] vector<lower=0>[CNo] AB;
        } 

        transformed parameters {
            array[DmNo] simplex[CNo] AB_normalized;
            array[DmNo] simplex[CNo] AW_normalized;

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
            array[DmNo] matrix[CNo,CNo] PCM_md;
            array[DmNo] matrix[CNo,CNo] PCM_sigma;
            vector<lower=0,upper=1>[CNo] e;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;

            real<lower=0.0001> gamma_param; // prior for the Gamma distribution
            real<lower=0.001> sigma_coef; // coefficient for the covariance matrix
        } 

        parameters { 
            array[DmNo] vector[CNo] W_eta; 
            vector[CNo] wStar_eta;
            array[DmNo] real<lower=0> kappa;

            array[DmNo] matrix<lower=0>[CNo,CNo] PCM;
        } 

        transformed parameters {
            array[DmNo] matrix[CNo,CNo] PCM_normalized;

            for(i in 1:DmNo) 
                for(j in 1:CNo)
                    PCM_normalized[i][,j] = col(PCM[i],j) ./  sum(col(PCM[i],j));

            array[DmNo] simplex[CNo] W;
            simplex[CNo] wStar;

            wStar = softmax(wStar_eta);
            
            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);
        } 

        model {
            wStar_eta ~ multi_normal(mu, Sigma);
            kappaStar ~ gamma(gamma_param,gamma_param);
            wStar ~ dirichlet(0.01*e);

            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, Sigma);
                kappa[i] ~ gamma(gamma_param,gamma_param);

                for(j in 1:CNo)
                    PCM_normalized[i][,j] ~ dirichlet(kappa[i]*W[i]); 

                for(j in 1:CNo){
                    for (k in 1:CNo) {
                        PCM_md[i][j,k] ~ normal(PCM[i][j,k], PCM_sigma[i][j,k]);
                    }
                }
            }
            
        }
    """

    _correlated_model_clustering = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            array[DmNo] matrix[CNo,CNo] PCM_md;
            array[DmNo] matrix[CNo,CNo] PCM_sigma;
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> DmC;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;

            real<lower=0.0001> gamma_param; // prior for the Gamma distribution
            real<lower=0.001> sigma_coef; // coefficient for the covariance matrix
        } 

        parameters { 
            array[DmNo] vector[CNo] W_eta;
            array[DmNo] real<lower=0> kappa;

            array[DmNo] simplex[DmC] theta;
            array[DmC] vector[CNo] wc_eta;
            array[DmNo] matrix<lower=0>[CNo,CNo] PCM;
        }

        transformed parameters {
            array[DmNo] matrix[CNo,CNo] PCM_normalized;

            for(i in 1:DmNo) 
                for(j in 1:CNo)
                    PCM_normalized[i][,j] = col(PCM[i],j) ./  sum(col(PCM[i],j));
       
            array[DmNo] simplex[CNo] W;
            array[DmC] simplex[CNo] wc;

            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);
            
            for(i in 1:DmC)
                wc[i] = softmax(wc_eta[i]);  
        } 

        model {
            for (i in 1:DmC)
                wc_eta[i] ~ multi_normal(mu, Sigma);
 
            for (i in 1:DmNo) {     
                array[DmC] real contribution; 
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + multi_normal_lpdf( W_eta[i] | wc_eta[j], Sigma*sigma_coef);
                target += log_sum_exp(contribution);   

                for(j in 1:CNo){
                    for (k in 1:CNo) {
                        PCM_md[i][j,k] ~ normal(PCM[i][j,k], PCM_sigma[i][j,k]);
                    }
                }
                kappa[i] ~ gamma(gamma_param,gamma_param);
                for(j in 1:CNo){
                    PCM_normalized[i][,j] ~ dirichlet(kappa[i]*W[i]);
                }   
    
            }
        }
    """

    _correlated_model_sorting = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            array[DmNo] vector[CNo] AB_md; 
            array[DmNo] vector[CNo] AW_md; 
            array[DmNo] vector[CNo] AB_sigma; 
            array[DmNo] vector[CNo] AW_sigma; 
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> AltNo;
            matrix[AltNo, CNo] Alt;
            int<lower=2> AltC;
            vector<lower=0,upper=1>[AltC] eAlt;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;
        } 

        parameters { 
            array[DmNo] vector[CNo] W_eta; 
            vector[CNo] wStar_eta;
            real<lower=0> kappaStar;  
            array[DmNo] real<lower=0> kappa;

            vector[AltC] altMu;

            array[DmNo] vector<lower=0>[CNo] AB;
            array[DmNo] vector<lower=0>[CNo] AW;
        } 

        transformed parameters {
            array[DmNo] simplex[CNo] AB_normalized;
            array[DmNo] simplex[CNo] AW_normalized;

            for(i in 1:DmNo){
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]); 
                AW_normalized[i] = AW[i] ./ sum(AW[i]);
            }

            array[DmNo] simplex[CNo] W;
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

    def __init__(self, PCM_md, PCM_sigma, alternatives = None, dm_cluster_number=-1, alt_sort_number=-1, num_chain=3, num_samples=1000, opt={}):
        self.PCM_MD = np.array(PCM_md)
        self.PCM_Sigma = np.array(PCM_sigma)

        MCDMProblem.__init__(self, alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples,opt)

    @property
    def input_data(self):
        data = self._get_common_data()
        data['PCM_md'] = self.PCM_MD
        data['PCM_sigma'] = self.PCM_Sigma
        data['e'] =  np.ones(self.criteria_no)
        
        return data

    @property
    def dm_no(self):
        return self.PCM_MD.shape[0]
    
    @property
    def criteria_no(self):
        return self.PCM_MD.shape[1]

    def _check_input_data(self):
        assert self.PCM_MD.shape == self.PCM_Sigma.shape, "PCM_MD and PCM_Sigma must be of the same size!"

        assert self.PCM_MD.shape[0] >=1, "No input"
        assert self.PCM_MD.shape[1] >=2, "The number of criteria must be more than 2!"

        return True
