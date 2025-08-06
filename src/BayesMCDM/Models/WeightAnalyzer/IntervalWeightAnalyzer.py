from ..SWING.StandardSWING import StandardSWING
from ..MCDMProblem import MCDMProblem
import numpy as np

class IntervalWeightAnalyzer(StandardSWING, MCDMProblem):
    _basic_model = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] vector[CNo] W_l;
            array[DmNo] vector[CNo] W_h;
            vector<lower=0,upper=1>[CNo] e;
            real <lower=0> gamma_param;
        } 
 
        parameters { 
            simplex[CNo] wStar;
            real<lower=0> kappaStar;

            array[DmNo] vector<lower=0, upper=1>[CNo] W_trnf; // [0,1] intervals
        } 

        transformed parameters {
            array[DmNo] vector[CNo] W;

            for(i in 1:DmNo) {
                W[i] = W_l[i] + (W_h[i] - W_l[i]) .* W_trnf[i]; // moving [0,1] intervals to the desired intervals

                W[i] = W[i] ./ sum(W[i]); // Normalizing for a better fit
            }

        } 

        model {
            kappaStar ~ gamma(gamma_param, gamma_param);
            wStar ~ dirichlet(0.01*e);

            for (i in 1:DmNo) {
                W[i] ~ dirichlet(kappaStar*wStar);
            }
        } 
    """

    _basic_model_clustering = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] vector[CNo] W_l;
            array[DmNo] vector[CNo] W_h;
            vector<lower=0,upper=1>[CNo] e;
            real <lower=0> gamma_param;

            int<lower=2> DmC;
        } 

        parameters { 
            simplex[CNo] wStar;
            real<lower=0> kappaStar;

            array[DmNo] vector<lower=0, upper=1>[CNo] W_trnf; // [0,1] intervals
            
            array[DmC] simplex[CNo] wc;
            array[DmC] real<lower=0> ksi;
            array[DmNo] simplex[DmC] theta;
        } 

        transformed parameters {
            array[DmNo] vector[CNo] W;

            for(i in 1:DmNo) {
                W[i] = W_l[i] + (W_h[i] - W_l[i]) .* W_trnf[i]; // moving [0,1] intervals to the desired intervals
                W[i] = W[i] ./ sum(W[i]); // Normalizing for a better fit
            }
        } 

        model {
            for(i in 1:DmC){
                wc[i] ~ dirichlet(0.01*e);
                ksi[i] ~ gamma(gamma_param, gamma_param);
            }

            array[DmC] real contribution;

            for (i in 1:DmNo) {
                for(j in 1:DmC)
                     contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
                target += log_sum_exp(contribution);
            }
        }     
    """

    _basicModelSorting = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_l[DmNo]; 
            vector[CNo] AB_h[DmNo]; 
            vector[CNo] AW_l[DmNo]; 
            vector[CNo] AW_h[DmNo];  
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

            vector<lower=0, upper=1>[CNo] AW_trnf[DmNo];
            vector<lower=0, upper=1>[CNo] AB_trnf[DmNo];
        } 

        transformed parameters {
            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; 
                AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; 

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
                            
            for (i in 1:DmNo) {
                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
            
            for (n in 1:AltNo)
                target += log_sum_exp(soft_z[n]);
        } 
    """
    
    _correlatedModel = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_l[DmNo]; 
            vector[CNo] AB_h[DmNo]; 
            vector[CNo] AW_l[DmNo]; 
            vector[CNo] AW_h[DmNo];  
            vector<lower=0,upper=1>[CNo] e;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;
        } 

        parameters { 
            vector[CNo] wStar_eta; 
            real<lower=0> kappaStar;
            
            vector[CNo] W_eta[DmNo]; 
            real<lower=0> kappa[DmNo]; 

            vector<lower=0, upper=1>[CNo] AW_trnf[DmNo];
            vector<lower=0, upper=1>[CNo] AB_trnf[DmNo];
        } 

        transformed parameters {
            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; 
                AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; 

                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }

            simplex[CNo] W[DmNo];
            simplex[CNo] wStar;

            wStar = softmax(wStar_eta);
            
            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);
        } 

        model {
            //kappaStar ~ gamma(.01,.01);
            //wStar ~ dirichlet(0.01*e);
            //W ~ dirichlet(kappaStar*wStar);

            wStar_eta ~ multi_normal(mu, Sigma);
            kappa ~ gamma(.01,.01);
                            
            for (i in 1:DmNo) {
                W_eta[i] ~ multi_normal(wStar_eta, 0.1*Sigma);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        }
    """

    _correlatedModelClustering = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_l[DmNo]; 
            vector[CNo] AB_h[DmNo]; 
            vector[CNo] AW_l[DmNo]; 
            vector[CNo] AW_h[DmNo];  
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
            
            vector<lower=0, upper=1>[CNo] AW_trnf[DmNo];
            vector<lower=0, upper=1>[CNo] AB_trnf[DmNo];
        } 

        transformed parameters {
            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; 
                AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; 

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
            real contribution[DmC];

            //wStar_eta ~ multi_normal(mu, Sigma);
            kappa ~ gamma(.01,.01);

            for(j in 1:DmC)
                wc_eta[j] ~ multi_normal(mu, 1*Sigma);
                            
            for (i in 1:DmNo) {
                W_eta[i] ~ multi_normal(mu, 1*Sigma);

                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + multi_normal_lpdf( W_eta[i] | wc_eta[j], Sigma*.01);
                target += log_sum_exp(contribution);   

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        }
    """

    _correlatedModelSorting = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_l[DmNo]; 
            vector[CNo] AB_h[DmNo]; 
            vector[CNo] AW_l[DmNo]; 
            vector[CNo] AW_h[DmNo];  
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
            real<lower=0> kappa[DmNo]; 
 
            vector[CNo] wStar_eta;
            real<lower=0> kappaStar; 

            vector<lower=0, upper=1>[CNo] AW_trnf[DmNo];
            vector<lower=0, upper=1>[CNo] AB_trnf[DmNo];
       
            vector[AltC] altMu;
        } 

        transformed parameters {
            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; 
                AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; 

                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }

            
            simplex[CNo] W[DmNo];
            simplex[CNo] wStar;

            wStar = softmax(wStar_eta);
            
            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);

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
                            
            for (i in 1:DmNo) {
                W_eta[i] ~ multi_normal(wStar_eta, 0.01*Sigma);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }

            for (n in 1:AltNo)
               target += log_sum_exp(soft_z[n]);
        }
    """
    
    def __init__(self, Prf_l, Prf_h, alternatives = None, dm_cluster_number=-1, alt_sort_number=-1, num_chain=3, num_samples=1000, opt={}):
        self.Prf_L = np.array(Prf_l)
        self.Prf_H = np.array(Prf_h)

        MCDMProblem.__init__(self, alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples, opt)


    @property
    def input_data(self):
        
        data = self._get_common_data()
        data['Prf_l'] = self.Prf_L         
        data['Prf_h'] = self.Prf_H
        data['e'] =  np.ones(self.criteria_no)

        return data
    
    @property
    def dm_no(self):
        return self.Prf_L.shape[0]

    @property
    def criteria_no(self):
        return self.Prf_L.shape[1]

    def _check_input_data(self):
        assert self.Prf_L.shape == self.Prf_H.shape, "Prf_l and Prf_h must be of the same size!"

        assert self.Prf_L.shape[0] >=1, "No input"
        assert self.Prf_L.shape[1] >=2, "The number of criteria must be more than 2!"

        return True

