from .StandardBWM import StandardBWM
from ..MCDMProblem import MCDMProblem
import numpy as np

class IntervalBWM(StandardBWM, MCDMProblem):
    _basic_model = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            array[DmNo] vector[CNo] AB_l;
            array[DmNo] vector[CNo] AB_h;
            array[DmNo] vector[CNo] AW_l;
            array[DmNo] vector[CNo] AW_h;
            vector<lower=0,upper=1>[CNo] e;

            real<lower=0.0001> gamma_param; // prior for the Gamma distribution
        } 
 
        parameters { 
            array[DmNo] simplex[CNo] W;
            array[DmNo] real<lower=0> kappa;

            simplex[CNo] wStar;
            real<lower=0> kappaStar;

            array[DmNo] vector<lower=0, upper=1>[CNo] AW_trnf; // [0,1] intervals
            array[DmNo] vector<lower=0, upper=1>[CNo] AB_trnf; // [0,1] intervals
        } 

        transformed parameters {
            array[DmNo] vector<lower=0>[CNo] AB; 
            array[DmNo] vector<lower=0>[CNo] AW;
            array[DmNo] simplex[CNo] AB_normalized;
            array[DmNo] simplex[CNo] AW_normalized;

            for(i in 1:DmNo) {
                AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; // moving [0,1] intervals to the desired intervals
                AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; // moving [0,1] intervals to the desired intervals

                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]); // Normalizing for a better fit
                AW_normalized[i] = AW[i] ./ sum(AW[i]); // Normalizing for a better fit      
            }
        } 

        model {
            kappaStar ~ gamma(gamma_param,gamma_param);
            wStar ~ dirichlet(0.01*e);
            
            for (i in 1:DmNo) {
                W[i] ~ dirichlet(kappaStar*wStar);
                kappa[i] ~ gamma(gamma_param,gamma_param);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        } 
    """
    
    _basic_model_clustering = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            array[DmNo] vector[CNo] AB_l;
            array[DmNo] vector[CNo] AB_h;
            array[DmNo] vector[CNo] AW_l;
            array[DmNo] vector[CNo] AW_h;
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

            array[DmNo] vector<lower=0, upper=1>[CNo] AW_trnf;
            array[DmNo] vector<lower=0, upper=1>[CNo] AB_trnf;
        } 

        transformed parameters {
            array[DmNo] vector<lower=0>[CNo] AB;
            array[DmNo] vector<lower=0>[CNo] AW;
            array[DmNo] simplex[CNo] AB_normalized;
            array[DmNo] simplex[CNo] AW_normalized;

            for(i in 1:DmNo) {
                AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; 
                AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; 

                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }
        } 

        model {
            ksi ~ gamma(gamma_param, gamma_param);

            for(i in 1:DmC)
                wc[i] ~ dirichlet(0.01*e);

            array[DmC] real contribution;

            for (i in 1:DmNo) {
                for(j in 1:DmC)
                     contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
                target += log_sum_exp(contribution);

                kappa[i] ~ gamma(gamma_param,gamma_param);
                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        }     
    """

    _basic_model_sorting = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            array[DmNo] vector[CNo] AB_l;
            array[DmNo] vector[CNo] AB_h;
            array[DmNo] vector[CNo] AW_l;
            array[DmNo] vector[CNo] AW_h;
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

            array[DmNo] vector<lower=0, upper=1>[CNo] AW_trnf;
            array[DmNo] vector<lower=0, upper=1>[CNo] AB_trnf;
        } 

        transformed parameters {
            array[DmNo] vector<lower=0>[CNo] AB;
            array[DmNo] vector<lower=0>[CNo] AW;
            array[DmNo] simplex[CNo] AB_normalized;
            array[DmNo] simplex[CNo] AW_normalized;

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
            kappaStar ~ gamma(.001,.001);
            wStar ~ dirichlet(0.01*e);
                            
            for (i in 1:DmNo) {
                W[i] ~ dirichlet(kappaStar*wStar);
                kappa[i] ~ gamma(.001,.001);

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
            array[DmNo] vector[CNo] AB_l;
            array[DmNo] vector[CNo] AB_h;
            array[DmNo] vector[CNo] AW_l;
            array[DmNo] vector[CNo] AW_h;
            vector<lower=0,upper=1>[CNo] e;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;

            real<lower=0.0001> gamma_param; // prior for the Gamma distribution
            real<lower=0.001> sigma_coef; // coefficient for the covariance matrix
        } 

        parameters { 
            vector[CNo] wStar_eta; 
            //real<lower=0> kappaStar;

            array[DmNo] vector[CNo] W_eta;
            array[DmNo] real<lower=0> kappa;

            array[DmNo] vector<lower=0, upper=1>[CNo] AW_trnf;
            array[DmNo] vector<lower=0, upper=1>[CNo] AB_trnf;
        } 

        transformed parameters {
            array[DmNo] vector<lower=0>[CNo] AB;
            array[DmNo] vector<lower=0>[CNo] AW;
            array[DmNo] simplex[CNo] AB_normalized;
            array[DmNo] simplex[CNo] AW_normalized;

            for(i in 1:DmNo) {
                AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; 
                AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; 

                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }

            array[DmNo] simplex[CNo] W;
            simplex[CNo] wStar;

            wStar = softmax(wStar_eta);
            
            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);
        } 

        model {
            //kappaStar ~ gamma(gamma_param,gamma_param);

            wStar_eta ~ multi_normal(mu, Sigma);
                            
            for (i in 1:DmNo) {
                W_eta[i] ~ multi_normal(wStar_eta, sigma_coef*Sigma);
                kappa[i] ~ gamma(gamma_param,gamma_param);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        }
    """

    _correlated_model_clustering = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            array[DmNo] vector[CNo] AB_l;
            array[DmNo] vector[CNo] AB_h;
            array[DmNo] vector[CNo] AW_l;
            array[DmNo] vector[CNo] AW_h;
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

            array[DmC] vector[CNo] wc_eta;
            array[DmNo] simplex[DmC] theta;

            array[DmNo] vector<lower=0, upper=1>[CNo] AW_trnf;
            array[DmNo] vector<lower=0, upper=1>[CNo] AB_trnf;
        } 

        transformed parameters {
            array[DmNo] vector<lower=0>[CNo] AB;
            array[DmNo] vector<lower=0>[CNo] AW;
            array[DmNo] simplex[CNo] AB_normalized;
            array[DmNo] simplex[CNo] AW_normalized;

            for(i in 1:DmNo) {
                AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; 
                AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; 

                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }

            array[DmNo] simplex[CNo] W;
            array[DmC] simplex[CNo] wc;

            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);

            for(i in 1:DmC)
                wc[i] = softmax(wc_eta[i]);
        } 

        model {}

            wStar_eta ~ multi_normal(mu, Sigma);

            for(j in 1:DmC)
                wc_eta[j] ~ multi_normal(mu, 1*Sigma);
                            
            for (i in 1:DmNo) {
                W_eta[i] ~ multi_normal(mu, 1*Sigma);

                array[DmC] real contribution;
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + multi_normal_lpdf( W_eta[i] | wc_eta[j], Sigma*sigma_coef);
                target += log_sum_exp(contribution);   

                kappa[i] ~ gamma(gamma_param,gamma_param);
                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        }
    """

    _correlated_model_sorting = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            array[DmNo] vector[CNo] AB_l;
            array[DmNo] vector[CNo] AB_h;
            array[DmNo] vector[CNo] AW_l;
            array[DmNo] vector[CNo] AW_h;
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
            array[DmNo] real<lower=0> kappa;

            vector[CNo] wStar_eta;
            real<lower=0> kappaStar; 

            array[DmNo] vector<lower=0, upper=1>[CNo] AW_trnf;
            array[DmNo] vector<lower=0, upper=1>[CNo] AB_trnf;
       
            vector[AltC] altMu;
        } 

        transformed parameters {
            array[DmNo] vector<lower=0>[CNo] AB;
            array[DmNo] vector<lower=0>[CNo] AW;
            array[DmNo] simplex[CNo] AB_normalized;
            array[DmNo] simplex[CNo] AW_normalized;

            for(i in 1:DmNo) {
                AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; 
                AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; 

                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }


            array[DmNo] simplex[CNo] W;
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
    
    def __init__(self, AB_l, AB_h, AW_l, AW_h, alternatives = None, dm_cluster_number=-1, alt_sort_number=-1, num_chain=3, num_samples=1000, opt={}):
        self.AB_L = np.array(AB_l)
        self.AB_H = np.array(AB_h)
        self.AW_L = np.array(AW_l)
        self.AW_H = np.array(AW_h)

        MCDMProblem.__init__(self,alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples, opt)

    @property
    def input_data(self):
        data = self._get_common_data()
        data['AW_l'] = self.AW_L         
        data['AW_h'] = self.AW_H
        data['AB_l'] = self.AB_L
        data['AB_h'] = self.AB_H
        data['e'] =  np.ones(self.criteria_no)
        return data
    
    @property
    def dm_no(self):
        return self.AB_L.shape[0]
    
    @property
    def criteria_no(self):
        return self.AB_L.shape[1]

    def _check_input_data(self):
        assert self.AB_L.shape == self.AB_H.shape, "AB_l and AB_h must be of the same size!"
        assert self.AW_L.shape == self.AW_H.shape, "AW_l and AW_h must be of the same size!"
        assert self.AB_L.shape == self.AW_L.shape, "AB and AW (lower and upper bounds) must be of the same size!"

        assert self.AB_L.shape[0] >=1, "No input"
        assert self.AW_L.shape[1] >=2, "The number of criteria must be more than 2!"

        return True

if __name__ == "__main__":
    a_b =  np.array([
        [ 3, 4, 6, 1, 5, 2, 9, 7],
        [ 1, 2, 8, 4, 5, 3, 9, 6],
        [ 2, 2, 3, 1, 5, 5, 9, 8],
        [ 2, 1, 8, 2, 9, 3, 8, 8],
        [ 2, 4, 9, 1, 4, 3, 5, 5],
        [ 1, 2, 9, 1, 3, 5, 5, 4]])

    a_w =  np.array([
        [ 7, 6, 4, 9, 5, 8, 1, 3],
        [ 9, 8, 2, 5, 4, 5, 1, 3],
        [ 8, 8, 5, 9, 5, 5, 1, 2],
        [ 8, 9, 2, 8, 1, 8, 2, 2],
        [ 8, 6, 1, 9, 6, 7, 4, 4],
        [ 9, 8, 1, 9, 7, 5, 5, 6]]) 

    dmNo, cNo = a_w.shape
    altNo = 50
    x = np.random.rand(altNo // 2, cNo)
    altMat = np.concatenate([x*0.01,x])

    opt = {'CriteriaDependence': False}

    bwm = IntervalBWM(AB_l=a_b, AB_h=a_b, AW_l = a_w, AW_h=a_w)#, alternatives=altMat, alt_sort_number=2, opt=opt)
    bwm.sampling()
    print('Ok')
        