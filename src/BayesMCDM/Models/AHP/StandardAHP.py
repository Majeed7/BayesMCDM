from ..MCDMProblem import MCDMProblem 
import numpy as np

class StandardAHP(MCDMProblem):

    _basic_model = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] matrix[CNo,CNo] PCM;
            vector<lower=0,upper=1>[CNo] e;
            real<lower=0> gamma_param;
         }

         parameters {
            array[DmNo] simplex[CNo] W;
            array[DmNo] real<lower=0> kappa;

            simplex[CNo] wStar;
            real<lower=0> kappaStar;
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
                W[i] ~ dirichlet(kappaStar*wStar);
                kappa[i] ~ gamma(gamma_param,gamma_param);

                for(j in 1:CNo){
                    PCM_normalized[i][,j] ~ dirichlet(kappa[i]*W[i]);
                }   
            }
         }
    """

    _basic_model_clustering = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] matrix[CNo,CNo] PCM;
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> DmC;
            real<lower=0> gamma_param;
        }

        parameters {
            array[DmNo] simplex[CNo] W;
            array[DmNo] real<lower=0> kappa;

            array[DmC] simplex[CNo] wc;
            array[DmC] real<lower=0> ksi;
            array[DmNo] simplex[DmC] theta;
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
            
            for (i in 1:DmNo) {
                array[DmC] real contribution;
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
                target += log_sum_exp(contribution);

                kappa[i] ~ gamma(gamma_param,gamma_param);
                for(j in 1:CNo)
                    PCM_normalized[i][,j] ~ dirichlet(kappa[i]*W[i]); 
            }
        }
    """

    _correlated_model = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] matrix[CNo,CNo] PCM;
            vector<lower=0,upper=1>[CNo] e;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;

            real<lower=0> gamma_param; // parameter for gamma distribution
            real<lower=0> sigma_coef; // parameter for sigma
        }

        parameters {
            array[DmNo] vector[CNo] W_eta;
            array[DmNo] real<lower=0> kappa;

            vector[CNo] wStar_eta;
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

            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, sigma_coef*Sigma);
                kappa[i] ~ gamma(gamma_param,gamma_param);

                for(j in 1:CNo)
                    PCM_normalized[i][,j] ~ dirichlet(kappa[i]*W[i]); 
            }
        }
    """

    _correlated_model_clustering = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] matrix[CNo,CNo] PCM;
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> DmC;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;

            real<lower=0.0001> gamma_param; // prior for the Gamma distribution
            real<lower=0.0001> sigma_coef; // coefficient for the covariance matrix
        }

        parameters {
            array[DmNo] vector[CNo] W_eta;
            array[DmNo] real<lower=0> kappa;

            array[DmC] vector[CNo] wc_eta;
            array[DmNo] simplex[DmC] theta;
        }

        transformed parameters {
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
            wStar_eta ~ multi_normal(mu, Sigma);

            for (d in 1:DmC){
                wc_eta[d] ~ multi_normal(mu, Sigma);
            }

            for (i in 1:DmNo){

                array[DmC] real contribution;
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + multi_normal_lpdf( W_eta[i] | wc_eta[j], Sigma*sigma_coef);
                target += log_sum_exp(contribution);

                kappa[i] ~ gamma(gamma_param,gamma_param);
                for(j in 1:CNo)
                    PCM_normalized[i][,j] ~ dirichlet(kappa[i]*W[i]); 
            }
        }
    """

    def __init__(self, PCM, alternatives = None, dm_cluster_number=-1, alt_sort_number=-1, num_chain=3, num_samples=1000, opt={}):
        self.PCM = np.array(PCM)
        
        super().__init__(alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples, opt)


    @property
    def input_data(self):
        data = self._get_common_data()
        data['PCM'] = self.PCM
        data['e'] =  np.ones(self.criteria_no)

        return data

    @property
    def original_model(self):
        return self.__originalModel

    @property
    def dm_no(self):
        return self.PCM.shape[0]

    @property
    def criteria_no(self):
        return self.PCM[0].shape[1]

    def _check_input_data(self):

        return True

