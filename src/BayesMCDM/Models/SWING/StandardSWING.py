from ..MCDMProblem import MCDMProblem
import numpy as np

class StandardSWING(MCDMProblem):
    _basic_model = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] vector[CNo] Prf;
            vector<lower=0,upper=1>[CNo] e;
            real<lower=0> gamma_param;
        }

        parameters {
            simplex[CNo] wStar;
            real<lower=0> kappaStar;
        }

        transformed parameters {
            array[DmNo] simplex[CNo] W;

            for(i in 1:DmNo)
                W[i] = Prf[i] ./ sum(Prf[i]);       
        } 

        model {
            wStar ~ dirichlet(0.01*e);
            kappaStar ~ gamma(gamma_param,gamma_param);

            for (i in 1:DmNo){
                W[i] ~ dirichlet(kappaStar*wStar);
            }
        }
    """

    _basic_model_clustering = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] vector[CNo] Prf;
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> DmC;
            real<lower=0> gamma_param;
        }

        parameters {
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
            for (i in 1:DmNo) {
                array[DmC] real contribution;
                for(j in 1:DmC){
                    vector[CNo] conc = ksi[j] * wc[j];
                    for (k in 1:CNo) {
                        if (conc[k] < 1e-15) conc[k] = 1e-15;
                    }

                    contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | conc );
                }
                target += log_sum_exp(contribution);

            }
        }
    """

    _basicModelSorting = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] Prf[DmNo];
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
        }

        transformed parameters {
            vector[AltNo] v = Alt * wStar;
            v ./= (1-v);
            v = log(v);
            array[AltNo, AltC] real soft_z;
            for (n in 1:AltNo)
                for (k in 1:AltC)
                    soft_z[n, k] = -log(AltC) - 0.5 * pow(altMu[k] - v[n],2);

            simplex[CNo] Prf_normalized[DmNo];
            for(i in 1:DmNo) {
                Prf_normalized[i] = Prf[i] ./ sum(Prf[i]);       
            }

        }

        model {
            // Prior
            wStar ~ dirichlet(0.01*e);
            kappaStar ~ gamma(.001,.01);
            W ~ dirichlet(kappaStar*wStar);
            kappa ~ gamma(.01,.01);
            altMu ~ normal(0,5);

            for (i in 1:DmNo){
            
                Prf_normalized[i] ~ dirichlet(kappa[i]*W[i]);  
            }

            for (n in 1:AltNo)
                target += log_sum_exp(soft_z[n]);
        }
    """

    _correlated_model = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] vector[CNo] Prf;
            vector<lower=0,upper=1>[CNo] e;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;
            
            real<lower=0> gamma_param; // parameter for gamma distribution
            real<lower=0> sigma_coef; // parameter for sigma
        }

        parameters {

            vector[CNo] wStar_eta;
            real<lower=0> kappaStar;           
        }

        transformed parameters {
            array[DmNo] simplex[CNo] W;
            for(i in 1:DmNo)
                W[i] = Prf[i] ./ sum(Prf[i]);
            
            array[DmNo] vector[CNo] W_eta;
            for(i in 1:DmNo){
                vector[CNo] logW = log(W[i]);
                real mean_logW = mean(logW);
                W_eta[i] = logW - mean_logW;  // inverse softmax with zero-mean
            }

            simplex[CNo] wStar;
            wStar = softmax(wStar_eta);
        }

        model {
            wStar_eta ~ multi_normal(mu, Sigma);

            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, sigma_coef*Sigma);
            }
        }
    """

    _correlated_model_clustering = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] vector[CNo] Prf;
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> DmC;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;

            real<lower=0> gamma_param; // parameter for gamma distribution
            real<lower=0> sigma_coef; // parameter for sigma
        }

        parameters {
            array[DmC] vector[CNo] wc_eta;
            array[DmNo] simplex[DmC] theta;
        }

        transformed parameters {
            array[DmNo] simplex[CNo] W;
            for(i in 1:DmNo)
                W[i] = Prf[i] ./ sum(Prf[i]);       

            array[DmNo] vector[CNo] W_eta;
            for(i in 1:DmNo){
                vector[CNo] logW = log(W[i]);
                real mean_logW = mean(logW);
                W_eta[i] = logW - mean_logW;  // inverse softmax with zero-mean
            }

            array[DmC] simplex[CNo] wc;
            for(i in 1:DmC)
                wc[i] = softmax(wc_eta[i]);
        }

        model {
            for (d in 1:DmC)
                wc_eta[d] ~ multi_normal(mu, Sigma);

            for (i in 1:DmNo){
                array[DmC] real contribution;
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + multi_normal_lpdf( W_eta[i] | wc_eta[j], Sigma*sigma_coef);
                target += log_sum_exp(contribution);
            }
        }
    """

    _correlatedModelSorting = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            array[DmNo] vector[CNo] Prf;
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

            vector[AltC] altMu;
        }

        transformed parameters {
            simplex[CNo] Prf_normalized[DmNo];

            for(i in 1:DmNo)
                Prf_normalized[i] = Prf[i] ./ sum(Prf[i]);       

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
            kappaStar ~ gamma(0.001,0.001);
            kappa ~ gamma(0.001, 0.001);

            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, Sigma);

                Prf_normalized[i] ~ dirichlet(kappa[i]*W[i]);  
            }

            for (n in 1:AltNo)
                target += log_sum_exp(soft_z[n]);

        }
    """

    def __init__(self, Prf, alternatives = None, dm_cluster_number=-1, alt_sort_number=-1, num_chain=3, num_samples=1000, opt={}):
        self.Prf = np.array(Prf)
        
        super().__init__(alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples, opt)


    @property
    def input_data(self):
        data = self._get_common_data()
        data['Prf'] = self.Prf
        data['e'] =  np.ones(self.criteria_no)

        return data

    @property
    def original_model(self):
        return self.__originalModel

    @property
    def dm_no(self):
        return self.Prf.shape[0]

    @property
    def criteria_no(self):
        return self.Prf.shape[1]

    def _check_input_data(self):
        assert self.Prf.shape[0] >= 1, "No input"
        assert self.Prf.shape[1] >= 2, "The number of criteria must be more than 2!"

        return True


if __name__ == "__main__":
    
    prf =  np.array([
        [ 7, 6, 4, 9, 5, 8, 1, 3],
        [ 9, 8, 2, 5, 4, 5, 1, 3],
        [ 8, 8, 5, 9, 5, 5, 1, 2],
        [ 8, 9, 2, 8, 1, 8, 2, 2],
        [ 8, 6, 1, 9, 6, 7, 4, 4],
        [ 9, 8, 1, 9, 7, 5, 5, 6],
        ])

    dmNo, cNo = prf.shape
    altNo = 50
    x = np.random.rand(altNo // 2, cNo)
    altMat = np.concatenate([x*10,x])

    opt = {'CriteriaDependence': False, 'Sigma': np.eye(cNo) }

    swing = StandardSWING( Prf=prf*100, opt=opt, alternatives=altMat, alt_sort_number=2)#, dm_cluster_number=2)
    swing.sampling()
    print('Ok')

