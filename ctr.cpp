#include "ctr.h"
#include "opt.h"

extern gsl_rng * RANDOM_NUMBER;
int min_iter = 15;

c_ctr::c_ctr() {
  m_beta = NULL;
  m_theta = NULL;
  m_U = NULL;
  m_V = NULL;
  m_S = NULL;

  m_num_factors = 0; // m_num_topics
  m_num_items = 0; // m_num_docs
  m_num_users = 0; // num of users
}

c_ctr::~c_ctr() {
  // free memory
  if (m_beta != NULL) gsl_matrix_free(m_beta);
  if (m_theta != NULL) gsl_matrix_free(m_theta);
  if (m_U != NULL) gsl_matrix_free(m_U);
  if (m_V != NULL) gsl_matrix_free(m_V);
}

void c_ctr::read_init_information(const char* theta_init_path, 
                                  const char* beta_init_path,
                                  const c_corpus* c) {
  int num_topics = m_num_factors;
  m_theta = gsl_matrix_alloc(c->m_num_docs, num_topics);
  printf("\nreading theta initialization from %s\n", theta_init_path);
  FILE * f = fopen(theta_init_path, "r");
  mtx_fscanf(f, m_theta);
  fclose(f);

  //normalize m_theta, in case it's not
  for (size_t j = 0; j < m_theta->size1; j ++) {
    gsl_vector_view theta_v = gsl_matrix_row(m_theta, j);
    vnormalize(&theta_v.vector);
  }

  m_beta = gsl_matrix_alloc(num_topics, c->m_size_vocab);
  printf("reading beta initialization from %s\n", beta_init_path);
  f = fopen(beta_init_path, "r");
  mtx_fscanf(f, m_beta);
  fclose(f);

  // exponentiate if it's not
  if (mget(m_beta, 0, 0) < 0)
    mtx_exp(m_beta);
}

void c_ctr::set_model_parameters(int num_factors, 
                                 int num_users, 
                                 int num_items) {
  m_num_factors = num_factors;
  m_num_users = num_users;
  m_num_items = num_items;
}

void c_ctr::init_model(int ctr_run) {

  // user factors
  m_U = gsl_matrix_calloc(m_num_users, m_num_factors);

  // item factors
  m_V = gsl_matrix_calloc(m_num_items, m_num_factors);

  // social factors
  m_S = gsl_matrix_calloc(m_num_users, m_num_factors);

  if (ctr_run) {
    gsl_matrix_memcpy(m_V, m_theta);
  }
  else {
    // this is for convenience, so that updates are similar.
    m_theta = gsl_matrix_calloc(m_num_items, m_num_factors);

    // initialize the item factors uniformly?
    for (size_t i = 0; i < m_V->size1; i ++) 
      for (size_t j = 0; j < m_V->size2; j ++) 
        mset(m_V, i, j, runiform());
  }
}

void c_ctr::learn_map_estimate_social(const c_data* users, const c_data* items,
        const c_data* followers, const c_corpus* c,
                                      const ctr_hyperparameter* param,
                                      const char* directory) {


  // init model parameters
  printf("\ninitializing the model ...\n");
  init_model(param->ctr_run);

  // filename
  char name[500];

  // start time
  time_t start, current;
  time(&start);
  int elapsed = 0;

  int iter = 0;
  double likelihood = -exp(50), likelihood_old;
  double converge = 1.0;

  /// create the state log file 
  sprintf(name, "%s/state.log", directory);
  FILE* file = fopen(name, "w");
  fprintf(file, "iter time likelihood converge\n");


  /* alloc auxiliary variables */
  gsl_matrix* XX = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* A  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* B  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* C  = gsl_matrix_alloc(m_num_factors, m_num_factors);

  gsl_matrix* UU  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* VV  = gsl_matrix_alloc(m_num_factors, m_num_factors);

  gsl_matrix* UU_copy  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* VV_copy  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  
  gsl_matrix* S  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* SS  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* SS_copy  = gsl_matrix_alloc(m_num_factors, m_num_factors);

  gsl_vector* x  = gsl_vector_alloc(m_num_factors);
  gsl_vector* y  = gsl_vector_alloc(m_num_factors);

  gsl_matrix* phi = NULL;
  gsl_matrix* word_ss = NULL;
  gsl_matrix* log_beta = NULL;
  gsl_vector* gamma = NULL;

  if (param->ctr_run && param->theta_opt) {
    int max_len = c->max_corpus_length();
    phi = gsl_matrix_calloc(max_len, m_num_factors);
    word_ss = gsl_matrix_calloc(m_num_factors, c->m_size_vocab);
    log_beta = gsl_matrix_calloc(m_num_factors, c->m_size_vocab);
    gsl_matrix_memcpy(log_beta, m_beta);
    mtx_log(log_beta);
    gamma = gsl_vector_alloc(m_num_factors);
  }

  /* tmp variables for indexes */
  int i, j, m, n, l, k;
  int* item_ids; 
  int* user_ids;
  int* follower_ids;

  double result;

  /// confidence parameters
  double a_minus_b = param->a - param->b;

  while ((iter < param->max_iter and converge > 1e-6 ) or iter < min_iter) {

    likelihood_old = likelihood;
    likelihood = 0.0;

    // SS^T
    gsl_matrix_set_zero(SS);

    // lambda_q * S * diag(b) * S^T
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, param->b*param->lambda_q, m_S, m_S, 0., SS);

    gsl_matrix_set_zero(VV);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, param->b, m_V, m_V, 0., VV);
    gsl_matrix_add_diagonal(VV, param->lambda_u); 

    for (int i = 0; i < m_num_users; ++i) {

        gsl_matrix_memcpy(SS_copy, SS);
        gsl_matrix_add(SS_copy, VV);

        gsl_vector_set_zero(x);

        int* follower_ids = followers->m_vec_data[i];
        int n_followers = followers->m_vec_len[i];
                                                                
        if (n_followers > 0) {
            for (int l = 0; l < n_followers; ++l) {

                int j = follower_ids[l];

                gsl_vector_const_view s = gsl_matrix_const_row(m_S, j); 
                gsl_blas_dger(param->lambda_q*a_minus_b, &s.vector, &s.vector, SS_copy); 
                gsl_blas_daxpy(param->lambda_q*param->a, &s.vector, x);
            }
        }

        int* item_ids = users->m_vec_data[i];
        int n_ratings = users->m_vec_len[i];

        if (n_ratings > 0) {
            for (int l = 0; l < n_ratings; l ++) {

                int j = item_ids[l];

                gsl_vector_const_view v = gsl_matrix_const_row(m_V, j); 
                gsl_blas_dger(a_minus_b, &v.vector, &v.vector, SS_copy); 
                gsl_blas_daxpy(param->a, &v.vector, x);
            }

        }

        //printf("n_followers=%d, n_ratings = %d\n", n_followers, n_ratings);
        gsl_vector_view u = gsl_matrix_row(m_U, i);

        if (n_followers > 0 or n_ratings > 0) {
            matrix_vector_solve(SS_copy, x, &(u.vector));
        }

        // update the likelihood
        gsl_blas_ddot(&u.vector, &u.vector, &result);
        likelihood += -0.5 * param->lambda_u * result;
    }

    // update V
    if (param->ctr_run && param->theta_opt) gsl_matrix_set_zero(word_ss);

    gsl_matrix_set_zero(XX);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, param->b, m_U, m_U, 0., XX);

    for (int j = 0; j < m_num_items; ++j) {

        gsl_vector_view v = gsl_matrix_row(m_V, j);
        gsl_vector_view theta_v = gsl_matrix_row(m_theta, j);

        int* user_ids = items->m_vec_data[j];
        int m = items->m_vec_len[j];
        if (m > 0) {

            gsl_matrix_memcpy(A, XX);
            gsl_vector_set_zero(x);

            for(int l = 0; l < m; l ++) {
                int i = user_ids[l];

                gsl_vector_const_view u = gsl_matrix_const_row(m_U, i);  
                gsl_blas_dger(a_minus_b, &u.vector, &u.vector, A);
                gsl_blas_daxpy(param->a, &u.vector, x);
                
            }

            // adding the topic vector
            // even when ctr_run=0, m_theta=0
            gsl_blas_daxpy(param->lambda_v, &theta_v.vector, x);

            gsl_matrix_memcpy(B, A); // save for computing likelihood 

            // here different from U update
            gsl_matrix_add_diagonal(A, param->lambda_v);  
            matrix_vector_solve(A, x, &v.vector);

            // sum_ij { cij/2 (rij - ui^t*vj)}
            // update the likelihood for the relevant part
            // assume every prediction was correct
            likelihood += -0.5 * m * param->a;
            for (l = 0; l < m; l ++) {
                i = user_ids[l];
                gsl_vector_const_view u = gsl_matrix_const_row(m_U, i);  
                gsl_blas_ddot(&u.vector, &v.vector, &result);

                // adjust for incorrect predictions
                likelihood += param->a * result;
            }

            //likelihood += -0.5 * mahalanobis_prod(B, &v.vector, &v.vector);
            // likelihood part of theta, even when theta=0, which is a
            // special case
            gsl_vector_memcpy(x, &v.vector);
            gsl_vector_sub(x, &theta_v.vector);
            gsl_blas_ddot(x, x, &result);
            likelihood += -0.5 * param->lambda_v * result;

            if (param->ctr_run && param->theta_opt) {
                const c_document* doc =  c->m_docs[j];
                likelihood += doc_inference(doc, &theta_v.vector, log_beta, phi, gamma, word_ss, true); 
                optimize_simplex(gamma, &v.vector, param->lambda_v, &theta_v.vector); 
            }
        }
        else {
            // m=0, this article has never been rated
            if (param->ctr_run && param->theta_opt) {
                const c_document* doc =  c->m_docs[j];
                doc_inference(doc, &theta_v.vector, log_beta, phi, gamma, word_ss, false); 
                vnormalize(gamma);
                gsl_vector_memcpy(&theta_v.vector, gamma);
            }
        }
    }

    // update S 
    gsl_matrix_set_zero(UU);

    //  U * diag(b) * U^T + lambda_q * I
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, param->lambda_q*param->b, m_U, m_U, 0., UU);
    gsl_matrix_add_diagonal(UU, param->lambda_q); 

    for (int i = 0; i < m_num_users; ++i) {

        gsl_matrix_memcpy(A, UU);
        gsl_vector_set_zero(x);

        int* follower_ids = followers->m_vec_data[i];
        int n_followers = followers->m_vec_len[i];

        // social relations?
        if (n_followers > 0) {

            for (int l = 0; l < n_followers; l ++) {

                int j = follower_ids[l];

                gsl_vector_const_view u = gsl_matrix_const_row(m_U, j); 
                gsl_blas_dger(param->lambda_q*a_minus_b, &u.vector, &u.vector, A); 
                gsl_blas_daxpy(param->lambda_q*param->a, &u.vector, x);

            }
        }

        gsl_vector_view s = gsl_matrix_row(m_S, i); 
        matrix_vector_solve(A, x, &s.vector);
    }

    // update likelihood for S factors
    for(int i = 0; i < m_num_users; ++i) {
        gsl_vector_const_view s = gsl_matrix_const_row(m_S, i); 
        gsl_blas_ddot(&s.vector, &s.vector, &result);
        likelihood += -param->lambda_q*0.5 * result;
    }



    // what the fuck?
    for(int i = 0; i < m_num_users; ++i) {
        gsl_vector_const_view u = gsl_matrix_const_row(m_U, i); 
        
        for(int m = 0; m < m_num_users; ++m) {
            gsl_vector_const_view s = gsl_matrix_const_row(m_S, m); 
            gsl_blas_ddot(&u.vector, &s.vector, &result);

            likelihood += -param->lambda_q*0.25*param->b*result*result;
        }

        int* follower_ids = followers->m_vec_data[i];
        int n_followers = followers->m_vec_len[i];

        for(int l = 0; l < n_followers; ++l) {

            int j = follower_ids[l];
            gsl_vector_const_view s = gsl_matrix_const_row(m_S, j); 
            gsl_blas_ddot(&u.vector, &s.vector, &result);
            likelihood += -param->lambda_q*0.25*param->a + 2*param->lambda_q*0.25*param->a * result
                          -param->lambda_q*0.25*a_minus_b * result * result;
        }


    }
    
    // update beta if needed
    if (param->ctr_run && param->theta_opt) {
        gsl_matrix_memcpy(m_beta, word_ss);
        for (k = 0; k < m_num_factors; k ++) {
            gsl_vector_view row = gsl_matrix_row(m_beta, k);
            vnormalize(&row.vector);
        }
        gsl_matrix_memcpy(log_beta, m_beta);
        mtx_log(log_beta);
    }

    time(&current);
    elapsed = (int)difftime(current, start);

    iter++;
    converge = fabs((likelihood-likelihood_old)/likelihood_old);

    if (likelihood < likelihood_old) printf("likelihood is decreasing!\n");

    fprintf(file, "%04d %06d %10.5f %.10f\n", iter, elapsed, likelihood, converge);
    fflush(file);
    printf("iter=%04d, time=%06d, likelihood=%.5f, converge=%.10f\n", iter, elapsed, likelihood, converge);

    // save intermediate results
    if (iter % param->save_lag == 0) {

        sprintf(name, "%s/%04d-U.dat", directory, iter);
        FILE * file_U = fopen(name, "w");
        mtx_fprintf(file_U, m_U);
        fclose(file_U);

        sprintf(name, "%s/%04d-V.dat", directory, iter);
        FILE * file_V = fopen(name, "w");
        mtx_fprintf(file_V, m_V);
        fclose(file_V);

        if (param->ctr_run) { 
            sprintf(name, "%s/%04d-theta.dat", directory, iter);
            FILE * file_theta = fopen(name, "w");
            mtx_fprintf(file_theta, m_theta);
            fclose(file_theta);

            sprintf(name, "%s/%04d-beta.dat", directory, iter);
            FILE * file_beta = fopen(name, "w");
            mtx_fprintf(file_beta, m_beta);
            fclose(file_beta);
        }
    }
  }

  // save final results
  sprintf(name, "%s/final-U.dat", directory);
  FILE * file_U = fopen(name, "w");
  mtx_fprintf(file_U, m_U);
  fclose(file_U);

  sprintf(name, "%s/final-V.dat", directory);
  FILE * file_V = fopen(name, "w");
  mtx_fprintf(file_V, m_V);
  fclose(file_V);

  if (param->ctr_run) { 
      sprintf(name, "%s/final-theta.dat", directory);
      FILE * file_theta = fopen(name, "w");
      mtx_fprintf(file_theta, m_theta);
      fclose(file_theta);

      sprintf(name, "%s/final-beta.dat", directory);
      FILE * file_beta = fopen(name, "w");
      mtx_fprintf(file_beta, m_beta);
      fclose(file_beta);
  }

  // free memory
  gsl_matrix_free(XX);
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_vector_free(x);

  if (param->ctr_run && param->theta_opt) {
      gsl_matrix_free(phi);
      gsl_matrix_free(log_beta);
      gsl_matrix_free(word_ss);
      gsl_vector_free(gamma);
  }
}

double c_ctr::doc_inference(const c_document* doc, const gsl_vector* theta_v, 
        const gsl_matrix* log_beta, gsl_matrix* phi,
        gsl_vector* gamma, gsl_matrix* word_ss, 
        bool update_word_ss) {

    double likelihood = 0;
    gsl_vector* log_theta_v = gsl_vector_alloc(theta_v->size);
    gsl_vector_memcpy(log_theta_v, theta_v);
    vct_log(log_theta_v);

    int n, k, w;
    double x;
    for (n = 0; n < doc->m_length; n ++) {
        w = doc->m_words[n]; 
        for (k = 0; k < m_num_factors; k ++)
            mset(phi, n, k, vget(theta_v, k) * mget(m_beta, k, w));

        gsl_vector_view row =  gsl_matrix_row(phi, n);
        vnormalize(&row.vector);

        for (k = 0; k < m_num_factors; k ++) {
            x = mget(phi, n, k);
            if (x > 0) 
                likelihood += x*(vget(log_theta_v, k) + mget(log_beta, k, w) - log(x));
        }
    }

    gsl_vector_set_all(gamma, 1.0); // smoothing with small pseudo counts
    for (n = 0; n < doc->m_length; n ++) {
        for (k = 0; k < m_num_factors; k ++) {
            x = doc->m_counts[n] * mget(phi, n, k);
            vinc(gamma, k, x);      
            if (update_word_ss) minc(word_ss, k, doc->m_words[n], x);
        }
    }

    gsl_vector_free(log_theta_v);
    return likelihood;
}


//void c_ctr::learn_map_estimate_social(const c_data* users, const c_data* items,
//        const c_data* followers, const c_corpus* c,
//                                      const ctr_hyperparameter* param,
//                                      const char* directory) {
//
//  // init model parameters
//  printf("\ninitializing the model ...\n");
//  init_model(param->ctr_run);
//
//  // filename
//  char name[500];
//
//  // start time
//  time_t start, current;
//  time(&start);
//  int elapsed = 0;
//
//  int iter = 0;
//  double likelihood = -exp(50), likelihood_old;
//  double converge = 1.0;
//
//  /// create the state log file 
//  sprintf(name, "%s/state.log", directory);
//  FILE* file = fopen(name, "w");
//  fprintf(file, "iter time likelihood converge\n");
//
//
//  /* alloc auxiliary variables */
//  gsl_matrix* XX = gsl_matrix_alloc(m_num_factors, m_num_factors);
//  gsl_matrix* A  = gsl_matrix_alloc(m_num_factors, m_num_factors);
//  gsl_matrix* B  = gsl_matrix_alloc(m_num_factors, m_num_factors);
//  gsl_matrix* C  = gsl_matrix_alloc(m_num_factors, m_num_factors);
//  gsl_matrix* S  = gsl_matrix_alloc(m_num_factors, m_num_factors);
//
//  gsl_vector* x  = gsl_vector_alloc(m_num_factors);
//  gsl_vector* y  = gsl_vector_alloc(m_num_factors);
//
//  gsl_matrix* phi = NULL;
//  gsl_matrix* word_ss = NULL;
//  gsl_matrix* log_beta = NULL;
//  gsl_vector* gamma = NULL;
//
//  if (param->ctr_run && param->theta_opt) {
//    int max_len = c->max_corpus_length();
//    phi = gsl_matrix_calloc(max_len, m_num_factors);
//    word_ss = gsl_matrix_calloc(m_num_factors, c->m_size_vocab);
//    log_beta = gsl_matrix_calloc(m_num_factors, c->m_size_vocab);
//    gsl_matrix_memcpy(log_beta, m_beta);
//    mtx_log(log_beta);
//    gamma = gsl_vector_alloc(m_num_factors);
//  }
//
//  /* tmp variables for indexes */
//  int i, j, m, n, l, k, h;
//  int* item_ids; 
//  int* user_ids;
//  int* follower_ids;
//
//  double result;
//
//  /// confidence parameters
//  double a_minus_b = param->a - param->b;
//
//  while ((iter < param->max_iter and converge > 1e-6 ) or iter < min_iter) {
//
//    likelihood_old = likelihood;
//    likelihood = 0.0;
//
//    gsl_matrix_set_zero(S);
//    gsl_matrix_set_zero(XX);
//
//    // lambda_q * S * diag(b) * S^T
//    gsl_blas_dgemm(CblasTrans, CblasNoTrans, param->b*param->lambda_q, m_S, m_S, 0., S);
//
//    //  V * diag(b) * V^T + lambda_u * I
//    gsl_blas_dgemm(CblasTrans, CblasNoTrans, param->b, m_V, m_V, 0., XX);
//    gsl_matrix_add_diagonal(XX, param->lambda_u); 
//
//    for (i = 0; i < m_num_users; i ++) {
//    //for (int i = 0; i < m_num_users; i++) {
//
//      //int* item_ids = users->m_vec_data[i];
//      //int* follower_ids = followers->m_vec_data[i];
//
//      gsl_matrix_memcpy(C, S);
//      gsl_matrix_memcpy(A, XX);
//
//      item_ids = users->m_vec_data[i];
//      follower_ids = followers->m_vec_data[i];
//
//      int n_ratings = users->m_vec_len[i];
//      int n_followers = followers->m_vec_len[i];
//
//      if (n_followers > 0) {
//        // social relations?
//        for (l = 0; l < n_followers; l ++) {
//            // int j = follower_ids[l];
//            j = follower_ids[l];
//            gsl_vector_const_view s = gsl_matrix_const_row(m_S, j); 
//            gsl_blas_dger(a_minus_b, &s.vector, &s.vector, C); 
//            gsl_blas_daxpy(param->lambda_q*param->a, &s.vector, x);
//        }
//      }
//
//      if (n_ratings > 0) {
//        // this user has rated some articles
//        for (l = 0; l < n_ratings; l++) {
//            j = item_ids[l];
//            gsl_vector_const_view v = gsl_matrix_const_row(m_V, j); 
//            gsl_blas_dger(a_minus_b, &v.vector, &v.vector, A); 
//            gsl_blas_daxpy(param->a, &v.vector, x);
//        }
//
//        // how about truncated CG here :]
//        gsl_vector_view u = gsl_matrix_row(m_U, i);
//        gsl_matrix_add(A, C);
//        matrix_vector_solve(A, x, &(u.vector));
//
//        // update the likelihood
//        gsl_blas_ddot(&u.vector, &u.vector, &result);
//        likelihood += -0.5 * param->lambda_u * result;
//      }
//    }
//
//    // update V
//    if (param->ctr_run && param->theta_opt) {
//        gsl_matrix_set_zero(word_ss);
//    }
//
//
//    // we should really re-calculate UU^T at the end:|
//    gsl_matrix_set_zero(XX);
//
//    //  U * diag(b) * U^T + lambda_v * I
//    gsl_blas_dgemm(CblasTrans, CblasNoTrans, param->b, m_U, m_U, 0., XX);
//    gsl_matrix_memcpy(C, XX);
//
//    gsl_matrix_scale(C, param->lambda_q);
//    gsl_matrix_add_diagonal(C, param->lambda_q);   // should be lambda_s
//    gsl_matrix_add_diagonal(XX, param->lambda_v); 
//
//    for (i = 0; i < m_num_users; ++i) {
//
//        gsl_vector_set_zero(x);
//        follower_ids = followers->m_vec_data[i];
//        m = followers->m_vec_len[i];
//
//        for (l=0; l < m; ++l) {
//            j = follower_ids[l];
//            gsl_vector_const_view u = gsl_matrix_const_row(m_U, j); 
//            gsl_blas_daxpy(param->lambda_q*param->a, &u.vector, x);
//        }
//
//        gsl_vector_view s = gsl_matrix_row(m_S, i);
//        matrix_vector_solve(C, x, &(s.vector));
//    } 
//
//
//    gsl_matrix_scale(XX, param->b);
//
//    for (j = 0; j < m_num_items; j ++) {
//
//        gsl_matrix_memcpy(A, XX);
//
//        gsl_vector_view v = gsl_matrix_row(m_V, j);
//        gsl_vector_view theta_v = gsl_matrix_row(m_theta, j);
//
//        user_ids = items->m_vec_data[j];
//        m = items->m_vec_len[j];
//
//        if (m > 0) {
//
//            // m > 0, some users have rated this article
//            gsl_vector_set_zero(x);
//
//            for (l = 0; l < m; l ++) {
//                i = user_ids[l];
//                gsl_vector_const_view u = gsl_matrix_const_row(m_U, i);  
//                gsl_blas_dger(a_minus_b, &u.vector, &u.vector, A);
//                gsl_blas_daxpy(param->a, &u.vector, x);
//            }
//
//            // adding the topic vector
//            // even when ctr_run=0, m_theta=0
//            gsl_blas_daxpy(param->lambda_v, &theta_v.vector, x);
//            gsl_matrix_memcpy(B, A); // save for computing likelihood 
//
//            // here different from U update
//            gsl_matrix_add_diagonal(A, param->lambda_v);  
//            matrix_vector_solve(A, x, &v.vector);
//
//            // sum_ij { cij/2 (rij - ui^t*vj)}
//            // update the likelihood for the relevant part
//            // assume every prediction was recorded
//            likelihood += -0.5 * m * param->a;
//            for (l = 0; l < m; l ++) {
//                i = user_ids[l];
//                gsl_vector_const_view u = gsl_matrix_const_row(m_U, i);  
//                gsl_blas_ddot(&u.vector, &v.vector, &result);
//
//                // adjust for incorrect predictions
//                likelihood += param->a * result;
//        }
//
//        // is this some type of regularization parameter?
//        likelihood += -0.5 * mahalanobis_prod(B, &v.vector, &v.vector);
//
//        // likelihood part of theta, even when theta=0, which is a
//        // special case
//        gsl_vector_memcpy(x, &v.vector);
//        gsl_vector_sub(x, &theta_v.vector);
//        gsl_blas_ddot(x, x, &result);
//
//        likelihood += -0.5 * param->lambda_v * result;
//        
//        if (param->ctr_run && param->theta_opt) {
//            const c_document* doc =  c->m_docs[j];
//            likelihood += doc_inference(doc, &theta_v.vector, log_beta, phi, gamma, word_ss, true); 
//            optimize_simplex(gamma, &v.vector, param->lambda_v, &theta_v.vector); 
//        }
//      }
//      else {
//            // m=0, this article has never been rated
//            if (param->ctr_run && param->theta_opt) {
//                const c_document* doc =  c->m_docs[j];
//                doc_inference(doc, &theta_v.vector, log_beta, phi, gamma, word_ss, false); 
//                vnormalize(gamma);
//                gsl_vector_memcpy(&theta_v.vector, gamma);
//            }
//        }
//    }
//
//    // update beta if needed
//    if (param->ctr_run && param->theta_opt) {
//        gsl_matrix_memcpy(m_beta, word_ss);
//        for (k = 0; k < m_num_factors; k ++) {
//            gsl_vector_view row = gsl_matrix_row(m_beta, k);
//            vnormalize(&row.vector);
//        }
//        gsl_matrix_memcpy(log_beta, m_beta);
//        mtx_log(log_beta);
//    }
//
//
//    // lambda_s/2 * sum_k (s_k^T s_k)
//    for(i = 0; i < m_num_users; ++i) {
//        gsl_vector_const_view s = gsl_matrix_const_row(m_S, i); 
//        gsl_blas_ddot(&s.vector, &s.vector, &result);
//        likelihood += -0.5*param->lambda_s*result;
//    }
//
//    // lambda_q/2 sum_im {d_im*0.5 (qim-ui^tsm)^2}
//    for(i = 0; i < m_num_users; ++i) {
//        gsl_vector_const_view u = gsl_matrix_const_row(m_U, i); 
//        likelihood += -0.5*param->lambda_s*result;
//
//        follower_ids = followers->m_vec_data[i];
//        h = followers->m_vec_len[i];
//
//        for(l = 0; l < h; ++l) {
//            j = followers->m_vec_len[l];
//            gsl_vector_const_view s = gsl_matrix_const_row(m_S, j); 
//            gsl_blas_ddot(&u.vector, &s.vector, &result);
//            result = result*result;
//            likelihood += param->lambda_q*0.5*0.5*result;
//   
//        }
//    }
//
//    time(&current);
//    elapsed = (int)difftime(current, start);
//
//    iter++;
//    converge = fabs((likelihood-likelihood_old)/likelihood_old);
//
//    if (likelihood < likelihood_old) printf("likelihood is decreasing!\n");
//
//    fprintf(file, "%04d %06d %10.5f %.10f\n", iter, elapsed, likelihood, converge);
//    fflush(file);
//    printf("iter=%04d, time=%06d, likelihood=%.5f, converge=%.10f\n", iter, elapsed, likelihood, converge);
//
//    // save intermediate results
//    if (iter % param->save_lag == 0) {
//
//      sprintf(name, "%s/%04d-U.dat", directory, iter);
//      FILE * file_U = fopen(name, "w");
//      mtx_fprintf(file_U, m_U);
//      fclose(file_U);
//
//      sprintf(name, "%s/%04d-V.dat", directory, iter);
//      FILE * file_V = fopen(name, "w");
//      mtx_fprintf(file_V, m_V);
//      fclose(file_V);
//      
//      sprintf(name, "%s/%04d-S.dat", directory, iter);
//      FILE * file_S = fopen(name, "w");
//      mtx_fprintf(file_S, m_S);
//      fclose(file_S);
//
//      if (param->ctr_run) { 
//        sprintf(name, "%s/%04d-theta.dat", directory, iter);
//        FILE * file_theta = fopen(name, "w");
//        mtx_fprintf(file_theta, m_theta);
//        fclose(file_theta);
//
//        sprintf(name, "%s/%04d-beta.dat", directory, iter);
//        FILE * file_beta = fopen(name, "w");
//        mtx_fprintf(file_beta, m_beta);
//        fclose(file_beta);
//      }
//    }
//  }
//
//  // save final results
//  sprintf(name, "%s/final-U.dat", directory);
//  FILE * file_U = fopen(name, "w");
//  mtx_fprintf(file_U, m_U);
//  fclose(file_U);
//
//  sprintf(name, "%s/final-V.dat", directory);
//  FILE * file_V = fopen(name, "w");
//  mtx_fprintf(file_V, m_V);
//  fclose(file_V);
//
//  sprintf(name, "%s/final-S.dat", directory);
//  FILE * file_S = fopen(name, "w");
//  mtx_fprintf(file_S, m_S);
//  fclose(file_S);
//
//  if (param->ctr_run) { 
//    sprintf(name, "%s/final-theta.dat", directory);
//    FILE * file_theta = fopen(name, "w");
//    mtx_fprintf(file_theta, m_theta);
//    fclose(file_theta);
//
//    sprintf(name, "%s/final-beta.dat", directory);
//    FILE * file_beta = fopen(name, "w");
//    mtx_fprintf(file_beta, m_beta);
//    fclose(file_beta);
//  }
//
//  // free memory
//  gsl_matrix_free(XX);
//  gsl_matrix_free(A);
//  gsl_matrix_free(B);
//  gsl_vector_free(x);
//
//  if (param->ctr_run && param->theta_opt) {
//    gsl_matrix_free(phi);
//    gsl_matrix_free(log_beta);
//    gsl_matrix_free(word_ss);
//    gsl_vector_free(gamma);
//  }
//}
//


