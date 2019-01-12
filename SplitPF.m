function [x_est, x_P_resamp, P_w, x_P] = SplitPF(A, B, C, u, t, z, x, N, x_P, PZR_intl_cov, PZR_proc_cov, PZR_msmt_cov,...
                                        M_intl_cov, M_proc_cov, M_proc_range, local_ratio)

% convert covariances to std devs
PZR_intl_sigma = sqrt(PZR_intl_cov);
M_intl_sigma = sqrt(M_intl_cov);
PZR_proc_sigma = sqrt(PZR_proc_cov);
M_proc_sigma = sqrt(M_proc_cov); 
PZR_msmt_sigma = sqrt(PZR_msmt_cov);

for i = 1:N
   if t == 1
       % for first time step, initialize particles
       x_P(2,1,i) = x(2) + M_intl_sigma*randn;
       if x_P(2,1,i) < 0
           x_P(2,1,i) = 0;
       end
       x_P(1,1,i) = x(1) + PZR_intl_sigma*randn;
       z_update(i) = C*x_P(:,1,i);
       P_w_unn(1,1,i) = exp( -0.5 * ( z - z_update(i) )^2 / PZR_msmt_sigma^2) / PZR_msmt_sigma / sqrt(2*pi);
   else
       R = rand;
       % mutate particles based on mutation probability
       if R <= local_ratio
           x_P(2,1,i) = x_P(2,1,i) + M_proc_sigma*randn;
           if x_P(2,1,i) < 0
               x_P(2,1,i) = 0;
           end
           x_P(1,1,i) = A(1,:)*x_P(:,1,i) + B(1)*u + PZR_proc_sigma*randn;
           z_update(i) = C*x_P(:,1,i);
           P_w_unn(1,1,i) = exp( -0.5 * ( z - z_update(i) )^2 / PZR_msmt_sigma^2) / PZR_msmt_sigma / sqrt(2*pi);
       else
           x_P(2,1,i) = M_proc_range*rand;
           x_P(1,1,i) = A(1,:)*x_P(:,1,i) + B(1)*u + PZR_proc_sigma*randn*0;          
           z_update(i) = C*x_P(:,1,i);
           P_w_unn(1,1,i) = exp( -0.5 * ( z - z_update(i) )^2 / PZR_msmt_sigma^2) / PZR_msmt_sigma / sqrt(2*pi);

       end
   end
   % make sure particle weights dont become 0, causes numerical issues
   if P_w_unn(1,1,i) == 0
       P_w_unn(1,1,i) = eps;
   end
       
end
% normalize particle weights
P_w(1,1,:) = P_w_unn/sum(P_w_unn);

% resample the particles based on their weights (multinomial resampling)
index_vector = [1:N]';
resamp_indx_vec = randsample(index_vector, N, true, P_w);

for i = 1:N
    x_P_resamp(:,1,i) = x_P(:,1,resamp_indx_vec(i));
end

% calculate an estimate from the resampled particles
x_est(1) = mean(x_P_resamp(1,1,:));
x_est(2) = mean(x_P_resamp(2,1,:));



