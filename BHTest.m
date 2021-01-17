function [leak_prob, leak_magnitude, P_z_Z_H] = BHTest(leak_prob_prev, var_msmt, z, x_est, m_leak_dot_vec, N, P_w_unn)

% This script is not used in the SAMPF, only for MMPF and IMMPF

M = length(m_leak_dot_vec);
sigma_measurement = sqrt(var_msmt);
leak_prob = zeros(M,1);

for m = 1:M
    %P_z_Z_H = exp(-0.5*(z - x_est(m))^2/cov(m))/sqrt(2*pi*cov(m));
    %P_z_Z_H = exp(-0.5*(z - x_est(m))^2/sigma_measurement^2)/sigma_measurement/sqrt(2*pi);
    %P_z_Z_H = (1/N)*sum(P_w_unn(:,m));
    P_z_Z_H(m,1) = mean(P_w_unn(:,m));
    leak_prob(m,1) = P_z_Z_H(m,1)*leak_prob_prev(m,1);
    
%     if leak_prob(m,1) >= 0 && leak_prob(m,1) <= 10^-14
%         leak_prob(m,1) = 10^-13;
%     end
% 
    if leak_prob(m,1) >= 0 && leak_prob(m,1) < 10^-10
        leak_prob(m,1) = 10^-10;
    end

%     if leak_prob(m,1) >= 0 && leak_prob(m,1) < 10^-6
%         leak_prob(m,1) = 10^-6;
%     end
end

% normalize leak probabilities
leak_prob = leak_prob/sum(leak_prob);

% calculate magnitude of leak
leak_magnitude = dot(leak_prob, m_leak_dot_vec);