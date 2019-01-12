% Cameron Brown

clear all
close all
clc

%% Load in GSE data and define parameters

format compact
rng(422); %422 is the seed used for the thesis results

load('9_27_18_Trial1Final.mat'); % load data from 5% leak during steady state
%load('12_20_18_Trial1Final.mat');
load('SystemIDResults.mat'); % load model constants from system ID
data.thwbreakf16 = data.thwbreakf16*60./data.rho_cl_l_a/0.133681; %converting leak magnitude from lbm/s to gpm

datasize = size(data.time); 
T = datasize(1); % time length of data

N = 10000; % The total number of particles the PF uses
local_srch_ratio = .99; % ratio of particles that will search locally
global_srch_ratio = 1 - local_srch_ratio; % ratio of particles that will search globally

h = 0.5; % sample time of the discrete system

PZR_msmt_cov = 0.0054; % covariance of the measurement noise of PZR level
PZR_proc_cov = 0.0014; % covariance of the process noise of the PZR level
PZR_intl_cov = PZR_proc_cov; % covariance of the particle distribution around initial PZR level

% M = leak magnitude
M_msmt_local_cov = 0; % covariance of the measurement noise of leak magnitude for local search
M_proc_local_cov = 100; % covariance of the process noise of leak magnitude for local search
M_msmt_global_cov = M_msmt_local_cov; % covariance of measurement noise of leak magnitude for global search
M_proc_global_range = 10000; % range of the uniform distribution of the leak magnitude particles in the global search
M_intl_cov = M_proc_local_cov;

% continuous time constants for the pressurizer model
c1 = vhat(1);
c2 = vhat(2);
c3 = vhat(3);
c4 = vhat(4);
c5 = vhat(5);

% effective and net mass flow rates (input u is the sum of these)
m_eff_dot = (c1*data.drhodt_pzr_l + c2*data.drhodt_pzr_v + c3*data.drhodt_hl_l + c4*data.drhodt_cl_l)/c5;
m_net_dot = data.mnet_cvcs;

%% Declare variables

x = zeros(2,T); % actual state vector
z = zeros(1,T); % measurement vector

x_est = zeros(2, T); % estimated state vector

P_w_unn = zeros(N,1); % unnormalized particle weights
P_w = zeros(N,1); % normalized particle weights

x_P = zeros(2,1,N); % initialize particles

C_CT = [1 0]; % C is a constant

% output vectors just in case we need them
x_out = [];
Pw_out = [];
x_est_out = [];
u_out = [];
x_P_out = [];
x_P_pre_out = [];

%% Begin recursive algorithm
tic
for t=1:T
    % define continuous time state-space matrices inside loop because they
    % are time-varying
    a11 = -(data.drhodt_pzr_l(t) - data.drhodt_pzr_v(t)) / (data.rho_pzr_l(t) - data.rho_pzr_v(t));
    a12 = -c5 / (data.rho_pzr_l(t) - data.rho_pzr_v(t));
    A_CT(:,:,t) = [a11 a12;
                     0   0  ];
    b11 = a12;     
    B_CT(:,:,t) = [b11;
                    0 ];
    sys_CT = ss(A_CT(:,:,t), B_CT(:,:,t), C_CT, 0);
    
    % discretize system and extract matrices
    sys_DT = c2d(sys_CT, h);
    [A_DT(:,:,t), B_DT(:,:,t), C_DT, D_DT] = ssdata(sys_DT);   
    
    % first state is the pzr level measurement, second state is the leak
    % magnitude
    x(:,t) = [data.lt459_(t);
              data.thwbreakf16(t)];

    % measurement is just the pzr level
    z(t) = C_DT*x(:,t);
    
    % call the particle filter
    if t == 1
        % input is zero at zeroth timestep
        u = 0;
        [x_est(:,t), x_P(:,1,:), P_w, x_P_pre] = SplitPF(A_DT, B_DT, C_DT, u, t, z(t), x(:,t), N, x_P, PZR_intl_cov, PZR_proc_cov, PZR_msmt_cov,...
                                        M_intl_cov, M_proc_local_cov, M_proc_global_range, local_srch_ratio);
    else
        % calculate input of previous timestep
        u = m_eff_dot(t-1) + m_net_dot(t-1);
        [x_est(:,t), x_P(:,1,:), P_w, x_P_pre(:,1,:)] = SplitPF(A_DT(:,:,t-1), B_DT(:,:,t-1), C_DT, u, t, z(t), x(:,t), N, x_P, PZR_intl_cov, PZR_proc_cov, PZR_msmt_cov,...
                                        M_intl_cov, M_proc_local_cov, M_proc_global_range, local_srch_ratio, x_est(1,t-1));
    end
    
%     windowSize = 5;
%     a = 1;
%     b = (1/windowSize)*ones(1,windowSize);
%     if t >= windowSize
%         x_est(2,t) = mean(x_est(2, t-windowSize+1:t));
%     end
    
    Pw_out = [Pw_out reshape(P_w, [N,1])];
    u_out = [u_out u];
    x_P_out = [x_P_out x_P];
    x_P_pre_out = [x_P_pre_out x_P_pre];
end
toc
time_per_iteration = toc/(T/2)

% avg_b4_loca = mean(x_est(2,1:58))
% 
% figure(1)
% plot(1:T, x(1,:), 1:T, x_est(1,:))
% xlabel('Timestep (0.5s)')
% ylabel('Pressurizer Level (%)')
% legend('Actual', 'Estimate', 'Location', 'southwest')

figure(2)
plot(1:T, x(2,:), 1:T, x_est(2,:))
xlabel('Timestep (0.5s)')
ylabel('Leak Magnitude (gpm)')
legend('Actual', 'Estimate', 'Location', 'northwest')


M_P_pre = x_P_pre_out(2,:,:);
M_P_pre = reshape(M_P_pre, [T, N]);
M_P_pre = M_P_pre';

L_P_pre = x_P_pre_out(1,:,:);
L_P_pre = reshape(L_P_pre, [T, N]);
L_P_pre = L_P_pre';


M_P = x_P_out(2,:,:);
M_P = reshape(M_P, [T, N]);
M_P = M_P';

L_P = x_P_out(1,:,:);
L_P = reshape(L_P, [T, N]);
L_P = L_P';

% for t = 1:T
%     figure(43
%     scatter(M_P_pre(:,t), L_P_pre(:,t))
%     title(num2str(t))
%     pause
% end

% for t = 1:T
%     figure(4)
%     scatter(M_P(:,t), L_P(:,t), 'filled')
%     title(num2str(t))
%     pause
% end

% for t = 1:T
%     figure(5)
%     stem3(M_P(:,t), L_P(:,t), ones(1,N)/N)
%     title(num2str(t))
%     pause
% end
% % 
% for t = 1:T
%     figure(6)
%     stem3(M_P_pre(:,t), L_P_pre(:,t), Pw_out(:,t))
%     title(num2str(t))
%     pause
% end
% 
% tt = 71;
% figure(7)
% stem3(M_P_pre(:,tt), L_P_pre(:,tt), Pw_out(:,tt))
% title(num2str(tt))
% % 
% tt = 68;
% figure(8)
% stem3(M_P(:,tt), L_P(:,tt), ones(1,N)/N)
% title(num2str(tt))
% % % 
% tt = 71;
% figure(9)
% histogram(M_P_pre(:,tt))
% title(num2str(tt))
% 
% tt = 20;
% figure(10)
% histogram(L_P_pre(:,tt))
% title(num2str(tt))

M_rms_error = rms(x(2,:)-x_est(2,:));
rms_after_LOCA = rms(x(2,59:T)-x_est(2,59:T));
[M_error_inf_norm, timestep_at_max] = max(abs(x(2,:)-x_est(2,:)));

for t = 59:T
   if x_est(2,t) >= 0.63*x(2,t)
       time_constant = (t-59)/2;
       break
   end
end

% x_est(:,832) = [];
% x_est(:,833) = [];
% x_est(:,920) = [];
% x(:,832) = [];
% x(:,833) = [];
% x(:,920) = [];
% rms_after_wo_spikes = rms(x(2,720:end)-x_est(2,720:end));

% for t = T:-1:720
%    if abs(x_est(2,t)-x(2,t)) >= 350
%        x_est(:,t) = [];
%        x(:,t) = [];
%    end
% end
% 
% rms_after_wo_spikes = rms(x(2,720:end)-x_est(2,720:end));



% for t = 720:T
%     if x_est(2,t) >= 0.1*x(2,t)
%         rise_time_t1 = t;
%         break
%     end
% end
% 
% for t = 720:T
%     if x_est(2,t) >= 0.9*x(2,t)
%         rise_time_t2 = t;
%         break
%     end
% end
% 
% rise_time = (rise_time_t2 - rise_time_t1)/2;
