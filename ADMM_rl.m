function [theta_new, z_new, lambda_new] = ADMM_rl(numAgent,...
    agents,phi_s, syspar, alg,idx,alpha,z_old)
global episode
theta_old = agents.theta;
lambda_old = agents.lambda;

dimObs = syspar.dimObs;
dimAct = syspar.dimAct;
numAct = syspar.numAct;
numBatch = syspar.numBatch;

rho_x = alg.rho_x;
tau = alg.tau;
gamma = alg.gamma;
rho = alg.rho_z;

gd = 0;
for i = idx-numBatch+1:idx
    gd_tmp = 0;
    total_return = 0;
    for t = 1:size(episode{i},1)
        state = episode{i}(t,1:dimObs);
        action = episode{i}(t,dimObs+1);
        reward = episode{i}(t,dimObs+dimAct+1);
        pi_s = softmax(theta_old * phi_s(state,syspar));
        pi_a_s = pi_s(action);
        gd_tmp = gd_tmp + 1/pi_a_s * (pi_a_s*(ind2vec(action,numAct) - pi_s))*phi_s(state,syspar)';
        total_return = total_return + alpha^(t-1)*reward;
    end
        gd = gd + gd_tmp * total_return;
end
gd_batch = 1/numBatch * gd;
theta_new = 1/(rho+tau)*(gd_batch + lambda_old + tau*theta_old + rho*z_old);
% theta_new = 1/(20+tau)*(gd_batch + lambda_old + tau*theta_old + 20*z_old);

% lambda_new = lambda_old + gamma*rho*(-theta_new + z_old);
lambda_new = lambda_old + gamma*rho*(-theta_new + z_old);

z_new = z_old + 1/numAgent*((theta_new - 1/rho * lambda_new) - (theta_old - 1/rho*lambda_old));
% theta_new(isnan(theta_new))=0;

