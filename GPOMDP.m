function [theta,gd_batch] = GPOMDP(theta_old, ...
    phi_s, syspar, alg,idx,alpha)
global episode

dimObs = syspar.dimObs;
dimAct = syspar.dimAct;
numAct = syspar.numAct;
numBatch = syspar.numBatch;
gamma = alg.gamma;
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
        gd = gd + gd_tmp * reward;
    end
%         gd = gd + gd_tmp * total_return;
end
gd_batch = 1/numBatch * gd;
theta = theta_old + gamma*gd_batch;
theta(isnan(theta))=0;
% gd_final = 1/numBatch*gd_batch;
% theta = theta_old + gamma*gd_final;