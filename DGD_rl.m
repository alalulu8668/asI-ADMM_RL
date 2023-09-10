function theta_new = DGD_rl(numAgent,agents,i_ag,idx,W, alg,...
    phi_s,alpha, syspar)
global episode
theta = zeros(size(agents{i_ag}.theta));
gd = 0;

dimObs = syspar.dimObs;
dimAct = syspar.dimAct;
numAct = syspar.numAct;
numBatch = syspar.numBatch;
gamma = alg.gamma;
theta_old = agents{i_ag}.theta;
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
% use W matrix for neighbours, and do the update

% for j = 1:numAgent
%     theta = theta + W(i_ag,j)*agents{j}.theta;
% end
% theta_new = theta + gamma*gd_batch;
theta_new = theta_old + gamma*gd_batch;
% send to neighbours
