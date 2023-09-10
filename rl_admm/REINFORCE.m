function [theta] = REINFORCE(theta_old, phi_s, numBatch,idx, alpha, dimObs,dimAct, numObs, numAct,gamma)
global episode
gd_batch = 0;
for i = idx-numBatch+1:idx
    gd_tmp = 0;
%     gd = 0;
    
    epi_len = size(episode{i},1);
    for t = 1:epi_len
        total_return = 0;
        state = episode{i}(t,1:dimObs);
        action = episode{i}(t,dimObs+1);
%         reward = episode{i}(t:,dimObs+dimAct+1);
        pi_s = softmax(theta_old * phi_s(state,numObs));
        pi_a_s = pi_s(action);
        gd_tmp =  1/pi_a_s * (pi_a_s*(ind2vec(action,numAct) - pi_s))*phi_s(state,numObs)';
        for tt = t:epi_len
            reward = episode{i}(tt,dimObs+dimAct+1);
            total_return = total_return + alpha^(tt-t)*reward;
        end
%         total_return = sum(episode{i}(t:1:end,dimObs+dimAct+1));
        gd = gd_tmp * total_return;
        theta_old = theta_old + gamma*gd;
    end

end
theta = theta_old;
theta(isnan(theta))=0;
% gd_final = 1/numBatch*gd_batch;
% theta = theta_old + gamma*gd_final;