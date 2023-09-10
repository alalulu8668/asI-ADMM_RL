%% generate env- wateFallGridWorld-stoc
clear all, close all
% clc

% ---- system parameter
env = rlPredefinedEnv('WaterFallGridWorld-Stochastic');
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

syspar.numObs = size(obsInfo.Elements,1);
syspar.dimObs = obsInfo.Dimension(1);

syspar.numAct = size(actInfo.Elements,1);
syspar.dimAct = actInfo.Dimension(1);


% ----- episode sets
numEpisodes = 2000;
syspar.numBatch = 1;
numMCRun = 5; % monte-carlo run
maxSteps = 50;
alpha = 1;    % discount factor


% agents
numAgent = 10;
agents = cell(1,numAgent);
syspar.mapping = 'table';
phi_s = @phi_s;


% connected nw
eta = 0.8;  
V = round(numAgent*(numAgent-1)/2*eta);
[~,B,G] = Connected_Graph(numAgent,V,1);
% plot(G);
D = degree(G);
comc = zeros(numMCRun,numEpisodes);

% algorithm 
alg{1}.gamma = 0.01;    % 1.GD-based
alg{2}.rho_x = 1; alg{2}.rho_z = 3; alg{2}.tau = 6; alg{2}.gamma = 1;      % 2.IADMM
alg{3}.gamma = 0.02; % 3.incremental GD-based
alg{4}.gamma = 1; alg{4}.eps_deg = 1; W = local_degree1(B,numAgent,alg{4}.eps_deg);     % 4. DGD
alg{5}.eta = 0.9; alg{5}.rho = 3; alg{5}.tau = 6; alg{5}.gamma = 1; %5: Stochastic-ADMM
alg{6}.eta = 0.9; alg{6}.rho = 3; alg{6}.tau = 6; alg{6}.gamma = 0.7; %6: adaptive ADAM
alg{7}.eta = 0.9; alg{7}.rho = 3; alg{7}.tau = 6; alg{7}.gamma = 1; %7: ADAM, global mu

alg_selec = [3,5]; % 1: gd; 2:ADMM  3:incremental_GD 4:DGD 5: Stochastic-ADMM
mode = 'sim';
% mode = 'plot';
filepath = sprintf('data/%dagent/ppo',numAgent);
if ~exist(filepath,'dir')
    mkdir(filepath)
end

global episode
episode = [];

% ci= 0;
% changes_para = [0.1,0.3,0.4,0.5,0.7,0.9];
% for para = changes_para
%     ci=ci+1;
%     alg{5}.eta  = para;
%%
his =   1;
if his == 1 % use 0 if first run
% ini_state = [];
% save('hist_init_state');
%     load(sprintf('seed%d',numAgent));
    load('seed10');
end
for alg_n = alg_selec
    
for rr = 1:numMCRun
 
    % ------ initial for each MC run
    numObs = syspar.numObs;
    dimObs = syspar.dimObs;
    dimAct = syspar.dimAct;
    numBatch = syspar.numBatch;
    numAct = syspar.numAct;
    i_ag = 1;
    gd = 0;
    z = zeros(numAct, numObs);
    mu = z;
    comc_tmp = 0;
    for i=1:numAgent   
    agents{i}.theta = zeros(numAct, numObs);
    agents{i}.lambda = zeros(numAct, numObs);
    agents{i}.mu = zeros(numAct,numObs);
    end
    
    % %%%%%%%%%%%%%%%%% MAIN CODE %%%%%%%%%%%%%%%%%%%%%%
    for i =1:numEpisodes
        % --- current agent   
        if i_ag > numAgent || i == 1
            i_ag = 1;
        end

        % use history initial
        if  his == 0
            s{rr,i} = rng;
            observation = reset(env);

        else
            if i<=size(s,2)
                id = i;
            else
                id = mod(i,size(s,2))+1;
            end

            rng(s{rr,id});
            observation = reset(env);
        end
        episode{i} = [];
        isDone = false(1);
        t = 0;

        total_return = 0;

        theta = agents{i_ag}.theta;
        while isDone == false(1) 

            pi_s = softmax(theta*phi_s(observation,syspar));
            action = randsample(numAct, 1, true, pi_s);
            [observation_next,reward,isDone] = step(env, action);
            record = [observation, action, reward, observation_next];
            episode{i} = [episode{i};record];
            observation = observation_next;
            total_return = total_return + reward;
            t=t+1;
            if isDone == true(1) || t > maxSteps
                break
            end
        end


        % --- stats update
        stats{rr}.steps(i) = t;
        stats{rr}.reward(i) = total_return;

        if mod(i,numBatch) == 0

            % --- gradient
            switch alg_n
                % gradient based
            case 1                
                [~,agents{i_ag}.gd] = GPOMDP(theta, ...
                    phi_s, syspar, alg{alg_n}, i, alpha);         
                gd = gd + agents{i_ag}.gd;
                    for ia = 1:numAgent
                        agents{ia}.theta = theta + alg{alg_n}.gamma*gd;
                    end                


                %  ADMM based
            case 2
                [theta_new, z_update,lambda] = ADMM_rl(numAgent,...
                     agents{i_ag}, phi_s, syspar, alg{alg_n}, i, alpha, z);
                agents{i_ag}.theta = theta_new;
                agents{i_ag}.lambda = lambda;
                z = z_update;

            case 3 % IGD
                [theta_new,agents{i_ag}.gd] = GPOMDP(theta, ...
                    phi_s, syspar, alg{alg_n}, i, alpha);   
                gd = gd + agents{i_ag}.gd;
                if i_ag == numAgent
                    for ii = 1:numAgent
                    agents{ii}.theta = agents{ii}.theta + alg{alg_n}.gamma*gd;
                    end
                    gd=0;
                end

            case 4
                theta_new = DGD_rl(numAgent, agents, i_ag,i, numBatch,W, gamma_dgd, phi_s,alpha, dimObs,dimAct, numObs, numAct);
                agents{i_ag}.theta = theta_new;
                % send to neighbours
                for nn = 1:length(B{i_ag})
                    neighbour = B{i_ag}(nn);
                    agents{neighbour}.theta = theta_new;
                end

            case 5
                [theta_new, z_update,lambda,mu] = ADMM_adamI(numAgent,...
                agents{i_ag}, phi_s, syspar, alg{alg_n}, i, alpha, z);
                agents{i_ag}.theta = theta_new;
                agents{i_ag}.lambda = lambda;
                agents{i_ag}.mu = mu;
                z = z_update;

            case 6 % adam 

                [theta_new, z_update,lambda,mu] = ADMM_adamI2(numAgent,...
                agents{i_ag}, phi_s, syspar, alg{alg_n}, i, alpha, z);
                agents{i_ag}.theta = theta_new;
                agents{i_ag}.lambda = lambda;
                agents{i_ag}.mu = mu;
                z = z_update;
            case 7 % ADAM with global mu
                [theta_new, z_update,lambda,mu_update] = ADMM_adamI3(numAgent,...
                agents{i_ag}, phi_s, syspar, alg{alg_n}, i, alpha, z, mu);
                agents{i_ag}.theta = theta_new;
                agents{i_ag}.lambda = lambda;
                mu = mu_update;
                z = z_update;


            comc_tmp = D(i_ag) + comc_tmp;
            comc(rr,i) = comc_tmp;

            end
            stats{rr}.theta{i} = agents{i_ag}.theta;
            i_ag = i_ag +1;
        end

    %     gamma_dgd = gamma_dgd*0.999;
    end

end

if his==0
    save(sprintf('seed%d',numAgent),'s');
end
save(sprintf(strcat(filepath,'/alg%d'),alg_n),'stats');
save(strcat(filepath,'/comc'));
save(strcat(filepath,'/RL_ADMM_tmp.mat'));
end
%% data process
reward = [];
steps = [];
err = [];
mode = 'sim';
color = [
         0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];

if strcmp(mode,'sim')
    alg_selec = alg_selec;
else 
    alg_selec = 2:1:4;
end

for alg_n = alg_selec
    load(sprintf(strcat(filepath,'/alg%d'),alg_n),'stats');
%     reward{alg_n}.total = zeros(1,numEpisodes);
    numEpi = size(stats{1}.reward,2);
    for rr = 1:numMCRun
    avr=0; % average reward
    av_steps = 0;
    theta = reshape(cell2mat(stats{rr}.theta),[numAct,numObs,numEpi]);
        for ia = 1:numAgent
            agents{ia}.steps = stats{rr}.steps(ia:numAgent:numEpi-numAgent+ia);
            agents{ia}.reward = stats{rr}.reward(ia:numAgent:numEpi-numAgent+ia);
            avr = avr + agents{ia}.reward;
            av_steps = av_steps + agents{ia}.steps;
            agents{ia}.theta = theta(:,:,ia:numAgent:numEpi-numAgent+ia);
        end
        reward{alg_n}.distr(rr,:) = movmean(avr/numAgent,40);
        steps{alg_n}.distr(rr,:) = av_steps/numAgent;
        reward{alg_n}.all(rr,:) = movmean(stats{rr}.reward,100);
        
        % calculate consensus error 
        for i_itr = 1:numEpi/numAgent
            tmp = 0;
            for ia = 1:numAgent
                for ja = 1:numAgent
                    tmp = tmp+ norm(agents{ia}.theta(:,:,i_itr) - agents{ja}.theta(:,:,i_itr));
                end
            end
            tmp = tmp / (numAgent^2);
            err{alg_n}.con_err(rr,i_itr) = tmp;
        end
    end
        reward{alg_n}.avr = mean(reward{alg_n}.distr);
        err{alg_n}.con_err = mean(err{alg_n}.con_err)/(64*4);
        save(sprintf(strcat(filepath,'/alg%d'),alg_n),'reward','err','-append');
end



% ###################### PLOT ###################### 
%%
% figure(2)
for alg_n = alg_selec
    X = size(reward{alg_n}.distr,2);
    h(alg_n)= stdshade(reward{alg_n}.distr(:,1:X), 0.15, color(alg_n,:),1:1:X); hold on
end

leg_s = {'SGD','IADMM','IGD','DGD','FAST ADMM','FAST adp ADMM','aaa'};
grid on
xlabel('Iteration'), ylabel('Globally Average Reward');
legend(h(alg_selec),leg_s(alg_selec));hold on

% end
% savefig(figure(2),sprintf('%dFADMM.fig',alg{5}.eta*10))

% 
figure(4)
% load(strcat(filepath,'/comc'));
for alg = alg_selec
    X2 = (numAgent-1)*size(reward{alg}.distr,2);
    h(alg)=stdshade(reward{alg}.distr,0.15,color(alg-1,:),1:(numAgent-1):X2); hold on    
end
X4 = round(sum(reshape(comc(1,:),numAgent, size(comc,2)/(numAgent)),1)./35);
h(4)=stdshade(reward{4}.distr(:,1:X),0.15, color(4-1,:), X4(1:X)); hold on    
grid on
xlabel('Communication cost'), ylabel('Episode Reward');
legend([h(2),h(3),h(4)],{'ADMM-based','Incremental Gradient Descent','Distributed Gradient Descent'});

% figure(5)
% plot(err{alg}.con_err);
% xlabel('Iteration'), ylabel('Consensus Error');
