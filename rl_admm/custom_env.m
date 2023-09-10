%% generate env- wateFallGridWorld-stoc
clear all, close all
% clc

% ---- system parameter
env = rlPredefinedEnv('WaterFallGridWorld-Stochastic');
env2 = rlPredefinedEnv('WaterFallGridWorld-Stochastic');
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

syspar.numObs = size(obsInfo.Elements,1);
syspar.dimObs = obsInfo.Dimension(1);

syspar.numAct = size(actInfo.Elements,1);
syspar.dimAct = actInfo.Dimension(1);

syspar.numEpisodes = 4000;
syspar.numMCRun = 5;    % monte-carlo run
syspar.maxSteps = 50;
syspar.discountFactor = 1;% discount factor
syspar.numBatch = 1;

% agents
numAgent = 10;
agents = cell(1,numAgent);
syspar.mapping = 'table';
phi_s = @phi_s;

% connected nw
eta = 0.5;  
V = round(numAgent*(numAgent-1)/2*eta);
% [~,B,G] = Connected_Graph(numAgent,V,1);
% plot(G);
% [A,B,C,G] = Graph_with_Hamilton(numAgent,V,1);
% save('graph20','A','B','C','G');
load(sprintf('graph%d.mat',numAgent));
D = degree(G);

% algorithm
% 1.GD-based
alg{1}.gamma = 0.02;    
% 2.IADMM
alg{2}.rho_x = 1; alg{2}.rho_z = 1; alg{2}.tau = 6; alg{2}.gamma = 1;
alg{3}.gamma = 0.01; % 3.IGD-based
% 4. DGD
alg{4}.gamma = 0.018; alg{4}.eps_deg = 1; W = local_degree1(C,numAgent,alg{4}.eps_deg);
alg{5}.eta = 0.7; alg{5}.rho = 1; alg{5}.tau = 6; alg{5}.gamma = 1; %5: local ADAM-ADMM
alg{6}.eta = 0.7; alg{6}.rho = 1; alg{6}.tau = 6; alg{6}.gamma = 1; %6: adaptive ADAM
alg{7}.eta = 0.9; alg{7}.rho = 1; alg{7}.tau = 6; alg{7}.gamma = 1; %7: ADAM, global mu
alg{8}.eta = 0.9; alg{8}.rho = 1; alg{8}.tau = 6; alg{8}.gamma = 1; %8: adaptive 

%--- 1: gd; 2:ADMM  3:incremental_GD 4:DGD 5: Stochastic-ADMM
alg_selec = [8]; 
fig_alg = [1,3,4,7,8]; % Data process only


% filepath = sprintf('data/%dagent/hetenv_init',numAgent);
filepath = sprintf('data/%dagent/ppo',numAgent);
if ~exist(filepath,'dir')
    mkdir(filepath)
end
% save(sprintf(strcat(filepath,'/para%d'),numAgent),'alg','syspar');

global episode
episode = [];

% ----- System setup,Don't touch!!!-------
numEpisodes = syspar.numEpisodes;
numMCRun = syspar.numMCRun; 
maxSteps = syspar.maxSteps;
alpha = syspar.discountFactor;   
numObs = syspar.numObs;
dimObs = syspar.dimObs;
dimAct = syspar.dimAct;
numBatch = syspar.numBatch;
numAct = syspar.numAct;
%-----------------------

%%
his =   1;
if his == 1 % use 0 if first run
% ini_state = [];
% save('hist_init_state');
%     load(sprintf('seed%d',numAgent));
    load('seed50');
end
 
for alg_n = alg_selec
    
for rr = 1:numMCRun
 
    % ------ initial for each MC run

    i_ag = 1;
    ii = 1;
    gd = 0;
    z = zeros(numAct, numObs);
    mu = z;
    comc_tmp = 0;
    for i=1:numAgent   
    agents{i}.theta = zeros(numAct, numObs);
    agents{i}.lambda = zeros(numAct, numObs);
    agents{i}.mu = zeros(numAct,numObs);
    % env for different agent
    agents{i}.gw = env;
        if i==numAgent
            agents{i}.gw = env2;
            agents{i}.gw.Model.R(agents{i}.gw.Model.R == 10) = 40;
        end
    end
% -------- main code

for i =1:numEpisodes
    % --- current agent   
    if i_ag > numAgent || i == 1
        i_ag = 1;
    end
    env = agents{i_ag}.gw;
    
    % ----- use history initial or NEW
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
%         observation = reset(env);
        observation = MyReset(env,i_ag);
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

        
        switch alg_n
        case 1 % --- Gradient               
            [~,agents{i_ag}.gd] = GPOMDP(theta, ...
                phi_s, syspar, alg{alg_n}, i, alpha);         
            gd = agents{i_ag}.gd;

            for ia = 1:numAgent
                agents{ia}.theta = agents{ia}.theta + alg{alg_n}.gamma*gd;
            end
            
        case 2 %  IADMM based
            [theta_new, z_update,lambda] = ADMM_rl(numAgent,...
                 agents{i_ag}, phi_s, syspar, alg{alg_n}, i, alpha, z);
            agents{i_ag}.theta = theta_new;
            agents{i_ag}.lambda = lambda;
            z = z_update;
        
        case 3  % IGD
            [theta_new,agents{i_ag}.gd] = GPOMDP(theta, ...
                phi_s, syspar, alg{alg_n}, i, alpha);   
            gd = gd + agents{i_ag}.gd;
            if ii == numAgent
                ii = 0;
                for ia = 1:numAgent
                agents{ia}.theta = agents{ia}.theta + 1/numAgent*alg{alg_n}.gamma*gd;
                end
%                 gd=0;
            end
            ii = ii +1;
            
        case 4 % DGD
            theta_new = DGD_rl(numAgent, agents, i_ag,i,W,...
                alg{alg_n}, phi_s,alpha,syspar);
            agents{i_ag}.theta = theta_new;
            % send to neighbours
            for nn = 1:length(C{i_ag})
                neighbour = C{i_ag}(nn);
                agents{neighbour}.theta = theta_new;
            end
            
        case 5 % local ADAM
            [theta_new, z_update,lambda,mu] = ADMM_adamI(numAgent,...
            agents{i_ag}, phi_s, syspar, alg{alg_n}, i, alpha, z);
            agents{i_ag}.theta = theta_new;
            agents{i_ag}.lambda = lambda;
            agents{i_ag}.mu = mu;
            z = z_update;
            
        case 6 % Adaptive gamma ADAM
                      
            [theta_new, z_update,lambda,mu] = ADMM_adamI2ad(numAgent,...
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
            
        case 8 % ADAM with global mu
%             [theta_new, z_update,lambda,mu_update] = ADMM_adamI4(numAgent,...
%             agents{i_ag}, phi_s, syspar, alg{alg_n}, i, alpha, z, mu);
%             agents{i_ag}.theta = theta_new;
%             agents{i_ag}.lambda = lambda;
%             mu = mu_update;
%             z = z_update;
             [theta_new, z_update,lambda,mu] = ADMM_centralized(numAgent,...
            agents{i_ag}, phi_s, syspar, alg{alg_n}, i, alpha, z, mu);
            agents{i_ag}.theta = theta_new;
            agents{i_ag}.lambda = lambda;
            if ii == numAgent
                ii = 0;
                temp_z =0;
                for ia = 1:numAgent
                    temp_z = temp_z + agents{ia}.theta - 1/alg{8}.rho * agents{ia}.lambda;
                end
                z = temp_z/numAgent;
            end
            ii = ii +1;
            
        % ------------- end ---------  
        end
        stats{rr}.theta{i} = agents{i_ag}.theta;
        i_ag = B{i_ag};
%         i_ag = i_ag +1;
    end
    
%     gamma_dgd = gamma_dgd*0.999;
end

end

if his==0
    save(sprintf('seed%d',numAgent),'s');
end
save(sprintf(strcat(filepath,'/alg%d'),alg_n),'stats');
save(strcat(filepath,'/RL_ADMM_tmp.mat'));
end
%% data process
reward = [];
steps = [];
err = [];


alg_selec = fig_alg;


for alg_n = alg_selec
    load(sprintf(strcat(filepath,'/alg%d'),alg_n),'stats');
%     reward{alg_n}.total = zeros(1,numEpisodes);
    numEpi = size(stats{1}.reward,2);
    for rr = 1:numMCRun
    avr=0; % average reward
    av_steps = 0;
    theta = reshape(cell2mat(stats{rr}.theta),[numAct,numObs,numEpi]);
        for ia = 1:numAgent
            agents{ia}.steps = stats{rr}.steps(ia:numAgent:(numEpi-numAgent+ia));
            agents{ia}.reward = stats{rr}.reward(ia:numAgent:(numEpi-numAgent+ia));
            avr = avr + agents{ia}.reward;
            av_steps = av_steps + agents{ia}.steps;
            agents{ia}.theta = theta(:,:,ia:numAgent:numEpi-numAgent+ia);
            reward{alg_n}.each(rr,ia,:) = movmean(agents{ia}.reward,80);
        end
        reward{alg_n}.distr(rr,:) = movmean(avr/numAgent,40);
        steps{alg_n}.distr(rr,:) = av_steps/numAgent;
        reward{alg_n}.all(rr,:) = movmean(stats{rr}.reward,100);
        
        % calculate consensus error 
        for i_itr = 1:(numEpi/numAgent)
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
        err{alg_n}.con_err = mean(err{alg_n}.con_err)/(56*numAgent);
        save(sprintf(strcat(filepath,'/alg%d'),alg_n),'reward','err','-append');
end



%% ###################### PLOT ###################### 

% figure(2)
Degree = 0;
for i=1:1:numAgent
    Degree = Degree + length(C{i});
end
pixl0 = 1:(numEpisodes/numAgent);
pixl = 2.*(1:(numEpisodes/numAgent));
pix_centralized = 2*numAgent*(1:(numEpisodes/numAgent));
pixl_deg = Degree.*pixl/12;

figAx={pix_centralized,pixl0,pixl,pixl_deg,pixl,pixl,pixl};
leg_s = {'Centralized-SGD','prox. I-ADMM','IGD','DGD','FAST ADMM','locA-ADMM','asI-ADMM','Centralized-ADMM'};

color = [
    0.8500    0.3250    0.0980 % red
    0.4660    0.6740    0.1880 % green    
         0    0.4470    0.7410 % blue    
    0.9290    0.6940    0.1250 % yellow
    0.4940    0.1840    0.5560 % purple
    0.3010    0.7450    0.9330 % light blue
    0.6350    0.0780    0.1840 % dark red
    1           0       1
    0           1       1       % cyan
    0           0       0       % black
    ];
ci = 1;
% fig_alg=alg_selec;

%%%%%%%%%%%----- fig--------
fig(1)=figure(1);
for alg_n = fig_alg
    load(sprintf(strcat(filepath,'/alg%d'),alg_n),'reward','err');
    X = size(reward{alg_n}.distr,2);
    h(alg_n)= stdshade(reward{alg_n}.distr(:,1:X), 0.15, color(ci,:),1:1:X); hold on
    ci = ci+1;
end
grid on
xlabel('Iteration'), ylabel('Globally Average Reward');
% ylim([-16,-2]);
legend(h(fig_alg),leg_s(fig_alg),'Interpreter','latex');hold on

% ---- communication cost
ci = 1;
fig(2)=figure(2);
for alg_n = fig_alg
    hh(alg_n)= stdshade(reward{alg_n}.distr(:,1:X), 0.15, color(ci,:),figAx{alg_n}); hold on
    ci = ci+1;
end
grid on
xlabel('Communication cost'), ylabel('Globally Average Reward');
xlim([0,800]);ylim([-16,-2]);
legend(hh(fig_alg),leg_s(fig_alg),'Interpreter','latex');hold on

% ---- consensus error
fig(3) = figure(5);
plot(err{7}.con_err,'r-','LineWidth',1.5);hold on
% plot(err{3}.con_err,'b-','LineWidth',1.5);
grid on
xlabel('Iteration'), ylabel('Consensus Error');

% -------- each agent
fig(4) = figure(6);
lines = {'-',':','--','-.'};
ci = 1;
li = 1;
fig_n = 1;
for alg_n= [7,3] 
    for iagent = [1,5,9] 
        data = reward{alg_n}.each(:,iagent,:);
        h4(fig_n) = stdshade(data(:,1:X), 0.15, color(ci,:),figAx{2},lines{li});hold on    
        legs{fig_n} = strcat(leg_s{alg_n},',',sprintf('Agent %d',iagent));
        fig_n = fig_n+1;
        ci = ci+1;
    end
    
    li = li+1;
end
grid on
xlabel('Communication cost'), ylabel('Individual Average Reward');
legend(h4,legs,'Interpreter','latex');
savefig(fig,sprintf(strcat(filepath,'/Fig_agent%d.fig'),numAgent));





%% some real test


theta = agents{1}.theta;
maxSteps = 100;

for i = 1:50
    isDone = false(1);
    t = 0;
    total_return = 0;
    episode_t{i} = [];
    dist{i}=[];
    observation = env.reset;
while isDone == false(1) 

    pi_s = softmax(theta*phi_s(observation,syspar));
    action = randsample(numAct, 1, true, pi_s);
    [observation_next,reward,isDone] = step(env, action);
    record = [observation, action, reward, observation_next];
    episode_t{i} = [episode_t{i};record];
    observation = observation_next;
    total_return = total_return + reward;
    
    t=t+1;
    if isDone == true(1) || t > maxSteps
        break
    end
end
    reward_t(i) = total_return;
    dist{i} = get_dist(episode_t{i});
end
function dist = get_dist(episode)
epi = episode(:,1);
pos = zeros(size(epi,1),2);
terminal = [4,5];
% get distance
pos(:,2) = ceil(epi/8);
pos(:,1) = rem(epi,8);
pos(pos == 0) = 8;

dist = vecnorm((pos - terminal),2,2);
end
