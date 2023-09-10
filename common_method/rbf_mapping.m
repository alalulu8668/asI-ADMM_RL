function [x_mean,x_std,weight,offset] = rbf_mapping(env,state,rbf)
% featurize
gamma = rbf.gamma;
n_component = rbf.n_component;
n_rbf = size(gamma,2);

[n_features,~ ] = size(state);

i=1;
for epi = 1:5000
    observation = reset(env);
    isDone = false(1);
while isDone == false(1) 
    record(i,:) = observation;
    action = randi([0,1],1)*20 - 10;
    [observation_next,reward,isDone] = step(env, action);
    
    record(i+1,:) = observation_next;
    i = i+1;
    observation = observation_next;
    if isDone == true(1) || i > 10000
            break
    end
end
end
x_mean = mean(record)';
x_std = std(record,0,1)'; 


weight = [];
offset = [];
for i = 1:n_rbf
    
    weight = [weight,sqrt(2*gamma(i))*randn(n_features,n_component)];
    offset = [offset,2*pi*rand(1,n_component)];
end

