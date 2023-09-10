% it's ADAM based; x is updated by a random walk over the graph G; in each
% iteration, x is sent to the next adjacent node for updating as a token.
%
function [x,acc,run_time] = sDADMM(H_o,t_o,It_num,T,B,L,v,x_o,rho,batch,s)
global tl;
 
% y = cell(1,T); z = cell(1,T);
sample_size = size(H_o{1},1);
alpha_t = cell(1,T);
c = rho;
% for i=1:1:T
%     y{i} = zeros(L,v); z{i} = zeros(L,v);
% end
run_time = zeros(1, It_num);

%initialize alpha
for i = 1:1:T
    alpha_t{i} = zeros(L, v);
end

%total loss
acc = zeros(1,It_num);
%xi initialization
for i=1:1:T
    x{i} = zeros(L,v);
end

%iter loop
for it = 1:1:It_num
    
    time_period = zeros(1,T); %record time slot for T agents
 
    x_i_t = cell(1,T);
    %each agent i
    for i = 1:1:T
        %gradient in agent i
        t1 = tic;
        card_i = length(B{i});
        neig_x_i = zeros(L, v);
        %calculate sum of neighbors
        for j = 1:1:card_i
            neig_x_i = neig_x_i + x{B{i}(j)};
        end
        %update x
        samp_idx = randsample(s,sample_size,batch);
        H{i} = H_o{i}(samp_idx,:);
        t{i} = t_o{i}(samp_idx,:);
        x_i_t{i} = inv(H{i}'*H{i} + 2*c*card_i*eye(L))*...
            (H{i}'*t{i} + c*card_i*x{i} + c*neig_x_i - alpha_t{i});
        time_period(i) = toc(t1);
    end
    
    for i=1:1:T
        x{i} = x_i_t{i};
    end
    
    for i=1:1:T
        t1 = tic;
        neig_x_i = zeros(L,v);
        for j = 1:1:length(B{i})
            neig_x_i = neig_x_i + x{B{i}(j)};
        end
        alpha_t{i} = alpha_t{i} + c*(length(B{i})*x{i} - neig_x_i);
        time_period(i) = time_period(i) + toc(t1) + 10*rand(1)/tl;
    end
    
    if it==1
        run_time(it) = max(time_period);
    else
        run_time(it) = run_time(it-1) + max(time_period);
    end
    
    for i=1:1:T
%         acc(it) = acc(it) + norm(x{i}-x_o)^2/L/T;
        acc(it) = acc(it) + norm(x{i}-x_o)/norm(x_o)/T;
    end
%     acc(it) = sqrt(acc(it));
end %end for
%%
%total loss


end