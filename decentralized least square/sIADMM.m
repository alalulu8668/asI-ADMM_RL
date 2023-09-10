% it's ADAM based; x is updated by a random walk over the graph G; in each
% iteration, x is sent to the next adjacent node for updating as a token.
%
function [x,y,z,acc,run_time] = sIADMM(H,t,It_num,beta,T,B,L,v,x_o,tau,batch)
global tl;
 

% P = transitionP(B,T);
run_time = zeros(1,It_num);
y = cell(1,T); z = cell(1,T);
sample_size = size(H{1},1);
for i=1:1:T
    y{i} = zeros(L,v); z{i} = zeros(L,v);
end
acc = zeros(1,It_num);
x = zeros(L,v);
for it=1:1:It_num
     
    %% sIADMM
    if mod(it,T)== 1
        i_k1 = 1;
    else
        i_k1 = i_k1+1;
    end
        t1 = tic;
%         i_k1 = 1;
        samp_idx = randi(sample_size,1,batch);
        H_samp = H{i_k1}(samp_idx,:);
        t_samp = t{i_k1}(samp_idx,:);
        y_t = inv(H_samp'*H_samp + beta*eye(L))*(H_samp'*t_samp + beta*x + z{i_k1});
%         y_t = inv(beta*eye(L))*(-H{i_k1}'*H{i_k1}*y{i_k1} + H{i_k1}'*t{i_k1} + beta*x + z{i_k1});
        z_t = z{i_k1} + beta*(x - y_t);
        x = x + (y_t - z_t/beta)/T  - (y{i_k1} - z{i_k1}/beta)/T ;
        y{i_k1} = y_t; 
        z{i_k1} = z_t;
%         run_time(it) = toc(t1) + 10*rand(1)/tl;
%     else
%         t1 = tic;
% %         i_k1 = B{i_k1}(randi(length(B{i_k1}),1));
%         i_k1 = i_k1+1;
%         y_t = inv(H{i_k1}'*H{i_k1} + beta*eye(L))*(H{i_k1}'*t{i_k1} + beta*x + z{i_k1});
% %         y_t = inv(beta*eye(L))*(-H{i_k1}'*H{i_k1}*y{i_k1} + H{i_k1}'*t{i_k1} + beta*x + z{i_k1});
%         z_t = z{i_k1} + beta*(x - y_t);
%         x = x + (y_t - z_t/beta)/T - (y{i_k1} - z{i_k1}/beta)/T ;
%         y{i_k1} = y_t; 
%         z{i_k1} = z_t;  
        
        run_time(it) = toc(t1) + 10*rand(1)/tl;
%     end
    %%%%%%%%%%%%%
    for i=1:1:T
%         acc(it) = acc(it) + norm(y{i}-x_o)^2/L/T;
        acc(it) = acc(it) + norm(y{i}-x_o)/norm(x_o)/T;
    end
%     acc(it) = sqrt(acc(it));
%     acc(it) = norm(x-x_o)^2/L ;
end
 
end 