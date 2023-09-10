%
%
%

function [x,y,z,x_acc,acc,Run_time] = PWADMM3(H,t,It_num,beta,T,B,L,v,x_o,K,tau)
global tl;
 
% beta = 5;
run_time = zeros(K, It_num);
gamma = 1;

P = transitionP(B,T);

% tau = 0.8;
y = cell(1,T); z = cell(K,T); x = cell(1,K); x_loc = cell(T,K); x_acc = cell(1,K);
for i=1:1:T
    y{i} = zeros(L,v); 
    z{i} = zeros(L,v); 
    for j=1:1:K
       x_loc{i,j} = zeros(L,v); 
    end
end
acc = zeros(1,It_num);
for i=1:1:K
    x{i} = zeros(L,v);
end
for it=1:1:It_num
 
    %% PWADMM
    % y:x, z:lamda, x:z
    if it==1
        a = randperm(T);
        i_k = a(1:K);  % random choose K start point
        sum_a = zeros(L,v);
        for i=1:1:K
            t1 = tic;
            x_loc{i_k(i),i} = x{i};
            
%             for j=1:1:K
%                sum_a = sum_a + x_loc{i_k(i),j}/K; 
%             end
%             sum_a = z_t;
            y_t = inv(H{i_k(i)}'*H{i_k(i)} + beta*eye(L) + tau*eye(L))*(H{i_k(i)}'*t{i_k(i)} +...
                beta*sum_a + z{i_k(i)} + tau*y{i_k(i)});
            z_t = z{i_k(i)} + gamma*beta*(sum_a - y_t);
            x_t = sum_a + (y_t - z_t/beta)/T - (y{i_k(i)} - z{i_k(i)}/beta)/T;
            y{i_k(i)} = y_t;
            z{i_k(i)} = z_t;
            x_loc{i_k(i),i} = x_t;
            sum_a = x_t;
            run_time(i,it) = toc(t1) + 10*rand(1)/tl;
        end
    else
        for i=1:1:K
            t1 = tic;       
            i_k(i) = nei_choose(i_k(i),P,B);
            x_loc{i_k(i),i} = x{i};
%             sum_a = zeros(L,v);
%             for j=1:1:K
%                sum_a = sum_a + x_loc{i_k(i),j}/K; 
%             end
            
            y_t = inv(H{i_k(i)}'*H{i_k(i)} + beta*eye(L) + tau*eye(L))*(H{i_k(i)}'*t{i_k(i)}...
                + beta*sum_a + z{i_k(i)} + tau*y{i_k(i)});
            z_t = z{i_k(i)} + gamma*beta*(sum_a - y_t);
            x_t = sum_a + (y_t - z_t/beta)/T - (y{i_k(i)} - z{i_k(i)}/beta)/T;
            y{i_k(i)} = y_t;
            z{i_k(i)} = z_t;
            x_loc{i_k(i),i} = x_t;
            sum_a = x_t;
            run_time(i,it) = run_time(i,it-1) + toc(t1) + 10*rand(1)/tl;
 
        end
    end
    %%%%%%%%%%%%%
    for i=1:1:T
        acc(it) = acc(it) + norm(y{i}-x_o)/norm(x_o)/T;
    end

 
end

 Run_time = zeros(1,It_num);
 for i=1:1:It_num 
     temp = [];
     for j=1:1:K
        temp = [temp run_time(j,i)]; 
     end
     Run_time(i) = max(temp);    
 end 

end