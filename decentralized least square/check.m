% consensus optimization with multiple masters
%  
%

clc; clear; clear all;

T = 20; eta = 0.3;  
V = round(T*(T-1)/2*eta); 
L = 2; v = 1; It_num0 = 10000; It_num1 = 1000; 
N = 30*ones(1,T);
beta_w = 2; K = 2; % parameter for W-ADMM

H = cell(1,T); t = cell(1,T);
HH = []; tt = [];
for i=1:1:T
    H{i} = rand(N(i),L); t{i} = rand(N(i),v);
    HH = [HH; H{i}]; tt = [tt; t{i}];
end
 
%% optimal solution
x_o = inv(HH'*HH)*HH'*tt;

 
gamma = 1;

P = transitionP(B,T);

% tau = 0.8;
y = cell(1,T); z = cell(K,T); x = cell(1,K); x_loc = cell(T,K); x_acc = cell(1,K);N
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
    if it==1
        a = randperm(T);
        i_k = a(1:K);  % random choose K start point
        for i=1:1:K
         
            x_loc{i_k(i),i} = x{i};
            sum_a = zeros(L,v);
            for j=1:1:K
               sum_a = sum_a + x_loc{i_k(i),j}/K; 
            end
            y_t = inv(H{i_k(i)}'*H{i_k(i)} + beta*eye(L) + tau*eye(L))*(H{i_k(i)}'*t{i_k(i)} +...
                beta*sum_a + z{i_k(i)} + tau*y{i_k(i)});
            z_t = z{i_k(i)} + gamma*beta*(sum_a - y_t);
            x{i} = x{i} + (y_t - z_t/beta)/T*K - (y{i_k(i)} - z{i_k(i)}/beta)/T*K;
            y{i_k(i)} = y_t;
            z{i_k(i)} = z_t;
            x_loc{i_k(i),i} = x{i};
       
        end
    else
        for i=1:1:K
            
            i_k(i) = nei_choose(i_k(i),P,B);
            x_loc{i_k(i),i} = x{i};
            sum_a = zeros(L,v);
            for j=1:1:K
               sum_a = sum_a + x_loc{i_k(i),j}/K; 
            end
            y_t = inv(H{i_k(i)}'*H{i_k(i)} + beta*eye(L) + tau*eye(L))*(H{i_k(i)}'*t{i_k(i)}...
                + beta*sum_a + z{i_k(i)} + tau*y{i_k(i)});
            z_t = z{i_k(i)} + gamma*beta*(sum_a - y_t);
            x{i} = x{i} + (y_t - z_t/beta)/T*K - (y{i_k(i)} - z{i_k(i)}/beta)/T*K;
            y{i_k(i)} = y_t;
            z{i_k(i)} = z_t;
            x_loc{i_k(i),i} = x{i};
 
        end
    end
    %%%%%%%%%%%%%
    for i=1:1:T
        acc(it) = acc(it) + norm(y{i}-x_o)/norm(x_o)/T;
    end

 
end