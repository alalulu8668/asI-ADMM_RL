% it's ADAM based; x is updated by a random walk over the graph G; in each
% iteration, x is sent to the next adjacent node for updating as a token.
%
function [x,acc,run_time] = EXTRA(H,t,It_num,T,B,L,v,x_o,alpha_fix)
global tl;
 
run_time = zeros(1,It_num);

eps_deg = 1;
W=local_degree1(B,T,eps_deg); 
W_tilde = (W + eye(T))/2;

%%
acc = zeros(1,It_num);
 
% [W, W_tilde] = WMatrix(n, B);
 
x = cell(1,T); x0 = cell(1,T); x_k = cell(1,T); x_k1 = cell(1,T); x_k2 = cell(1,T);
fd_derivative = cell(1,T);
fd_derivative_k1 = cell(1,T);
%initialization
for i=1:1:T
%     x{i} = zeros(L,v);
    x_k{i} = rand(L,v);
    x{i} = x_k{i};
    x0{i} = x{i};
    x_k1{i} = rand(L,v);
    x_k2{i} = rand(L,v);
    fd_derivative{i} = zeros(L,v);
    fd_derivative_k1{i} = zeros(L,v);
end

%% step1
%%x1 calculation

 

for i =1:1:T %every agent i
    nei = zeros(L,v);
    fd_derivative{i} = H{i}'*H{i}*x_k{i} - H{i}'*t{i};
    %sum for w*x
    for j = 1:1:T %every column j
        nei = nei + W(i,j)*x_k{j};
    end
    x_k1{i} = nei - alpha_fix*fd_derivative{i};
end

%% step2
for it = 1:1:It_num
    
    
    time_period = zeros(1,T); %record time slot for T agents
    
%     for i= 1:1:T
%         %first order derivative
%         fd_derivative{i} = 2*H{i}'*H{i}*x_k{i} - 2*H{i}'*t{i};
%         fd_derivative_k1{i} = 2*H{i}'*H{i}*x_k1{i} - 2*H{i}'*t{i};
%     end
 
    for i= 1:1:T
        t1 = tic;
        %first order derivative
        fd = H{i}'*H{i}*x_k{i} - H{i}'*t{i};
        fd_1 = H{i}'*H{i}*x_k1{i} - H{i}'*t{i};
        nei_w = zeros(L,v);
        nei_w_tilde = zeros(L,v);
        
        %sum for w*x and w_tilde*x
        for j = 1:1:T %every column j
            nei_w = nei_w + W(i,j)*x_k1{j};
            nei_w_tilde = nei_w_tilde + W_tilde(i,j)*x_k{j};
        end
        
        %step2 update
        x_k2{i} = x_k1{i} + nei_w - nei_w_tilde - alpha_fix*(fd_1 - fd);
        
%         %update x_k+1, x_k
%         x_k{i} = x_k1{i};
%         x_k1{i} = x_k2{i};
%         %x_k{i} = t;
        time_period(i) = toc(t1);
    end
    
    %%
    %global x
    for i=1:1:T
        t0 = tic;
        x_k{i} = x_k1{i};
        x_k1{i} = x_k2{i};
        x{i} = x_k2{i};
        time_period(i) = time_period(i) + toc(t1) + 10*rand(1)/tl;
    end   
    if it==1
        run_time(it) = max(time_period);
    else
        run_time(it) = run_time(it-1) + max(time_period);
    end
    %%
    %accuracy
    for i=1:1:T
%         acc(it) = acc(it) + norm(x{i}-x_o)^2/L/T;
        acc(it) = acc(it) + norm(x{i}-x_o)/norm(x0{i}-x_o)/T;
    end
%     acc(it) = sqrt(acc(it));

    %update x_k+1, x_k
    
    %x_k{i} = t;
    
end

end
