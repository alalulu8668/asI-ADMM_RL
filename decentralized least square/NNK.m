% 
%H,t,It_num,T,B,L,v,x_o,rho,alpha
function [x,acc,run_time] = NNK(H,t,It_num,T,B,L,v,x_o,alpha,K)

global tl;
run_time = zeros(1,It_num);
 
eps_deg = 1;
W_t=local_degree1(B,T,eps_deg); 
W = (W_t + eye(T))/2;

% wdiag = diag(W);
% delta = min(wdiag);
% Delta = max(wdiag);
% Her = cell(1,T);
% for i=1:1:T
%    Her{i} = H{i}'*H{i}; 
% end


epsilon = 0.7;
%%
BB = cell(T,T);
for i=1:1:T
    BB{i,i} = (1 - W(i,i))*eye(L,L);
    for j=1:1:T
        BB{i,j} = W(i,j)*eye(L,L);
    end
end

acc = zeros(1,It_num);
 
x = cell(1,T); x_k = cell(1,T); 
 
%initialization
for i=1:1:T
    x_k{i} = rand(L,v);
    x{i} = x_k{i};
 
end

 
D = cell(1,T);
g = cell(1,T);
for it = 1:1:It_num
     time_period = zeros(1,T); %record time slot for T agents
    for i= 1:1:T
        t1 = tic;
        D{i} = alpha * H{i}'*H{i} + 2*(1-W(i,i))*eye(L);
        g{i} = (1-W(i,i))*x{i} + alpha * (H{i}'*H{i}*x{i} - H{i}'*t{i});
        for j=1:1:length(B{i})
            g{i} = g{i} - W(i,B{i}(j))*x{B{i}(j)};
        end
        d{i} = -inv(D{i})*g{i};
 
        for k=1:1:K-1
            Bd = zeros(L,v);
            for j=1:1:length(B{i})
                Bd = Bd + BB{i,B{i}(j)}*d{i};
            end
            d{i} = inv(D{i})*(Bd - g{i});
        end
         
        x_k{i} = x{i} + epsilon * d{i};
        time_period(i) = toc(t1) + 10*rand(1)/tl;
    end
    
    %%
    %global x
    for i=1:1:T   
        t1 = tic;
        x{i} = x_k{i};
        time_period(i) = time_period(i) + toc(t1);
    end   
    if it==1
        run_time(it) = max(time_period);
    else
        run_time(it) = run_time(it-1) + max(time_period);
    end
    %%
    %accuracy
    for i=1:1:T
        acc(it) = acc(it) + norm(x{i}-x_o)^2/L/T;
    end

    
end
% run_time = 1;
end
