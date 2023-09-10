% 
%H,t,It_num,T,B,L,v,x_o,rho,alpha
function [x,acc,run_time] = DGD(H,t,It_num,T,B,L,v,x_o,alpha_fix)

global tl;
 

run_time = zeros(1,It_num);


eps_deg = 1;
W=local_degree1(B,T,eps_deg); 

%%
acc = zeros(1,It_num);
x = cell(1,T); x0 = cell(1,T); x_k = cell(1,T); x_k1 = cell(1,T);  
 
%initialization
for i=1:1:T
    x_k{i} = rand(L,v);
    x{i} = x_k{i};
    x0{i} = x{i};
    x_k1{i} = rand(L,v); 
end
 
for it = 1:1:It_num
    time_period = zeros(1,T); %record time slot for T agents
    for i= 1:1:T
        t1 = tic;
        %first order derivative
        fd = H{i}'*H{i}*x_k{i} - H{i}'*t{i};       
        nei_w = zeros(L,v);
        %sum for w*x and w_tilde*x
        for j = 1:1:T %every column j
            nei_w = nei_w + W(i,j)*x_k{j}; 
        end
        x_k1{i} = nei_w - alpha_fix * fd; 
        time_period(i) = toc(t1) + 10*rand(1)/tl;
    end    
    %%
    %global x
    for i=1:1:T
        t1 = tic;
        x_k{i} = x_k1{i};
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
%         acc(it) = acc(it) + norm(x{i}-x_o)^2/L/T;
        acc(it) = acc(it) + norm(x{i}-x_o)/norm(x0{i}-x_o)/T;
    end  
%     acc(it) = sqrt(acc(it));
end

end
