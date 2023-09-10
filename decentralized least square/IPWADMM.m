% intelligent PWADMM
%
%

function [x,y,z,acc,Run_time] = IPWADMM(H,t,It_num,beta,T,B,L,v,x_o,K,tau)
global tl;
 
run_time = zeros(K, It_num);
 
y = cell(1,T); z = cell(K,T); x = cell(1,K);
hit = zeros(1,T);
degree = zeros(1,T);
for i=1:1:T
    y{i} = zeros(L,v); 
    z{i} = zeros(L,v);
    B_t{i} = [B{i} i];
    degree(i) = length(B_t{i});
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
            t1 = tic;
            y_t = inv(H{i_k(i)}'*H{i_k(i)} + beta*eye(L) + tau*eye(L))*(H{i_k(i)}'*t{i_k(i)} + beta*x{i} + z{i_k(i)} + tau*y{i_k(i)});
%             y_t = inv(beta*eye(L) + tau*eye(L))*(-H{i_k(i)}'*H{i_k(i)}*y{i_k(i)} + H{i_k(i)}'*t{i_k(i)} + beta*x{i} + z{i_k(i)} + tau*y{i_k(i)});
            z_t = z{i_k(i)} + beta*(x{i} - y_t);
            x{i} = x{i} + (y_t - z_t/beta)/T*K - (y{i_k(i)} - z{i_k(i)}/beta)/T*K;
            y{i_k(i)} = y_t;
            z{i_k(i)} = z_t;
            hit(i_k(i)) = hit(i_k(i)) + 1;
            run_time(i,it) = toc(t1);
        end
    else
        for i=1:1:K
            t1 = tic;
%             hit_t = hit(B_t{i_k(i)})./degree(B_t{i_k(i)});
            hit_t = hit(B_t{i_k(i)});
            [~,ind] = find(hit_t==min(hit_t));
            b = randperm(length(ind));
            i_k(i) = B_t{i_k(i)}(ind(b(1)));
            y_t = inv(H{i_k(i)}'*H{i_k(i)} + beta*eye(L) + tau*eye(L))*(H{i_k(i)}'*t{i_k(i)} + beta*x{i} + z{i_k(i)} + tau*y{i_k(i)});
%             y_t = inv(beta*eye(L) + tau*eye(L))*(-H{i_k(i)}'*H{i_k(i)}*y{i_k(i)} + H{i_k(i)}'*t{i_k(i)} + beta*x{i} + z{i_k(i)} + tau*y{i_k(i)});
            z_t = z{i_k(i)} + beta*(x{i} - y_t);
            x{i} = x{i} + (y_t - z_t/beta)/T*K - (y{i_k(i)} - z{i_k(i)}/beta)/T*K;
            y{i_k(i)} = y_t;
            z{i_k(i)} = z_t;
            hit(i_k(i)) = hit(i_k(i)) + 1;
            run_time(i,it) = run_time(i,it-1) + toc(t1) + 10*rand(1)/tl;
 
        end
    end
    %%%%%%%%%%%%%
    for i=1:1:T
%         acc(it) = acc(it) + norm(y{i}-x_o)^2/L/T;
        acc(it) = acc(it) + norm(y{i}-x_o)/norm(x_o)/T;
    end
%     acc(it) = sqrt(acc(it));
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