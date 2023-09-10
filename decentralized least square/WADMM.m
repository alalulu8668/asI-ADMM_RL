% it's ADAM based; x is updated by a random walk over the graph G; in each
% iteration, x is sent to the next adjacent node for updating as a token.
%
function [x,y,z,acc,run_time] = WADMM(H,t,It_num,beta,T,B,L,v,x_o)
global tl;
 

P = transitionP(B,T);
run_time = zeros(1,It_num);
y = cell(1,T); z = cell(1,T);
for i=1:1:T
    y{i} = zeros(L,v); z{i} = zeros(L,v);
end
acc = zeros(1,It_num);
x = zeros(L,v);
for it=1:1:It_num
     
    %% WADMM
    if it==1
        t1 = tic;
        i_k1 = randi(T,1);
        y_t = inv(H{i_k1}'*H{i_k1} + beta*eye(L))*(H{i_k1}'*t{i_k1} + beta*x + z{i_k1});
%         y_t = inv(beta*eye(L))*(-H{i_k1}'*H{i_k1}*y{i_k1} + H{i_k1}'*t{i_k1} + beta*x + z{i_k1});
        z_t = z{i_k1} + beta*(x - y_t);
        x = x + (y_t - z_t/beta)/T  - (y{i_k1} - z{i_k1}/beta)/T ;
        y{i_k1} = y_t; 
        z{i_k1} = z_t;
        run_time(it) = toc(t1) + 10*rand(1)/tl;
    else
        t1 = tic;
%         i_k1 = B{i_k1}(randi(length(B{i_k1}),1));
        i_k1 = nei_choose(i_k1,P,B);
        y_t = inv(H{i_k1}'*H{i_k1} + beta*eye(L))*(H{i_k1}'*t{i_k1} + beta*x + z{i_k1});
%         y_t = inv(beta*eye(L))*(-H{i_k1}'*H{i_k1}*y{i_k1} + H{i_k1}'*t{i_k1} + beta*x + z{i_k1});
        z_t = z{i_k1} + beta*(x - y_t);
        x = x + (y_t - z_t/beta)/T - (y{i_k1} - z{i_k1}/beta)/T ;
        y{i_k1} = y_t; 
        z{i_k1} = z_t;      
        run_time(it) = run_time(it-1) + toc(t1) + 10*rand(1)/tl;
    end
    %%%%%%%%%%%%%
    for i=1:1:T
%         acc(it) = acc(it) + norm(y{i}-x_o)^2/L/T;
        acc(it) = acc(it) + norm(y{i}-x_o)/norm(x_o)/T;
    end
%     acc(it) = sqrt(acc(it));
%     acc(it) = norm(x-x_o)^2/L ;
end
 
end 