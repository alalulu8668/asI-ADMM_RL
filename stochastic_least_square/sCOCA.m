%
%
%
function [x,acc,Pixl,run_time] = sCOCA(H_o,t_o,It_num1,T,B,L,v,...
    x_o,c,a,p,batch,s)
global tl;
 

run_time = zeros(1,It_num1);
pixl = zeros(1,It_num1);
x_hat = cell(1,T);
x = cell(1,T); 
acc = zeros(1,It_num1); 
lam = cell(1,T);
d_ii = zeros(1,T);
zeta = cell(1,T);
sample_size = size(H_o{1},1);
for i=1:1:T
   d_ii(i) = length(B{i});
   x{i} = zeros(L,v);
   lam{i} = zeros(L,v);
   x_hat{i} = zeros(L,v);
   zeta{i} = zeros(L,v);
end
h = zeros(1,T);
for it=1:1:It_num1   
    time_period = zeros(1,T); %record time slot for T agents
%%
    for i=1:1:T
        t1 = tic;
        samp_idx = randsample(s,sample_size,batch);
        H{i} = H_o{i}(samp_idx,:);
        t{i} = t_o{i}(samp_idx,:);
        sum = zeros(L,v);
        for j=1:1:length(B{i})
            sum = sum + x_hat{B{i}(j)};
        end
        x{i} = inv(H{i}'*H{i} + 2*c*d_ii(i)*eye(L) ) *( H{i}'*t{i} - lam{i} + c*d_ii(i)*x_hat{i} + c*sum);      
        zeta{i} = x_hat{i} - x{i}; 
        h(i) = norm(zeta{i}) - a*p^it; 
        time_period(i) = toc(t1);
    end
%% 
    for i=1:1:T   
        t1 = tic;
        if h(i)>=0       
           x_hat{i} = x{i}; 
           pixl(it) = pixl(it) + d_ii(i);
        end
        time_period(i) =  time_period(i) + toc(t1);
    end
%%    
    for i=1:1:T   
        t1 = tic;
        for j=1:1:length(B{i})            
            lam{i} = lam{i} + c*(x_hat{i} - x_hat{B{i}(j)});           
        end
        time_period(i) =  time_period(i) + toc(t1) + 10*rand(1)/tl;
    end
%%
    if it==1
        run_time(it) = max(time_period);
    else
        run_time(it) = run_time(it-1) + max(time_period);
    end
%accuracy
    for i=1:1:T
%         acc(it) = acc(it) + norm(x{i}-x_o)^2/L/T;
        acc(it) = acc(it) + norm(x{i}-x_o)/norm(x_o)/T;
    end  
%     acc(it) = sqrt(acc(it));
end

Pixl = zeros(1,It_num1);
for i=1:1:It_num1   
   Pixl(i) = pixl(1:i)*ones(i,1);   
end


end