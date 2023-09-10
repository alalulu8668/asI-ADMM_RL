% Jacobian ADMM: x is optimized in parallel
%
function [x,acc] = JADMM_SC(H,t,It_num,rho,T,V,A,L,v,x_o)
lam = zeros(V*L,v); acc = zeros(1,It_num);
x = cell(1,T);
for i=1:1:T
    x{i} = zeros(L,v);
end
for it=1:1:It_num
   
    x_i_t = cell(1,T);
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    for i=1:1:T
        HTA_i = H{i}'*t{i} + A{i}'*lam;
        for j=1:1:T
            if j~=i
                HTA_i = HTA_i - rho*A{i}'*A{j}*x{j};
            end
        end
        x_i_t{i} = inv(H{i}'*H{i} + rho*A{i}'*A{i})*HTA_i;
    end
    
    
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    for i=1:1:T
        x{i} = x_i_t{i};
        lam = lam - rho*A{i}*x{i};
    end
 
    %%%%%%%%%%%%%
    
    for i=1:1:T
        acc(it) = acc(it) + norm(x{i}-x_o)^2/L/T;
    end
 
end

end