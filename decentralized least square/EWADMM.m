%  it's Jacobian proximal ADMM based; in each iteration,only one connection
%  active
%
function [x,y,z,acc_EWADMM] = EWADMM(H,t,It_num,beta,T,B,L,x_o)
x = cell(1,T); lam = cell(1,L);
for i=1:1:T
    x{i} = zeros(L,1); lam{i} = zeros(L,1);
end
acc_EWADMM = zeros(1,It_num);

for it=1:1:It_num
    %% EWADMM
    
    
    
    
    
    
    for i=1:1:T
        acc_EWADMM(it) = acc_EWADMM(it) + norm(x-x_o)^2/L/T;
    end
end
 
end 