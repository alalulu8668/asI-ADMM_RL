function W_FDLA=local_degree1(B,T,eps_deg)
W_FDLA = zeros(T,T);
P = zeros(T,T);
for i=1:1:T
    for j=1:1:length(B{i})
        P(i,B{i}(j)) = 1;
    end
end
 
deg=sum(P,2);
for i=1:T
    for j=1:T
        if P(i,j)==1
            W_FDLA(i,j)=1/(max(deg(i),deg(j))+eps_deg);
        end;
    end;
end;
W_FDLA=diag(ones(T,1)-sum(W_FDLA,2))+W_FDLA;
end