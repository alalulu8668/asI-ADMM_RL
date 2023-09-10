function i_next = nei_choose(i_k1,P,B)
p = 0;
P_cum = zeros(1,length(B{i_k1})+1);
for i=1:1:length(B{i_k1})
   p = p + P(i_k1,B{i_k1}(i));
   P_cum(i) = p;
end
P_cum(end) = 1;
C = [B{i_k1} i_k1];
a = rand(1,1);
i_ind = find(P_cum<a);
i_next = C(length(i_ind)+1);

end