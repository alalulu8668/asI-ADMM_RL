function P = transitionP(B,T)
P = zeros(T,T);
for i=1:1:T
   for j=1:1:length(B{i})
        
       P(i,B{i}(j)) = 1/max(length(B{i})+1,length(B{B{i}(j)})+1);
   end
    P(i,i) = 1-sum(P(i,:));
end

end