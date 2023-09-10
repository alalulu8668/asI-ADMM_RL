%   generate the directed graph G=(V,E);
%   
%
%
function [A,B,G] = Connected_Graph(T,V,L)
A = cell(1,T); B = cell(1,T);  %%%%%% A is connection matrix for ADMM; B is the neighbors of each node
for i = 1:1:T
    A{i} = []; B{i} = [];
end
s_temp = [];
t_temp = [];
Ind = 1:1:T;
%% generate pairs
for i=1:1:T
    for j=i:1:T-1
        s_temp = [s_temp i];
        t_temp = [t_temp j+1];
    end
end

flag = 0;
while flag==0   
    a = randperm(length(s_temp));
    s = s_temp(a(1:V));  % 
    t = t_temp(a(1:V));
    G = graph(s,t);
    %% check the connectivity
    bins = conncomp(G);   %% returns the connected components of graph G as bins. The bin numbers indicate which component each node in the graph belongs to.
    if length(find(bins==1))==T     %%%% Âú×ãÁ¬Í¨ÐÔ 
        for i=1:1:V
            for j=1:1:T
                if j==s(i)
                    A{j} = [A{j};eye(L)];
                elseif j==t(i)
                    A{j} = [A{j};-eye(L)];
                else
                    A{j} = [A{j};zeros(L,L)];
                end              
            end
        end    
        flag = 1;
    end   
end

% plot(G)

for i=1:1:T
    a = find(s==i);
    B{i} = [B{i} t(a)];
    b = find(t==i);
    B{i} = [B{i} s(b)];
end
 



 





