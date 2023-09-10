%    
%
%      T -- number of agents;        V -- number of connections ;
%      L -- input dimension;         N -- number of training samples;
%      v -- output dimension;        K -- number of active random walk;
%    eta -- connection ratio;        
%
clc;clear;
close all;

% delete(gcp('nocreate'))
% parpool(20)

%% parameter

T = 100; eta = 0.3;  
V = round(T*(T-1)/2*eta); 
L = 10; v = 1; It_num0 = 10000; It_num1 = 1000; 
N = 100*ones(1,T);

batch = 10;
s = RandStream('mlfg6331_64');
% N = randi([10,50],1,T);
beta_w = 2;  % parameter for W-ADMM
beta_p = 2; K = 25; tau = 0;  % parameter for PW-ADMM
beta_ip = 2; K_i = T; tau_i = 0;  % parameter for IPW-ADMM
% beta_iadm = 3; tau_iadm = 0.5; % parameter for IADMM
beta_siadm = 3; tau_siadm = 1; gamma_siadm = 1;% parameter for sIADMM
beta_Fsiadm = 3; tau_Fsiadm = 1; gamma_Fsiadm = 1; eta_Fsiadm = 0.7;% parameter for FsIADMM
rho = 1;  % parameter for D-ADMM 
alpha0 = 0.01;  % parameter for EXTRA
alpha1 = 0.01;  % parameter for DGD
c = 1; a = 1; p = 0.85;  % parameter for COCA
alpha2 = 0.001; KK = 2;   % parameter for NNK

global tl; 
tl = 1e5; 
% load graph_T20ETA03;
Ite = 1;
acc_WADMM=cell(1,Ite); t_WADMM=cell(1,Ite); acc_PWADMM=cell(1,Ite); t_PWADMM=cell(1,Ite); acc_PWADMM2=cell(1,Ite); t_PWADMM2=cell(1,Ite); 
acc_IPWADMM=cell(1,Ite); t_IPWADMM=cell(1,Ite); 
acc_IADMM = cell(1,Ite); t_IADMM = cell(1,Ite);
acc_sIADMM = cell(1,Ite); t_sIADMM = cell(1,Ite);
acc_FsIADMM = cell(1,Ite); t_FsIADMM = cell(1,Ite);
acc_EXTRA=cell(1,Ite); t_EXTRA=cell(1,Ite); acc_DADMM=cell(1,Ite); t_DADMM=cell(1,Ite); acc_DGD=cell(1,Ite); t_DGD=cell(1,Ite);
acc_COCA=cell(1,Ite); pixl3=cell(1,Ite); t_COCA=cell(1,Ite);
acc_NNK=cell(1,Ite); t_NNK=cell(1,Ite);

for k=1:1:Ite
    
% [A,B,G] = Connected_Graph(T,V,L);
% plot(G);
load 'graph100.mat';
B = b{i}; C = c{i}; G = g{i};
% save('graph_T20ETA03','A','B','G');
%% data generate
 
% H = cell(1,T); t = cell(1,T); 
% HH = []; tt = []; x_o = rand(L,v);
% for i=1:1:T
%     H{i} = rand(N(i),L);  t{i} = H{i}*x_o + 0.1*randn(N(i),v);
%     HH = [HH; H{i}]; tt = [tt; t{i}];
%     
% end
 
%% generation 2
load('seed');
rng(ss);
x_o = randn(L,v);
H = cell(1,T); t = cell(1,T);
HH = []; tt = [];
for i=1:1:T
    H{i} = randn(N(i),L); 
    t{i} = H{i}*x_o + 0.1*randn(N(i),v);
    HH = [HH; H{i}]; tt = [tt; t{i}];
end
x_o = inv(HH'*HH)*HH'*tt;
% save('seed','ss')
%% optimal solution
% H = cell(1,T); t = cell(1,T);
% HH = []; tt = [];
% for i=1:1:T
%     H{i} = rand(N(i),L); t{i} = rand(N(i),v);
%     HH = [HH; H{i}]; tt = [tt; t{i}];
% end
%  
% %% optimal solution
% x_o = inv(HH'*HH)*HH'*tt;

%% updating process          
% [x_W,y_W,z_W,acc_WADMM{k},t_WADMM{k}] = WADMM(H,t,It_num0,beta_w,T,B,L,v,x_o);1
% [x_PW,y_PW,z_PW,x_acc,acc_PWADMM{k},t_PWADMM{k}] = PWADMM(H,t,It_num1,beta_p,T,B,L,v,x_o,K,tau);2  
% [x_PW2,y_PW2,z_PW2,x_acc2,acc_PWADMM2{k},t_PWADMM2{k}] = PWADMM2(H,t,It_num1,beta_p,T,B,L,v,x_o,K,tau);3  % utilize average token in update
% [x_IPW,y_IPW,z_IPW,acc_IPWADMM{k},t_IPWADMM{k}] = IPWADMM(H,t,It_num1,beta_ip,T,B,L,v,x_o,K_i,tau_i);4
% [x_IW,y_IW,z_IW,acc_IADMM{k},t_IADMM{k}] = IADMM(H,t,It_num0,beta_iadm,T,B,L,v,x_o,tau_iadm);5
[x_sIW,y_sIW,z_sIW,acc_sIADMM{k},t_sIADMM{k}] = sIADMM(H,t,It_num0,beta_siadm,T,B,L,v,x_o,tau_siadm,gamma_siadm,batch,s);6
[x_FsIW,y_FsIW,z_FsIW,acc_FsIADMM{k},t_FsIADMM{k}] = FsIADMM2(H,t,It_num0,beta_Fsiadm,T,B,L,v,x_o,tau_Fsiadm,gamma_Fsiadm,eta_Fsiadm,batch,s);7

% [x_EXTRA,acc_EXTRA{k},t_EXTRA{k}] = sEXTRA(H,t,It_num1,T,B,L,v,x_o,alpha0,batch,s);8
% [x_DADMM,acc_DADMM{k},t_DADMM{k}] = sDADMM(H,t,It_num1,T,B,L,v,x_o,rho,batch,s);9
% [x_DGD,acc_DGD{k},t_DGD{k}] = sDGD(H,t,It_num1,T,B,L,v,x_o,alpha1,batch,s);10
% [x_COCA,acc_COCA{k},pixl3{k},t_COCA{k}] = sCOCA(H,t,It_num1,T,B,L,v,x_o,c,a,p,batch,s);11
% [x_NNK,acc_NNK{k},t_NNK{k}] = NNK(H,t,It_num1,T,B,L,v,x_o,alpha2,KK);9
end 
save('main.mat');
 
% semilogy(acc_NNK{1})
 

%% %%%%%%%%%%%%% iteration -- accuracy
pixl0 = 1:1:It_num0;
pixl = 1:1:It_num1;
  
% % semilogy(pixl0,acc_WADMM,'--',pixl,acc_PWADMM,pixl0,acc_IPWADMM,pixl,acc_DADMM,'-.',pixl,acc_EXTRA,':',pixl,acc_COCA,'--',pixl,acc_DGD,'-.','LineWidth',1);
% % xlim([0,It_num1]);
% % hold on; 
% % xlabel('Iteration $k$','interpreter','latex')
% % ylabel('$\frac{1}{nN}\sum_{i}^{N}\|y_i-x_o \|^2 $','interpreter','latex')
% % % legend({'W-ADMM ($\beta=1.1$)','PW-ADMM ($\beta=1.1,\tau=0.8$)','D-ADMM ($\rho=0.1$)','EXTRA ($\alpha=0.1$)','DGD ($\alpha=0.01$)'},'interpreter','latex');
% % legend({'W-ADMM','PW-ADMM','IPW-ADMM','D-ADMM','EXTRA','COCA','DGD'},'interpreter','latex');
% % title('$N=100,\eta=0.3,K=100,n=2$','interpreter','latex')


pixl1 = K.*pixl;
pixli = K_i.*pixl;
Degree = 0;
for i=1:1:T
    Degree = Degree + length(B{i});
end
pixl2 = Degree.*pixl;

color = [
         0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
leg_s = {'sIADMM','COCA','DADMM','DGD','FAST ADMM','EXTRA'};
figure(2)

H=loglog(pixl0,acc_FsIADMM{1},pixl0, acc_sIADMM{1},'-. ','LineWidth',1.5);
%%
% subplot(2,1,1);
% colororder(color);
% 
% h=loglog(pixl0, acc_sIADMM{1},'-b',pixl2,acc_COCA{1},':r',pixl2,acc_DADMM{1},'-.r',...
%     pixl2,acc_DGD{1},'-.c',pixl0,acc_FsIADMM{1},'-g',pixl,acc_EXTRA{1},':m','LineWidth',2);
% legend(h,leg_s);hold on
% % semilogy(pixl0,acc_IADMM{1},':black',pixl0,acc_sIADMM{1},':r',pixl1,acc_PWADMM2{1},'--',...
% %     pixli,acc_PWADMM{1},'-.r',pixli,acc_IPWADMM{1},'blue','LineWidth',2);
% 
% subplot(2,1,2)
% semilogy(t_IADMM{1},acc_IADMM{1},':black',t_PWADMM2{1},acc_PWADMM2{1},'--',t_PWADMM{1},acc_PWADMM{1},'-.r',t_IPWADMM{1},acc_IPWADMM{1},'blue','LineWidth',2);
% % semilogy(pixl0,acc_WADMM{1},pixl1,acc_PWADMM{1});
 
% subplot(1,2,1)
% %%%%%%%%%%%%%% communication cost -- accuracy
% loglog(pixl0,acc_WADMM{1},'--',pixl1,acc_PWADMM{1},'black',pixli,acc_IPWADMM{1},'r',pixl2,acc_DADMM{1},'-.',pixl2,acc_EXTRA{1},':',pixl3{1},acc_COCA{1},'--',pixl2,acc_DGD{1},'-.','LineWidth',1);
% xlabel('communication cost','interpreter','latex')
% ylabel('accuracy','interpreter','latex')
% % legend({'W-ADMM ($\beta=1.1$)','PW-ADMM ($\beta=1.1,\tau=0.8$)','D-ADMM ($\rho=0.1$)','EXTRA ($\alpha=0.1$)','DGD ($\alpha=0.01$)' },'interpreter','latex');
% % legend({'W-ADMM','PW-ADMM','IPW-ADMM','D-ADMM$','EXTRA','COCA','DGD'},'interpreter','latex');
% title('$N=200,\eta=0.5,K=100,n=2$','interpreter','latex')
% 
% 
% subplot(1,2,2)
% %%%%%%%%%%%%%% running time -- accuracy
% semilogy(t_WADMM{1},acc_WADMM{1},'--',t_PWADMM{1},acc_PWADMM{1},'black',t_IPWADMM{1},acc_IPWADMM{1},'r',t_DADMM{1},acc_DADMM{1},'-.',t_EXTRA{1},acc_EXTRA{1},':',t_COCA{1},acc_COCA{1},'--',t_DGD{1},acc_DGD{1},'-.','LineWidth',1);
% xlabel('runing time $sec$','interpreter','latex')
% ylabel('accuracy','interpreter','latex')
% % legend({'W-ADMM','PW-ADMM','IPW-ADMM','D-ADMM','EXTRA','COCA','DGD'},'interpreter','latex');
% title('$N=200,\eta=0.5,K=100,n=2$','interpreter','latex')
% xlim([0 0.2]);
% 
% % hold on;
% %%
% % figure
% 
% %%%%%%%%%%%%%% communication cost -- accuracy
% % pixl0 = 1:1:It_num0;
% % pixl = 1:1:It_num1;
% % pixl1 = K.*pixl;
% % pixli = K_i.*pixl0;
% % Degree = 0;
% % for i=1:1:T
% %     Degree = Degree + length(B{i});
% % end
% % pixl2 = Degree.*pixl;
% % subplot(1,2,1)
% % loglog(pixl1,acc_PWADMM,'--black',pixli,acc_IPWADMM,'r','LineWidth',2);
xlabel('communication cost','interpreter','latex')
% % ylabel('$\frac{1}{nN}\sum_{i}^{N}\|y_i-x_o \|^2 $','interpreter','latex')
% % % legend({'W-ADMM ($\beta=1.1$)','PW-ADMM ($\beta=1.1,\tau=0.8$)','D-ADMM ($\rho=0.1$)','EXTRA ($\alpha=0.1$)','DGD ($\alpha=0.01$)' },'interpreter','latex');
% % legend({'PW-ADMM','IPW-ADMM'},'interpreter','latex');
title('$N=100,\eta=0.5,K=25 $','interpreter','latex')
% % hold on
% % 
% % subplot(1,2,2)
% % semilogy( t_PWADMM,acc_PWADMM,'--black',t_IPWADMM,acc_IPWADMM,'r','LineWidth',2);
xlabel('runing time $(s)$','interpreter','latex')
% % ylabel('$\frac{1}{nN}\sum_{i}^{N}\|y_i-x_o \|^2 $','interpreter','latex')
% % % legend({'W-ADMM','PW-ADMM','IPW-ADMM','D-ADMM','EXTRA','COCA','DGD'},'interpreter','latex');
% % legend({'PW-ADMM','IPW-ADMM'},'interpreter','latex');

% title('$N=50,\eta=0.1,K=50,n=2$','interpreter','latex')
ylabel('accuracy','interpreter','latex')

% % xlim([0 0.05]);
% % 
% % hold on
% 
% % end
% 
% x = 0:1:10;
% y = 100*ones(1,length(x));
% H = plot(x,y,x,y,x,y,x,y,x,y,x,y,x,y,x,y);
% legend(H,{'1','1','1','1','1','1','1','1'},'interpreter','latex')

% title('$N=50,\eta=0.3,K=50,n=2$','interpreter','latex')
 
% ylabel('accuracy','interpreter','latex')
% xlabel('iteration','interpreter','latex')
 
% ylabel('accuracy','interpreter','latex')
% xlabel('runing time (s)','interpreter','latex')


% x = 0:1:10;
% y = 100*ones(1,length(x));
% H = plot(x,y,x,y);
% legend(H,{'1','1'},'interpreter','latex')