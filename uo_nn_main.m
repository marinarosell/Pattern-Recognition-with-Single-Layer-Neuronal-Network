clear;
%
% Parameters for dataset generation
%
num_target =[3 8];
tr_freq    = .5;        
tr_p       = 250;       
te_q       = 250;       
tr_seed    = 49784363;    
te_seed    = 54409254;    
%
% Parameters for optimization
%
la = 1.0;                                                     % L2 regularization.
epsG = 10^-6; kmax = 10000;                                   % Stopping criterium.
ils=3; ialmax = 2; kmaxBLS=30; epsal=10^-3;c1=0.01; c2=0.45;  % Linesearch.
isd = 3; icg = 2; irc = 0 ; nu = 1.0;                         % Search direction.
isg_m = 0.05; isg_al0=2; isg_k=0.3;                           % stochastic gradient
%
% Optimization
%
t1=clock;
[Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target, tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,isg_m,isg_al0,isg_k,icg,irc,nu);
t2=clock;
fprintf(' wall time = %6.1d s.\n', etime(t2,t1));
fprintf('%6d %6d \n', tr_acc, te_acc);
uo_nn_Xyplot(Xte,yte,wo);
%
% Output
%


