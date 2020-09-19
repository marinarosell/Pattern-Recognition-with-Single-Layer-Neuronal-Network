function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target, tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,isg_m,isg_al0,isg_k,icg,irc,nu)
tic;    

    % Generate the trainning data set
    [Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
    
    %Find wo minimizing the loss function
    sig = @(X) 1./(1+exp(-X));
    y = @(X,w) sig(w'*sig(X));
    
    if isd == 7
        Xtrs = []; ytrs = []; repes = false(1, tr_p);
        while size(Xtrs,2) < round(isg_m * tr_p)
            index = round(rand()*tr_p);
            if repes(index) == 0
                Xtrs = [Xtrs, Xtr(:,index)];
                ytrs = [ytrs, ytr(index)];
            end
        end
        L = @(w) norm(y(Xtrs,w)-ytrs)^2 + (la*norm(w)^2)/2;
        gL = @(w) 2*sig(Xtrs)*((y(Xtrs,w)-ytrs).*y(Xtrs,w).*(1-y(Xtrs,w)))'+la*w;
        x=zeros(35,1);
        [wo, niter] = uo_solve(x,L,gL,epsG,kmax,ialmax,epsal,c1,c2,isd,icg,irc,nu,kmaxBLS,isg_m,isg_al0,isg_k,tr_p);
        fo = L(wo);
    else
        L = @(w) norm(y(Xtr,w)-ytr)^2 + (la*norm(w)^2)/2;
        gL = @(w) 2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))'+la*w;
        x=zeros(35,1);
        [wo, niter] = uo_solve(x,L,gL,epsG,kmax,ialmax,epsal,c1,c2,isd,icg,irc,nu,kmaxBLS,isg_m,isg_al0,isg_k,tr_p);
        fo = L(wo);
    end
    
    
    % Calculate training accuracy
    suma = 0;
    results = y(Xtr,wo);
    for i = 1:250
        if round(results(i)) == ytr(i)
            suma = suma + 1;
        end
    end
    
    tr_acc = 100/tr_p*suma;
    
    
    
    % Generate the testing data set
    [Xte,yte] = uo_nn_dataset(te_seed, te_q, num_target, tr_freq);
    
    % Calculate testing set accuracy
    suma = 0;
    results = y(Xte,wo);
    for i = 1:250
        if round(results(i)) == yte(i)
            suma = suma + 1;
        end
    end
    
    te_acc = 100/te_q*suma;
    
    tex = toc;
end
