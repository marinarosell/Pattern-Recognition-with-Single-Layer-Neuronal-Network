function [xk, k] = uo_solve(x,f,g,epsG,kmax,almax,epsal,c1,c2,isd,icg,irc,nu,maxiter,isg_m,isg_al0,isg_k,tr_p)
    % isd : search direction (1=GM, 2=CGM, 3=BFGS, 7=SGM)
    % icg : CGM variant (1=FR, 2=PR+)
    % irc : Restart for the CGM (0= no restart, 1=(RC1), 2=(RC2)
    % nu : v (of RC2) 
        
    n = size(x,1);
    
    betak = 0;
    k = 1;
    xk=x;
    while norm(g(xk)) >= epsG && k <= kmax

        
        % Search direction
        
        if isd == 1 || (k == 1 && isd == 2)  % Gradient Method and 1st iteration of CGM and BFGS
            dk = -g(xk);
            

        elseif isd == 2 % Conjugate Gradient Method
            if irc == 1 && mod(k,n) == 0 % Restart Condition 1
                dk = -g(xk);
            elseif irc == 2 && abs(g(xk)'*g(xk))/norm(g(xk))^2 > nu % Restart Condition 2
                dk = -g(xk);
            else
                if icg == 1 % Fletcher-Reeves
                    betak = (g(xk)'*g(xk))/norm(g(xk_))^2;
                elseif icg == 2 % Polak-Ribière+
                    betak = max(0,(g(xk)'*(g(xk)-g(xk_))/norm(g(xk_))^2));
                end
                dk = -g(xk) + betak * dk;
            end
            
        elseif isd == 3 % BFGS Method
            if k==1 Hk = eye(n);
            else
                yk = g(xk) - g(xk_);
                sk = xk - xk_;
                rhok = 1 / (yk' * sk);
                Hk =  (eye(n) - rhok * sk * yk') * Hk * (eye(n) - rhok * yk * sk') + rhok * sk * sk';
            end
            dk = -Hk * g(xk);
            
        elseif isd == 7 % Stochastic Gradient Method    
            %dk = -1/(isg_m * tr_p) .* g(xk);
            dk = -g(xk);
        end
                
        
        % Step length
        
        if isd == 7 % Stochastic Gradient Method 
            alsg = 0.01 * isg_al0;
            ksg = floor(isg_k * kmax);
            if k > ksg 
                alk = alsg;
            else 
                alk = (1-k/ksg) * isg_al0 + k/ksg * alsg;
            end
        else
            [alk,iout] = uo_BLSNW32(f,g,xk,dk,almax,c1,c2,maxiter,epsal);
        end
        % Update variables
        
        k = k+1;
        xk_ = xk;
        xk = xk+alk*dk;
    end
    

end