function [mu, q] = convergence_analysis(x, x_star, qi)
    k = floor(length(x)*0.9);
    x_k = x(5:end-1);
    x_kp1 = x(6:end);
    x_km1 = x(4:end-2);
    x_km2 = x(3:end-3);
    q = real(log((x_kp1-x_k)./(x_k-x_km1))./log((x_k-x_km1)./(x_km1-x_km2)));
    if qi>0
        q = qi;
    end
    N_mu = x(2:end) - x_star*ones(1, length(x)-1);
    D_mu = x(1:end-1) - x_star*ones(1, length(x)-1);
    mu = N_mu./(D_mu.^q(end));
    
end