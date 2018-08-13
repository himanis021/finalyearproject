function [mu, u, X, S, B, i] = CVXB( I, u, lambda, alpha, beta, gamma, rho, tau_u, tau_L, type )
% Joint Image segmentation, artifact detection and bias correction
% [mu, u, X, S, B] = CVXB( I, u, lambda, alpha, beta, gamma, rho, tau_u, tau_L, type )
%
% Input
% -----
% I         input image (MxN) for median filter
% u         segmentation initialization
% lambda    CV-model lambdas (2x1)
% alpha     weight of bias field smoothness
% beta      weight of interface regularization
% gamma     artifact classification cutoff
% rho       image formation model constraint penalty weight
% tau_u     time-step of MBO scheme
% tau_L     time-step of dual ascent / Lagrangian update
% type      model selector: 'CV', 'CVX', 'CVB', 'CVXB'
%
% Output
% ------
% mu        estimated region means (2x1)
% u         phase-field (MxN)
% X         artifacts (MxN)
% S         structure (MxN)
% B         bias field (MxN)
% i         number of iterations run (for convergence analysis)

% inits
[M,N] = size(I);

mu = zeros(2,1); % region statistics
X = zeros(M,N); % artifacts class
B = zeros(M,N); % or something smarter, like extreme LP-filtering, below

S = I-B;
L = zeros(M,N); % Lagrangian multiplier

% spectral median filters:
[XX,YY] = meshgrid((1:N)/N, (1:M)/M);
fx = 1/N;
fy = 1/M;
freqs_1 = XX - 0.5 - fx;
freqs_2 = YY - 0.5 - fy;
BfiltDCT = rho + alpha*(XX.^2 + YY.^2);
MBOfilt = 1 + beta*tau_u*(freqs_1.^2 + freqs_2.^2);

% if bias decomposition is used, then init through strong LP-filtering
if any(strcmp( type, {'CVB','CVXB'} ))
    B = real( idct2( dct2( I ) ./ BfiltDCT ./ BfiltDCT) );
    S = I-B;
end


for i = 1:200 % let's do some fixed number of iterations, for now
    
    % swap regions so "object" is smaller than "background"
    if sum(u(:)) > 0.5*numel(u)
        u = 1-u;
    end
    
    % mu-step  /  statistics update
    for region = 1:2
        mu(region) = sum( (1-X(:)) .* S(:) .* (u(:) == (2-region)) ) / sum( (1-X(:)) .* (u(:) == (2-region)) );
        % Tricks:   for 'CV' and 'CVX', S == I
        %           for 'CV' and 'CVB', X == 0
    end
    
    %%%%%%%%%%%%%%%%%
    
    uold = u; % saved for convergence criterion
    
    % u-step 1 : Data-step
    u = u + tau_u * (1 - X ) .* ( - lambda(1)*(S - mu(1)).^2 + lambda(2)*(S - mu(2)).^2 );
    
    % u-step 2 : Diffusion
    u = real( ifft2( ifftshift( fftshift( fft2( u ) ) ./ MBOfilt ) ) );
    
    % u-step 3 : Thresholding
    u = 1 .* (u > 0.5);
    
    %%%%%%%%%%%%%%%%%
    
    % X-step  /  artifacts detection
    if any(strcmp( type, {'CVX', 'CVXB'} ))
        X = 1 * ( ( lambda(1) * u .* (mu(1) - S).^2 + lambda(2) * (1-u) .* (mu(2) - S).^2 ) > gamma );
    else
        X = zeros(M,N);
    end
    
    %%%%%%%%%%%%%%%%%
    
    if any(strcmp( type, {'CVB', 'CVXB'} ))
        % B-step  /  bias field update
        B = real( idct2( dct2( rho*( I - S ) + 0.5*L ) ./ BfiltDCT ) );
        
        % S-step  /  structure update
        S = ( (1-X) .* (lambda(1)*mu(1)*u + lambda(2)*mu(2)*(1-u)) + rho*(I - B) + 0.5*L  ) ./ ( (1-X) .* (lambda(1)*u + lambda(2)*(1-u)) + rho );
        
        % L-step  /  Lagrangian update, dual ascent
        L = L + tau_L*(I - B - S);
    else
        B = zeros(M,N);
        S = I;
        L = zeros(M,N);
    end
    
    %%%%%%%%%%%%%%%%%
        
    % early stopping if segmentation has frozen, but minimum 10 iterations
    if (i>10) & all(uold == u) %#ok<AND2>
        break;
    end
end